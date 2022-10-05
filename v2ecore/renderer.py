import numpy as np
import cv2
import os
import atexit
import logging
from tqdm import tqdm
from typing import List
from engineering_notation import EngNumber  # only from pip
from enum import Enum
from numba import jit, njit

from v2ecore.emulator import EventEmulator
from v2ecore.v2e_utils import video_writer, read_image, checkAddSuffix, v2e_quit
from v2ecore.v2e_utils import hist2d_numba_seq

logger = logging.getLogger(__name__)


class ExposureMode(Enum):
    DURATION = 1
    COUNT = 2
    AREA_COUNT = 3
    SOURCE = 4


class EventRenderer(object):
    """Class for DVS rendering from events.
    and by generating DVS from image sequence.

    It only defines the video and event dataset output path
    and whether to rotate the images.

    @author: Zhe He
    @contact: hezhehz@live.cn
    """

    def __init__(
            self,
            full_scale_count=3,
            output_path=None,
            dvs_vid=None,
            preview=False,
            exposure_mode=ExposureMode.DURATION,  # 'count', 'area-count'
            exposure_value=1 / 300.0,
            area_dimension=None,
            # suffix using dvs_vid file name for the frame times
            # when not using constant_time
            frame_times_suffix='-frame_times.txt',
            avi_frame_rate=30):
        """ Init.

        Parameters
        ----------
        output_path: str,
            path of folder to hold output video
        dvs_vid: str or None, str name of video, e.g. dvs.avi
            else None.
        full_scale_count:int,
            full scale black/white DVS event count value
        exposure_mode: ExposureMode,
            mode to finish exposure of DVS frames
        exposure_value: Numeric,
            either float duration in seconds or int count
        area_dimension: int,
            size of area_count in pixels in output space
        preview: bool
            show preview in cv2 window
        """
        self.exposure_mode = exposure_mode
        self.exposure_value = exposure_value
        self.output_path = output_path
        # must be set by specific renderers,
        # which might only know it once they have data
        self.width = None
        self.height = None
        self.full_scale_count = full_scale_count
        self.accum_mode = 'duration'  # 'duration', 'count', 'area_count'
        # suffix using dvs_vid file name for the frame times
        # when not using constant_time
        self.dvs_frame_times_suffix = frame_times_suffix
        self.frame_rate_hz = None
        self.event_count = None
        self.frameIntevalS = None
        self.avi_frame_rate = avi_frame_rate

        self.area_counts = None  # 2d array of counts
        self.area_count = None
        self.area_dimension = area_dimension
        if self.exposure_mode == ExposureMode.DURATION:
            self.frame_rate_hz = 1 / self.exposure_value
            self.frameIntevalS = 1 / self.frame_rate_hz
        elif self.exposure_mode == ExposureMode.COUNT:
            self.event_count = int(self.exposure_value)
        elif self.exposure_mode == ExposureMode.AREA_COUNT:
            self.area_count = int(self.exposure_value)
        elif self.exposure_mode==ExposureMode.SOURCE:
            pass
        else:
            raise (f'exposure mode {self.exposure_mode} is unknown; must be duration, count, or area-count')

        self.video_output_file_name = dvs_vid
        self.video_output_file = None
        self.frame_times_output_file = None

        self.emulator = None
        self.preview = preview
        self.preview_resized = False  # flag to keep from sizing the preview
        self.numFramesWritten = 0
        atexit.register(self.cleanup)

        self.currentFrameStartTime = None  # initialized with first events
        # initialized with first events, then new frames are built
        # according to events
        self.currentFrame = None
        # this frame is what we are currently accumulating to.
        # It is saved between event packets passed to us

        self.printed_empty_packet_warning = False

    def cleanup(self):
        if self.video_output_file is not None:
            logger.info(
                "Closing DVS video output file {} "
                "after writing {} frames".format(self.video_output_file_name, self.numFramesWritten))
            if type(self.video_output_file) is not str:
                self.video_output_file.release()
            if self.frame_times_output_file is not None:
                self.frame_times_output_file.close()
            cv2.destroyAllWindows()

    def _check_outputs_open(self):
        """checks that output video and event datasets files are open"""

        if self.video_output_file is not None:
            return

        if not self.height or not self.width:
            raise ValueError('height and width not set for output video')

        if self.output_path is None and self.video_output_file is str:
            logger.warning('output_folder is None; will not write DVS video')

        if self.output_path and type(self.video_output_file_name) is str:
            fn = checkAddSuffix(
                os.path.join(self.output_path,
                             self.video_output_file_name), '.avi')
            logger.info('opening DVS video output file ' + fn)
            self.video_output_file = video_writer(
                fn, self.height, self.width,
                frame_rate=self.avi_frame_rate)
            fn = checkAddSuffix(
                os.path.join(
                    self.output_path, self.video_output_file_name),
                self.dvs_frame_times_suffix)
            logger.info('opening DVS frame times file ' + fn)
            self.frame_times_output_file = open(fn, 'w')
            s = '# frame times for {}\n# frame# time(s)\n'.format(
                self.video_output_file_name)
            self.frame_times_output_file.write(s)

    def render_events_to_frames(self, event_arr: np.ndarray,
                                height: int, width: int,
                                return_frames=False) -> np.ndarray:
        """ Incrementally render event frames.

        Frames are appended to the video output file.
        The current frame is held for the next packet to fill.
        Only frames that have been filled are returned.

        Frames are filled when an event comes
        that is past the end of the frame duration.
        These filled frames are returned.

        Parameters
        ----------
        event_arr:np.ndarray
            [n,4] consisting of n events each with [ts,x,y,pol],
            ts are in float seconds
        height: height of output video in pixels;
            events are histogramed to this width in pixels.
            I.e. if input has 100 pixels and height is 30 pixels,
            roughly 3 pixels will be collected to one output pixel
        width: width of output video in pixels
        return_frames: return Frames if True, return None otherwise

        Returns
        -------
        rendered frames from these events, or None if no new frame was filled.
        Frames are np.ndarray with [n,h,w] shape,
        where n is frame, h is height, and w is width
        """
        self.width = width
        self.height = height

        self._check_outputs_open()

        if event_arr is None or event_arr.shape[0] == 0:
            if not self.printed_empty_packet_warning:
                logger.info(
                    'event_arr is None or there are no events, '
                    'doing nothing, supressing further warnings')
                self.printed_empty_packet_warning = True
            return None

        ts = event_arr[:, 0]
        if self.exposure_mode == ExposureMode.DURATION:
            if self.currentFrameStartTime is None:
                self.currentFrameStartTime = ts[0]  # initialize this frame

            nextFrameStartTs = self.currentFrameStartTime + self.frameIntevalS

        if self.exposure_mode == ExposureMode.AREA_COUNT and \
                self.area_counts is None:
            nw = 1 + self.width // self.area_dimension
            nh = 1 + self.height // self.area_dimension
            self.area_counts = np.zeros(shape=(nw, nh), dtype=int)

        returnedFrames = None  # accumulate frames here

        # loop over events, creating new frames as needed,
        # until we get to frame for last event.
        # then hold this frame and its start time
        # until we get more events to fill it

        thisFrameIdx = 0  # start at first event
        numEvents = len(ts)
        histrange = np.asarray([(0, v) for v in (self.height, self.width)],
                               dtype=np.int64)

        doneWithTheseEvents = False

        # continue consuming events from input event_arr
        # until we are done with all these events,
        # output frames along the way

        #  @jit("UniTuple(int32, 2)(float64[:], float64, float64)",
        #       nopython=True)
        @jit(nopython=True)
        def search_duration_idx(ts, curr_start, next_start):
            start = np.searchsorted(ts, curr_start, side="left")
            end = np.searchsorted(ts, next_start, side="right")
            return start, end

        #  @jit("float64[:, :](float64[:, :], int32)",
        #       fastmath=True, nopython=True)
        @jit(nopython=True)
        def normalize_frame(curr_frame, full_scale_count):
            return (curr_frame + full_scale_count) / float(
                full_scale_count * 2)

        # @jit("Tuple((int64[:, :], int64))(float64[:, :], int64[:, :], "
        #      "int64, int64, int64)", nopython=True)
        @jit(nopython=True)
        def compute_area_counts(events, area_counts,
                                area_count, area_dimension, start):
            #  new_area_counts = np.copy(area_counts)
            ev_idx = start
            for ev_idx in range(start, events.shape[0]):
                x = int(events[ev_idx, 1] // area_dimension)
                y = int(events[ev_idx, 2] // area_dimension)
                count = 1 + area_counts[x, y]
                area_counts[x, y] = count
                if count >= area_count:
                    area_counts = np.zeros_like(area_counts)
                    break

            return area_counts, ev_idx

        start, end = 0, numEvents

        self.currentFrame = None  # filled by accummulate_events

        while not doneWithTheseEvents:
            # try to get events for current frame
            if self.exposure_mode == ExposureMode.DURATION:
                # find first event that is after the current frames start time
                start, end = search_duration_idx(
                    ts[thisFrameIdx:], self.currentFrameStartTime,
                    nextFrameStartTs)
                # if the event is after next frame start time,
                # then we finished current frame and can
                # append it to output list
            elif self.exposure_mode == ExposureMode.COUNT:
                start = thisFrameIdx
                end = start + self.event_count
            elif self.exposure_mode == ExposureMode.AREA_COUNT:
                start = thisFrameIdx
                # brute force, iterate over events to determine end
                self.area_counts, end = compute_area_counts(
                    event_arr, self.area_counts, self.area_count,
                    self.area_dimension, start)
            elif self.exposure_mode == ExposureMode.SOURCE:
                start = 0
                end = numEvents

            if end >= numEvents - 1:
                # we will return now after integrating remaining events
                doneWithTheseEvents = True
                # reset to end of current events to integrate all of them
                end = numEvents - 1

            events = event_arr[start:end]  # events in this frame
            # accumulate event histograms to the current frame,
            # clip values of zero-centered current frame with new events added
            self.accumulate_event_frame(events, histrange)

            # If not finished with current event_arr,
            # it means above we finished filling a frame, either with
            # time or with sufficient count of events.
            # Write out the completed frame
            if not doneWithTheseEvents or self.exposure_mode==ExposureMode.SOURCE:
                # we finished a frame above, but we will continue to
                # accumulate remaining events after writing out current frame
                if self.exposure_mode == ExposureMode.DURATION:
                    # increase time to next frame
                    self.currentFrameStartTime += self.frameIntevalS
                    nextFrameStartTs = self.currentFrameStartTime + \
                                       self.frameIntevalS
                elif self.exposure_mode == ExposureMode.COUNT or \
                        self.exposure_mode == ExposureMode.AREA_COUNT:
                    thisFrameIdx = end
                elif self.exposure_mode==ExposureMode.SOURCE:
                    pass

                # img output is 0-1 range
                img = normalize_frame(self.currentFrame, self.full_scale_count)

                # done with this frame, allocate new one in next loop
                self.currentFrame = None

                if return_frames:
                    returnedFrames = np.concatenate(
                        (returnedFrames, img[np.newaxis, ...])) \
                        if returnedFrames is not None else \
                        img[np.newaxis, ...]

                if self.video_output_file:
                    self.video_output_file.write(
                        cv2.cvtColor((img * 255).astype(np.uint8),
                                     cv2.COLOR_GRAY2BGR))
                    t = None

                    if self.exposure_mode==ExposureMode.SOURCE:
                        t=ts[0] if len(ts)>0 else float('nan')
                    else:
                        exposure_mode_cond = (
                                self.exposure_mode == ExposureMode.COUNT or
                                self.exposure_mode == ExposureMode.AREA_COUNT)
                        t = (ts[start] + ts[end]) / 2 if exposure_mode_cond else \
                            self.currentFrameStartTime + self.frameIntevalS / 2

                    self.frame_times_output_file.write(
                        '{}\t{:10.6f}\n'.format(self.numFramesWritten, t))
                    self.numFramesWritten += 1
                if self.preview:
                    name = str(self.video_output_file_name)
                    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                    cv2.imshow(name, img)
                    if not self.preview_resized:
                        cv2.resizeWindow(name, 800, 600)
                        self.preview_resized = True
                    k=cv2.waitKey(30)
                    if k==27 or k==ord('x'):
                        v2e_quit()

        return returnedFrames

    def accumulate_event_frame(self, events, histrange):
        """Accumulate event frame from an array of events.

        # Arguments
        events: np.ndarray
            an [N events x 4] array

        # Returns
        event_frame: np.ndarray
            an event frame
        """
        pol_on = (events[:, 3] == 1)
        pol_off = np.logical_not(pol_on)

        img_on = hist2d_numba_seq(
            np.array([events[pol_on, 2], events[pol_on, 1]],
                     dtype=np.float64),
            bins=np.asarray([self.height, self.width], dtype=np.int64),
            ranges=histrange)
        img_off = hist2d_numba_seq(
            np.array([events[pol_off, 2], events[pol_off, 1]],
                     dtype=np.float64),
            bins=np.asarray([self.height, self.width], dtype=np.int64),
            ranges=histrange)

        if self.currentFrame is None:
            self.currentFrame = np.zeros_like(img_on)

        # accumulate event histograms to the current frame,
        # clip values of zero-centered current frame with new events added
        self.currentFrame = np.clip(
            self.currentFrame + (img_on - img_off),
            -self.full_scale_count, self.full_scale_count)
