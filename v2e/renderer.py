import numpy as np
import cv2
import os
import atexit
import logging
from tqdm import tqdm
from typing import List
from engineering_notation import EngNumber  # only from pip
from enum import Enum

from v2e.emulator import EventEmulator
from v2e.v2e_utils import video_writer,all_images,read_image,checkAddSuffix

logger=logging.getLogger(__name__)

class ExposureMode(Enum):
    DURATION=1
    COUNT=2
    AREA_COUNT=3

class EventRenderer(object):
    """
    Class for DVS rendering from events and by generating DVS from image sequence.
    It only defines the video and event dataset output path and whether to rotate the images.

    @author: Zhe He
    @contact: hezhehz@live.cn
    """

    def __init__(
            self,
            full_scale_count=3,
            output_path:str=None,
            dvs_vid:str=None,
            preview:bool=False,
            exposure_mode:ExposureMode=ExposureMode.DURATION, # 'count', 'area-count'
            exposure_value=1/300.0,
            area_dimension=None,
            frame_times_suffix = '-frame_times.txt'  # suffix using dvs_vid file name for the frame times when not using constant_time
    ):
        """ Init.

        Parameters
        ----------
        frame_rate_hz: float
            frame rate of video; events will be integrated for 1/frame_rate_hz
        output_path: str,
            path of folder to hold output video
        dvs_vid: str or None, str name of video, e.g. dvs.avi
            else None.
        rotate180: bool,
            True to rotate the output frames 180 degrees.
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
        self.exposure_value=exposure_value
        self.output_path = output_path
        self.width = None
        self.height = None  # must be set by specific renderers, which might only know it once they have data
        self.full_scale_count=full_scale_count
        self.accum_mode='duration' # 'duration', 'count', 'area_count'
        self.dvs_frame_times_suffix=frame_times_suffix # suffix using dvs_vid file name for the frame times when not using constant_time
        self.frame_rate_hz=None
        self.event_count=None
        self.frameIntevalS = None

        self.area_counts=None # 2d array of counts
        self.area_count=None
        self.area_dimension=area_dimension
        if self.exposure_mode==ExposureMode.DURATION:
            self.frame_rate_hz=1/self.exposure_value
            self.frameIntevalS = 1 / self.frame_rate_hz
        elif self.exposure_mode==ExposureMode.COUNT:
            self.event_count=self.exposure_value
        elif self.exposure_mode==ExposureMode.AREA_COUNT:
            self.area_count=self.exposure_value
        else:
            raise('exposure mode must be either duration or count')

        self.video_output_file_name = dvs_vid
        self.video_output_file = None
        self.frame_times_output_file=None

        self.emulator = None
        self.preview=preview
        self.preview_resized=False # flag to keep from sizing the preview
        self.numFramesWritten=0
        atexit.register(self.cleanup)

        self.currentFrameStartTime=None # initialized with first events
        self.currentFrame=None # initialized with first events, then new frames are built according to events
        # this frame is what we are currently accumulating to. It is saved between event packets passed to us


    def cleanup(self):
        if self.video_output_file and (type(self.video_output_file) is not str): #todo not sure this is correct check for str type
            logger.info("Closing DVS video output file after writing {} frames".format(self.numFramesWritten))
            if type(self.video_output_file) is not str: self.video_output_file.release()
            if self.frame_times_output_file is not None: self.frame_times_output_file.close()
            cv2.destroyAllWindows()


    def _check_outputs_open(self):
        '''checks that output video and event datasets files are open'''

        if self.video_output_file is not None: return

        if not self.height or not self.width:
            raise ValueError('height and width not set for output video')

        if self.output_path is None and self.video_output_file is str:
            logger.warning('output_path is None; will not write DVS video')

        if self.output_path and type(self.video_output_file_name) is str:
            fn=checkAddSuffix(os.path.join(self.output_path, self.video_output_file_name),'.avi')
            logger.info('opening DVS video output file ' + fn)
            self.video_output_file = video_writer(fn, self.height, self.width)
            fn=checkAddSuffix(os.path.join(self.output_path, self.video_output_file_name),self.dvs_frame_times_suffix)
            logger.info('opening DVS frame times file ' + fn)
            self.frame_times_output_file=open(fn,'w')
            s='# frame times for {}\n# frame# time(s)\n'.format(self.video_output_file_name)
            self.frame_times_output_file.write(s)

    def render_events_to_frames(self, event_arr: np.ndarray, height: int, width: int)->np.ndarray:
        """ Incrementally render event frames.

        Frames are appended to the video output file.
        The current frame is held for the next packet to fill.
        Only frames that have been filled are returned.

        Frames are filled when an event comes that is past the end of the frame duration.
        These filled frames are returned.

        Parameters
        ----------
        event_arr:np.ndarray
            [n,4] consisting of n events each with [ts,y,x,pol], ts are in float seconds
        height: height of output video in pixels; events are hisogramed to this width in pixels. I.e. if input has 100 pixels and height is 30 pixels, roughly 3 pixels will be collected to one output pixel
        width: width of output video in pixels

        Returns
        -------
        rendered frames from these events, or None if no new frame was filled. Frames are np.ndarray with [n,h,w] shape, where n is frame, h is height, and w is width
        """
        self.width = width
        self.height = height

        self._check_outputs_open()

        if event_arr is None or event_arr.shape[0]==0:
            logger.info('event_arr is None or there are no events, doing nothing')
            return None

        ts=event_arr[:,0]
        if self.exposure_mode==ExposureMode.DURATION:
            if self.currentFrameStartTime is None:
                self.currentFrameStartTime=ts[0] # initialize this frame

            nextFrameStartTs=self.currentFrameStartTime+self.frameIntevalS

        if self.exposure_mode==ExposureMode.AREA_COUNT and self.area_counts is None:
            nw=1+self.width//self.area_dimension
            nh=1+self.height//self.area_dimension
            self.area_counts=np.zeros(shape=(nw,nh),dtype=int)


        returnedFrames=None # accumulate frames here

        # loop over events, creating new frames as needed, until we get to frame for last event.
        # then hold this frame and its start time until we get more events to fill it

        thisFrameIdx=0 # start at first event
        numEvents=len(ts)
        histrange = [(0, v) for v in (self.height, self.width)]

        done=False
        while not done: # continue consuming events from input event_arr until we are done with all these events, output frames along the way
            # try to get events for current frame
            if self.exposure_mode==ExposureMode.DURATION:
                # find first event that is after the current frames start time
                start=np.searchsorted(ts[thisFrameIdx:],self.currentFrameStartTime,side='left')
                # find last event that fits within current frame
                end=np.searchsorted(ts,nextFrameStartTs,side='right')
                # if the event is after next frame start time, then we finished current frame and can append it to output list
            elif self.exposure_mode==ExposureMode.COUNT:
                start=thisFrameIdx
                end=start+self.event_count
            elif self.exposure_mode==ExposureMode.AREA_COUNT:
                start=thisFrameIdx
                # brute force, iterate over events to determine end
                for end,ev in enumerate(event_arr, start=start):
                    x=int(ev[1]//self.area_dimension)
                    y=int(ev[2]//self.area_dimension)
                    count=1+self.area_counts[x,y]
                    self.area_counts[x,y]=count
                    if count>=self.area_count:
                        self.area_counts=np.zeros_like(self.area_counts)
                        break

            if end >= numEvents:
                done = True  # we will return now after integrating remaining events
                end = numEvents - 1  # reset to end of current events to integrate all of them

            events = event_arr[start: end] # events in this frame

            pol_on = (events[:, 3] == 1)
            pol_off = np.logical_not(pol_on)
            img_on, _, _ = np.histogram2d(
                events[pol_on, 2], events[pol_on, 1],
                bins=(self.height, self.width), range=histrange)
            img_off, _, _ = np.histogram2d(
                events[pol_off, 2], events[pol_off, 1],
                bins=(self.height, self.width), range=histrange)
            if self.currentFrame is None:
                self.currentFrame=np.zeros_like(img_on) # make a new empty frame

            # accumulate event histograms to the current frame, clip values of zero-centered current frame with new events added
            self.currentFrame = np.clip(self.currentFrame+(img_on - img_off), -self.full_scale_count, self.full_scale_count)

            if not done: # if not finished with current event_arr, write out the completed frame
                # we finished a frame above, but we will continue to accumulate remaining events after writing out current frame
                if self.exposure_mode == ExposureMode.DURATION:
                    self.currentFrameStartTime+=self.frameIntevalS # increase time to next frame
                    nextFrameStartTs=self.currentFrameStartTime+self.frameIntevalS
                elif self.exposure_mode == ExposureMode.COUNT or self.exposure_mode==ExposureMode.AREA_COUNT:
                    thisFrameIdx=end

                # img output is 0-1 range
                img = (self.currentFrame + self.full_scale_count) / float(self.full_scale_count * 2)
                self.currentFrame=None # done with this frame, allocate new one in next loop
                if (returnedFrames is not None):
                    returnedFrames = np.concatenate((returnedFrames, img[np.newaxis,...])) # put new frame under previous ones
                else:
                    returnedFrames=img[np.newaxis,...] # add axis at position zero for the frames

                if self.video_output_file:
                    self.video_output_file.write(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
                    t=None
                    if self.exposure_mode==ExposureMode.COUNT or self.exposure_mode==ExposureMode.AREA_COUNT:
                        t=(ts[start]+ts[end])/2
                    else:
                        t=self.currentFrameStartTime+self.frameIntevalS/2
                    self.frame_times_output_file.write('{}\t{:10.6f}\n'.format(self.numFramesWritten,t))
                    self.numFramesWritten+=1
                if self.preview:
                    name=str(self.video_output_file_name)
                    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
                    cv2.imshow(name,img)
                    if not self.preview_resized:
                        cv2.resizeWindow(name, 800, 600)
                        self.preview_resized = True
                    cv2.waitKey(30) # 30 hz playback


        return returnedFrames



    def generateEventsFromFramesAndExportEventsToHDF5(self, outputFileName: str, imageFileNames: List[str], frameTimesS: np.array) -> None:
        """Export events to a HDF5 file.

        Parameters
        ----------
        outputFileName : str
            file name of the HDF5 file
        """

        event_dataset = self.event_file.create_dataset(
            name="event",
            shape=(0, 4),
            maxshape=(None, 4),
            dtype="uint32")


        # generating events
        num_events = 0
        for i in tqdm(range(frameTimesS.shape[0] - 1),
                      desc="export_events: ", unit='fr'):
            new_frame = read_image(imageFileNames[i + 1])
            if self.emulator is None:
                self.emulator = EventEmulator(pos_thres=self.pos_thres, neg_thres=self.neg_thres, sigma_thres=self.sigma_thres)
            tmp_events = self.emulator.generate_events(
                new_frame,
                frameTimesS[i],
                frameTimesS[i + 1]
            )
            if tmp_events is not None:
                # convert data to uint32 (microsecs) format
                tmp_events[:, 0] = tmp_events[:, 0] * 1e6
                tmp_events[tmp_events[:, 3] == -1, 3] = 0
                tmp_events = tmp_events.astype(np.uint32)

                # save events
                event_dataset.resize(
                    event_dataset.shape[0] + tmp_events.shape[0],
                    axis=0)

                event_dataset[-tmp_events.shape[0]:] = tmp_events
                self.event_file.flush()

                num_events += tmp_events.shape[0]

        logger.info("Generated {} events".format(EngNumber(num_events)))