import numpy as np
import cv2
import os
import glob
import h5py
import atexit
import logging
from tqdm import tqdm
from typing import List
from engineering_notation import EngNumber  # only from pip

from src.emulator import EventEmulator
from src.v2e_utils import video_writer,all_images,read_image


class EventRenderer(object):
    """
    Class for DVS rendering from events and by generating DVS from image sequence.
    It only defines the video and event dataset output path and whether to rotate the images.

    @author: Zhe He
    @contact: hezhehz@live.cn
    """
    rotate: bool

    def __init__(
            self,
            video_path: str,
            pos_thres=0.2,
            neg_thres=0.2,
            sigma_thres=0.03,
            event_path=None,
            rotate=False,
    ):
        """ Init.

        Parameters
        ----------
        video_path: str, path of output video. Example: ../../XX.avi.
        event_path: str or None, str if the events need to be saved \
            else None.
        rotate: bool, True to rotate the output frames 90 degrees.
        """
        self.video_path = video_path
        self.rotate = rotate
        self.event_path = event_path
        self.width = None
        self.height = None  # must be set by specific renderers, which might only know it once they have data
        self.event_file = None
        self.video_output_file = None
        self.emulator = None
        # EventEmulator(base_frame=None, pos_thres=pos_thres, neg_thres=neg_thres, sigma_thres=sigma_thres)  # base frame initialized on first use

        atexit.register(self.cleanup)

    def cleanup(self):
        logging.info("Closing video and event dataset files...")
        if self.video_output_file: self.video_output_file.release()
        if self.event_file: self.event_file.close()

    def _check_outputs_open(self):
        '''checks that output video and event datasets files are open'''
        if not self.height or not self.width:
            raise Exception('height and width not set for output video')

        if not self.video_path:
            logging.warning('video_path not set')

        if self.video_path:
            self.video_output_file = video_writer(os.path.join(self.video_path, "v2e-dvs.avi"), self.height, self.width)
        if self.event_path:
            logging.info('opening event output dataset file ' + self.event_path)
            self.event_file = h5py.File(os.path.join(self.event_path,"v2e-events.h5"), "w")

    def renderEventsToFrames(self, event_arr: np.ndarray, height: int, width: int, frame_ts: np.array, full_scale_count=3):
        """ Incrementally render event frames, where events come from overridden method _get_events().
        Frames are appended to the video output file.

        Parameters
        ----------
        height: height of output video in pixels
        width: width of output video in pixels
        frame_ts: np.array, timestamps of output frames, for real events.
        interpolated_ts: np.array, timestamps of interpolated video, for generated events
        full_scale_count: int, count of DVS ON and OFF events per pixel for full white and black
        Returns
        -------
        rendered_frames: np.ndarray, rendered event frames.
        """
        self.width = width
        self.height = height
        self.frame_ts = frame_ts

        self._check_outputs_open()

        if event_arr is None:
            logging.info('event_arr is None, doing nothing')
            return None

        if self.event_path:
            np.save(self.event_path, event_arr)
            logging.info("events saved to " + str(self.event_path))

        histrange = [(0, v) for v in (self.height, self.width)]
        rendered_frames = list()

        for ts_idx in range(self.frame_ts.shape[0] - 1):
        # for ts_idx in tqdm(range(self.frame_ts.shape[0] - 1),
        #                    desc="rendering DVS histograms: ", unit='fr'):
            # assume time_list is sorted.

            start = np.searchsorted(event_arr[:, 0],
                                    self.frame_ts[ts_idx],
                                    side='left')
            end = np.searchsorted(event_arr[:, 0],
                                  self.frame_ts[ts_idx + 1],
                                  side='right')
            # select events, assume that pos_list is sorted
            if ts_idx < len(self.frame_ts) - 1:
                events = event_arr[start: end]
            else:
                events = event_arr[start:]

            pol_on = (events[:, 3] == 1)
            pol_off = np.logical_not(pol_on)
            img_on, _, _ = np.histogram2d(
                events[pol_on, 2], events[pol_on, 1],
                bins=(self.height, self.width), range=histrange)
            img_off, _, _ = np.histogram2d(
                events[pol_off, 2], events[pol_off, 1],
                bins=(self.height, self.width), range=histrange)
            if full_scale_count is not None:
                integrated_img = np.clip(
                    (img_on - img_off), -full_scale_count, full_scale_count)
            else:
                integrated_img = (img_on - img_off)
            rendered_frames.append(integrated_img)
            img = (integrated_img + full_scale_count) / float(full_scale_count * 2)

            if self.rotate:
                img = np.rot90(img, k=2)

            self.video_output_file.write(
                cv2.cvtColor(
                    (img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
            if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
                break

        rendered_frames = np.vstack(rendered_frames)
        return rendered_frames

    def renderVideoFromMemory(self, image_arr: np.ndarray, height: int, width: int, frame_ts: np.array, interpolated_ts: np.array, full_scale_count: int = 3):
        if not image_arr.shape[0] == frame_ts.shape[0]:
            raise ValueError(
                "first dim of image_arr does not match first dim of input_ts")
        self.all_images = image_arr
        self.render(self, height, width, frame_ts, interpolated_ts, full_scale_count=3)

    def renderVideoFromFolder(self, images_path: str, height: int, width: int, frame_ts: np.array, interpolated_ts, full_scale_count=3) -> np.ndarray:

        self.all_images = all_images(images_path)
        base_frame = read_image(self.all_images[0])  # todo inits emulator every time, not stateful, should be stateful to just take next sequence of images
        self.height = base_frame.shape[0]
        self.width = base_frame.shape[1]
        logging.info('(height,width)=' + str((self.height, self.width)))

        if self.emulator is None:
            self.emulator = EventEmulator(base_frame, pos_thres=self.pos_thres, neg_thres=self.neg_thres, sigma_thres=self.sigma_thres)
        super.render(self, height, width, frame_ts, interpolated_ts, full_scale_count=3)

    def _get_events(self):
        """Get all events."""

        event_list = list()

        for i in tqdm(range(self.interpolated_ts.shape[0] - 1),
                      desc="VideoSequenceFiles2EventsRenderer: ", unit='fr'):
            new_frame = read_image(self.all_images[i + 1])
            tmp_events = self.emulator.compute_events(
                new_frame,
                self.interpolated_ts[i],
                self.interpolated_ts[i + 1]
            )
            if tmp_events is not None:
                event_list.append(tmp_events)
        event_arr = np.vstack(event_list)
        logging.info("Generated {} events".format(EngNumber(event_arr.shape[0])))

        return event_arr

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
                self.emulator = EventEmulator(new_frame, pos_thres=self.pos_thres, neg_thres=self.neg_thres, sigma_thres=self.sigma_thres)
            tmp_events = self.emulator.compute_events(
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

        logging.info("Generated {} events".format(EngNumber(num_events)))
