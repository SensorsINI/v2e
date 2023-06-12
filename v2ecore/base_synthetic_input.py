# superclass for v2e synthetic input

import argparse
import atexit

import numpy as np
import cv2
import os
from tqdm import tqdm
from v2ecore.v2e_utils import *
import sys
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)


class base_synthetic_input(): # the class name should be the same as the filename, like in Java
    """ Generates moving dots on linear trajectories
    """
    BACKGROUND = 127 # defined as gray level of BACKGROUND of pix_arr

    def __init__(self, width: int = 346, height: int = 260, avi_path: Optional[str] = None, preview=True, args:Optional[List]=None, parent_args:Optional[
        argparse.Namespace]=None) -> None:
        """ prototype constructor

        :param width: width of frames in pixels
        :param height: height in pixels
        :param avi_path: folder to write video to, or None if not needed
        :param preview: set true to show the pix array as cv frame
        :param args: pass in unparsed list of extra synthetic method arguments from command line via this list
        :param parent_args: pass in parsed arguments to v2e from command line via this list
        """
        self.height=height
        self.width=width
        self.avi_path = avi_path  # to write AVI
        self.t_total = None
        self.time=0
        self.log = sys.stdout
        self.cv2name = 'v2e'
        self.codec = 'HFYU'
        self.preview = preview
        self.bg=base_synthetic_input.BACKGROUND
        self.pix_arr: np.ndarray = self.bg * np.ones((self.height, self.width), dtype=np.uint8)
        self.width = width
        self.height = height
        self.frame_number=0
        self.args=args
        self.parent_args=parent_args
        if self.preview:
            cv2.namedWindow(self.cv2name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.cv2name, self.width, self.height)
        self.video_writer=None
        if avi_path is not None:
            # construct AVI video output writer now that we know the frame size
            if not avi_path.endswith('avi'):
                avi_path=avi_path+'.avi'
            self.video_writer:video_writer = video_writer(
                output_path=avi_path,
                width=width,
                height=height, frame_rate=30
            )
            logger.info(f'Writing synthetic input frame video to {avi_path}')
        atexit.register(self.cleanup)



    def total_frames(self):
        """:returns: total number of frames"""
        return 0

    def next_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """ Returns the next frame and its time, or None when finished

        :returns: (frame, time)
            frame is a pix_arrary np.ndarray((self.height, self.w), dtype=np.uint8)
            Note y is the first dimension in accordance with OpenCV convention. Pixel 0,0 is at upper left of image.
            If there are no more frames frame should return None.
            time is in float seconds.
        """
        return (self.pix_arr, self.time)

    def write_video_frame(self,frame=None):
        """ writes the current self.pix_array to video output file as source frames
        :param frame
            the frame, or None to write self.pix_arr
        :returns: None
        """
        if not self.video_writer is None:
            self.video_writer.write(
                cv2.cvtColor(frame if frame is not None else self.pix_arr, cv2.COLOR_GRAY2BGR))

    def cleanup(self):
        if self.video_writer is not None:
            logger.info(f'closing AVI output file {self.avi_path}')
            self.video_writer.release()



if __name__ == "__main__":
    m = base_synthetic_input()
    (fr, time) = m.next_frame()
    with tqdm(total=m.total_frames(), desc='synthetic_input', unit='fr') as pbar:  # instantiate progress bar
        while fr is not None:
            (fr, time) = m.next_frame()
            pbar.update(1)
