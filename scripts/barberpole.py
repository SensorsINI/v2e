# generates barberpole illusion

# use it like this:
# v2e --leak_rate=0 --shot=0 --cutoff_hz=300 --sigma_thr=.08 --pos_thr=.15 --neg_thr=.15 \
# --dvs_exposure duration .01 --output_folder particles-slightly-less-faint-fast-2-particles --unique_output --dvs_aedat2=particles \
# --output_width=346 --output_height=260 --batch=64 --disable_slomo --synthetic_input=scripts.particles\
# --total_time=3 --contrast=1.15 --radius=.3 --speed_min=1000 --speed_max=3000 --dt=100e-6 --num_particles=2

import argparse
import atexit

import numpy as np
import cv2
import os
from tqdm import tqdm

from v2ecore.base_synthetic_input import base_synthetic_input
from v2ecore.v2e_utils import *
import sys
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class barberpole(base_synthetic_input): # the class name should be the same as the filename, like in Java
    """ Generates moving dots on linear trajectories
    """
    CONTRAST = 1.5  # contrast of barberpole peak to peak
    TOTAL_TIME = 1 # total time of animation
    DT=100e-6 # timestemp in seconds
    SPEED_PPS=1000 # apparent speed of barberpole along axis
    NUM_STRIPES=6 # number of barberpole stripes
    BB_WIDTH=0.8 # width and height of barberpole as fraction of w and h
    BB_HEIGHT=.2
    BB_ANGLE=30 # default angle of barberpole in degrees


    def __init__(self, width: int = 346, height: int = 260, avi_path: Optional[str] = None, preview=False,
                 arg_list = None) -> None:
        """ Constructs moving-dot class to make frames for v2e

        :param width: width of frames in pixels
        :param height: height in pixels
        :param avi_path: folder to write video to, or None if not needed
        :param preview: set true to show the pix array as cv frame
        """
        super().__init__(width, height, avi_path, preview, arg_list)
        parser=argparse.ArgumentParser(arg_list)
        parser.add_argument('--num_stripes',type=int,default=barberpole.NUM_STRIPES)
        parser.add_argument('--contrast',type=float,default=barberpole.CONTRAST)
        parser.add_argument('--total_time',type=float,default=barberpole.TOTAL_TIME)
        parser.add_argument('--speed_pps',type=float,default=barberpole.SPEED_PPS)
        parser.add_argument('--dt',type=float,default=barberpole.DT)
        parser.add_argument('--bb_width',type=float,default=barberpole.BB_WIDTH)
        parser.add_argument('--bb_height',type=float,default=barberpole.BB_HEIGHT)
        parser.add_argument('--bb_angle',type=float,default=barberpole.BB_ANGLE)
        args=parser.parse_args(arg_list)


        self.avi_path = avi_path  # to write AVI
        self.contrast: float = args.contrast  # bright to dark stripe ratio; compare this with pos_thres and neg_thres and sigma_thr, e.g. use 1.2 for dot to be 20% brighter than backgreound
        self.dt = args.dt  # frame interval sec
        self.bg=base_synthetic_input.BACKGROUND #127
        # moving particle distribution
        self.t_total = args.total_time
        self.speed_pps=args.speed_pps
        self.num_stripes=args.num_stripes
        self.bb_width=args.bb_width
        self.bb_height=args.bb_height
        self.bb_angle=args.bb_angle

        # computed values below here
        # self.t_total = 4 * np.pi * self.radius * self.cycles / self.speed_pps
        # t_total=cycles*period
        self.times = np.arange(0, self.t_total, self.dt)
        # constant speed
        self.w = width
        self.h = height
        self.frame_number = 0
        self.out = None
        self.log = sys.stdout
        self.cv2name = 'v2e'
        self.codec = 'HFYU'
        self.preview = preview
        self.y=np.array(range(self.h))
        self.x=np.array(range(self.w))
        dx=(1-self.bb_width)/2
        maxx=self.w-round(self.w*dx)
        minx=round(self.w*dx)
        dy=(1-self.bb_height)/2
        maxy=self.h-round(self.h*dy)
        miny=round(self.h*dy)
        self.pole_mask=np.zeros((self.h,self.w),dtype=np.uint8)
        for y in range(self.h):
            for x in range(self.w):
                if x>minx and x<maxx and y>miny and y<maxy:
                    self.pole_mask[y,x]=1
        self.pole_mask_y,self.pole_mask_x=np.where(self.pole_mask==0) # rows/cols of background

        logger.info(f'speed(pixels/s): {self.speed_pps}\n'
                    f'contrast(factor): {self.contrast}\n'
                    f'log_contrast(base_e): {np.log(self.contrast)}\n'
                    f'duration(s): {self.t_total}\n'
                    f'dt(s): {self.dt}\n'
                    f'codec: {self.codec}\n')
        if self.preview:
            cv2.namedWindow(self.cv2name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.cv2name, self.w, self.h)

        atexit.register(self.cleanup)



    def cleanup(self):
        pass

    def total_frames(self):
        """:returns: total number of frames"""
        return len(self.times)


    def next_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """ Returns the next frame and its time, or None when finished

       :returns: (frame, time)
           frame is a pix_arrary np.ndarray((self.height, self.w), dtype=np.uint8)
           Note y is the first dimension in accordance with OpenCV convention. Pixel 0,0 is at upper left of image.
           If there are no more frames frame should return None.
           time is in float seconds.
       """
        if self.frame_number >= len(self.times):
            if self.avi_path is not None:
                self.out.release()
            cv2.destroyAllWindows()
            logger.info(f'finished after {self.frame_number} frames')
            return None, self.times[-1]
        time = self.times[self.frame_number]
        # self.pix_arr.fill(self.bg)
        self.pix_arr=self.bb_func(self.y[:,None],self.x[None,:],  time)
        self.pix_arr[self.pole_mask_y,self.pole_mask_x]=self.bg

        if self.preview and self.frame_number % 1 == 0:
            cv2.imshow(self.cv2name, self.pix_arr)
        if self.avi_path is not None:
            self.out.write(cv2.cvtColor(self.pix_arr, cv2.COLOR_GRAY2BGR))
        if self.preview and self.frame_number % 50 == 0:
            k = cv2.waitKey(1)
            if k == ord('x'):
                logger.warning('aborted output after {} frames'.format(self.frame_number))
                cv2.destroyAllWindows()
                return None, time
        self.frame_number += 1
        return (self.pix_arr, time)

    def bb_func(self, y,x, t):
        wavelength=(self.bb_width*self.w)/self.num_stripes
        # compute high and low pixel values for contrast between high and low and average equal background
        low=(self.bg*2)/(self.contrast+1)
        high=self.contrast*low
        diff=(self.contrast-1)*low
        tan=np.tan((90-self.bb_angle)*np.pi/180)
        stripes=np.floor(low+diff*0.5*(1+np.tanh(10*np.sin(2*np.pi*(y-tan*x-t*self.speed_pps)/wavelength))))
        return np.uint8(stripes)


if __name__ == "__main__":
    m = barberpole()
    (fr, time) = m.next_frame()
    with tqdm(total=m.total_frames(), desc='moving-dot', unit='fr') as pbar:  # instantiate progress bar
        while fr is not None:
            (fr, time) = m.next_frame()
            pbar.update(1)
