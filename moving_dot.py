# generates a moving dot for DVSDENOISE21 paper
import numpy as np
import cv2
import numpy as np
import os
from tqdm import tqdm
from v2e.v2e_utils import *
import sys
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class moving_dot():
    """ Generates moving dot
    """
    def __init__(self, width:int=346, height:int=260, avi_path:Optional[str]=None) -> None:
        """ Constructs moving-dot class to make frames for v2e

        :param width: width of frames in pixels
        :param height: height in pixels
        :param avi_path: folder to write video to, or None if not needed
        """
        self.avi_path = avi_path  # to write AVI
        self.dt = .1e-3  # frame interval sec
        self.w = width
        self.h = height
        self.radius = 100  # of circular motion of dot
        self.dot_sigma = 1  # gaussian sigma of dot
        self.circum = 2 * np.pi * self.radius
        self.speed_pps = 3000  # final speed, pix/s
        self.period = self.circum / self.speed_pps
        self.cycles = 10
        self.t_total = 4 * np.pi * self.radius * self.cycles / self.speed_pps
        # t_total=cycles*period
        self.times = np.arange(0, self.t_total, self.dt)
        self.theta = (self.speed_pps * self.speed_pps / (8 * np.pi * self.radius * self.radius * self.cycles)) * self.times * self.times
        self.bg = 127  # background gray level
        self.contrast = 1.25
        self.fg = int(self.bg * self.contrast)  # foreground dot brightness
        self.d = int(self.dot_sigma * 3)
        self.fps = 60
        self.frame_number = 0
        self.out = None
        self.log = sys.stdout
        self.cv2name = 'moving-dot'
        self.codec='HFYU'
        print('moving-dot: hit x to exit early')
        logger.info(f'final_speed(pixels/s): {self.speed_pps}\n'
              f'dot_sigma(pixels): {self.dot_sigma}\n'
              f'radius(pixels): {self.radius}\n'
              f'contrast(factor): {self.contrast}\n'
              f'log_contrast(base_e): {np.log(self.contrast)}\n'
              f'bg: {self.bg}\n'
              f'fg: {self.fg}\n'
              f'duration(s): {self.t_total}\n'
              f'cycles: {self.cycles}\n'
              f'dt(s): {self.dt}\n'
              f'fps(Hz): {self.fps}\n'
              f'codec: {self.codec}\n')
        cv2.namedWindow(self.cv2name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.cv2name, self.w, self.h)

    def total_frames(self):
        """:returns: total number of frames"""
        return len(self.times)

    def next_frame(self)->Tuple[Optional[np.ndarray],float]:
        """ Returns the next frame and its time, or None when finished

        :returns: (frame, time)
            If there are no more frames frame is None.
            time is in seconds.
        """
        if self.frame_number>=len(self.times):
            if self.avi_path is not None:
                self.out.release()
            cv2.destroyAllWindows()
            logger.info(f'finished after {self.frame_number} frames')
            return None,self.times[-1]
        time=self.times[self.frame_number]
        pix_arr = self.bg * np.ones((self.h, self.w), dtype=np.uint8)
        # actual center of dot
        x = self.w / 2 + self.radius * np.cos(self.theta[self.frame_number])
        y = self.h / 2 + self.radius * np.sin(self.theta[self.frame_number])
        # nearest pixel
        x0, y0 = round(x), round(y)
        # range of indexes around x0,y0
        if time > 0:  # make sure there is one blank frame to start with, to set baseLogFrame
            for iy in range(-self.d, +self.d):
                for ix in range(-self.d, +self.d):
                    thisx, thisy = int(x0 + ix), int(y0 + iy)
                    # distances of this pixel to float dot location
                    ddx, ddy = thisx - x, thisy - y
                    dist2 = ddx * ddx + ddy * ddy
                    v = np.exp(-dist2 / (self.dot_sigma * self.dot_sigma))
                    v = self.bg + (self.fg - self.bg) * v
                    pix_arr[thisy][thisx] = v
        cv2.imshow(self.cv2name, pix_arr)
        if self.avi_path is not None:
            self.out.write(cv2.cvtColor(pix_arr, cv2.COLOR_GRAY2BGR))
        k = cv2.waitKey(1)
        if k == ord('x'):
            logger.warning(f'aborted output after {self.frame_number} frames')
            cv2.destroyAllWindows()
            return None,time
        self.frame_number+=1
        return (pix_arr,time)


if __name__ == "__main__":
    m=moving_dot()
    (fr,time)=m.next_frame()
    with tqdm(total=m.total_frames(), desc='moving-dot', unit='fr') as pbar: # instantiate progress bar
        while fr is not None:
            (fr,time)=m.next_frame()
            pbar.update(1)
