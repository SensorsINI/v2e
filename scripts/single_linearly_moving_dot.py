# generates moving dot(s)

# use it like this:
#v2e --synthetic_input=scripts.moving_dot --disable_slomo --dvs_aedat2=v2e.aedat --output_width=346 --output_height=260

# NOTE: There are nonintuitive effects of low contrast dot moving repeatedly over the same circle:
# The dot initially makes events and then appears to disappear. The cause is that the mean level of dot
# is encoded by the baseLogFrame which is initially at zero but increases to code the average of dot and background.
# Then the low contrast of dot causes only a single ON event on first cycle
import numpy as np
import cv2
import os
from tqdm import tqdm
from v2ecore.v2e_utils import *
import sys
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


@njit
def fill_dot(pix_arr: np.ndarray, x: float, x0: float, y: float, y0: float, d: int, fg: int, bg: int, dot_sigma: float):
    """ Generates intensity values for the 'dot'

    :param pix_arr: the 2d pixel array to fill values to
    :param x: center of dot x in pixels
    :param y: center of dot y in pixels
    :param x0: rounded x location, used to compute delta x
    :param y0: rounded y location, used to compute delta y
    :param d: square radius range to generate dot over
    :param fg: the foreground intensity (peak value) of center of dot
    :param bg: the background value outside of dot that we approach at edge of dot
    :param dot_sigma: the sigma of Gaussian, i.e. radius of dot
    """
    for iy in range(-d, +d):
        for ix in range(-d, +d):
            thisx, thisy = int(x0 + ix), int(y0 + iy)
            ddx, ddy = thisx - x, thisy - y  # distances of this pixel to float dot location
            dist2 = ddx * ddx + ddy * ddy  # square distance
            v = 10 * np.exp(-dist2 / (dot_sigma * dot_sigma))  # gaussian normalized intensity value
            if v > 1: # make a disk, not a gaussian blob
                v = 1
            elif v < .01:
                v = 0
            v = bg + (fg - bg) * v  # intensity value from 0-1 intensity
            pix_arr[thisy][thisx] = v


class single_linearly_moving_dot(): # the class name should be the same as the filename, like in Java
    """ Generates moving dot
    """

    def __init__(self, width: int = 346, height: int = 260, avi_path: Optional[str] = None, preview=True, arg_list = None) -> None:
        """ Constructs moving-dot class to make frames for v2e

        :param width: width of frames in pixels
        :param height: height in pixels
        :param avi_path: folder to write video to, or None if not needed
        :param preview: set true to show the pix array as cv frame
        """
        self.avi_path = avi_path  # to write AVI
        self.contrast: float = 3  # compare this with pos_thres and neg_thres and sigma_thr, e.g. use 1.2 for dot to be 20% brighter than backgreound
        self.bg: int = 100  # background gray level in range 0-255
        self.dt = 100e-6  # frame interval sec
        self.dot_sigma: float = 3  # gaussian sigma of dot in pixels
        self.speed_pps = 100  # final speed, pix/s
        self.t_total = width / self.speed_pps
        self.times = np.arange(0, self.t_total, self.dt)
        self.fg: int = int(self.bg * self.contrast)  # foreground dot brightness
        self.w = width
        self.h = height
        self.d: int = int(self.dot_sigma * 3)  # distance to bother creating gray levels
        self.fps = 60
        self.frame_number = 0
        self.out = None
        self.log = sys.stdout
        self.cv2name = 'moving-dot'
        self.codec = 'HFYU'
        self.preview = preview
        print('moving-dot: hit x to exit early')
        logger.info(f'final_speed(pixels/s): {self.speed_pps}\n'
                    f'dot_sigma(pixels): {self.dot_sigma}\n'
                    f'contrast(factor): {self.contrast}\n'
                    f'log_contrast(base_e): {np.log(self.contrast)}\n'
                    f'bg: {self.bg}\n'
                    f'fg: {self.fg}\n'
                    f'duration(s): {self.t_total}\n'
                    f'dt(s): {self.dt}\n'
                    f'fps(Hz): {self.fps}\n'
                    f'codec: {self.codec}\n')
        if self.preview:
            cv2.namedWindow(self.cv2name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.cv2name, self.w, self.h)

    def total_frames(self):
        """:returns: total number of frames"""
        return len(self.times)

    def next_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """ Returns the next frame and its time, or None when finished

        :returns: (frame, time)
            If there are no more frames frame is None.
            time is in seconds.
        """
        if self.frame_number >= len(self.times):
            if self.avi_path is not None:
                self.out.release()
            cv2.destroyAllWindows()
            logger.info('finished after {} frames'.format(self.frame_number))
            return None, self.times[-1]
        time = self.times[self.frame_number]
        pix_arr: np.ndarray = self.bg * np.ones((self.h, self.w), dtype=np.uint8)
        # actual center of dot
        x = (time*self.speed_pps)
        y = self.h / 2
        # nearest pixel
        x0, y0 = x,y
        fill_dot(pix_arr, x, x0, y, y0, self.d, self.fg, self.bg, self.dot_sigma)
        if self.preview:
            cv2.imshow(self.cv2name, pix_arr)
        if self.avi_path is not None:
            self.out.write(cv2.cvtColor(pix_arr, cv2.COLOR_GRAY2BGR))
        if self.preview and self.frame_number % 50 == 0:
            k = cv2.waitKey(1)
            if k == ord('x'):
                logger.warning('aborted output after {} frames'.format(self.frame_number))
                cv2.destroyAllWindows()
                return None, time
        self.frame_number += 1
        return (pix_arr, time)


if __name__ == "__main__":
    m = moving_dot()
    (fr, time) = m.next_frame()
    with tqdm(total=m.total_frames(), desc='moving-dot', unit='fr') as pbar:  # instantiate progress bar
        while fr is not None:
            (fr, time) = m.next_frame()
            pbar.update(1)
