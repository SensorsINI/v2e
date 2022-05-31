# generates barberpole illusion

# use it like this:
# v2e  --output_folder gradients --unique_output --dvs_aedat2=gradients \
# --output_width=346 --output_height=260 --batch=64 --disable_slomo --synthetic_input=scripts.gradients

import argparse
import atexit
import sys
from typing import Tuple, Optional

from tqdm import tqdm

from v2ecore.base_synthetic_input import base_synthetic_input
from v2ecore.v2e_utils import *

logger = logging.getLogger(__name__)


class gradients(base_synthetic_input):  # the class name should be the same as the filename, like in Java
    """ Generates moving dots on linear trajectories
    """
    CONTRAST = 2  # contrast of barberpole peak to peak
    TOTAL_TIME = 1  # total time of animation
    DT = 100e-6  # timestemp in seconds
    SPEED_PPS = 300  # apparent speed of barberpole along axis
    BUMP_WIDTH=.5 # width of the triangular bump as fraction of width of array

    def __init__(self, width: int = 346, height: int = 260, avi_path: Optional[str] = None, preview=False,
                 arg_list=None, parent_args=None) -> None:
        """ Constructs moving-dot class to make frames for v2e

        :param width: width of frames in pixels
        :param height: height in pixels
        :param avi_path: folder to write video to, or None if not needed
        :param preview: set true to show the pix array as cv frame
        """
        super().__init__(width, height, avi_path,  preview, arg_list)
        parser = argparse.ArgumentParser(arg_list)
        parser.add_argument('--contrast', type=float, default=gradients.CONTRAST)
        parser.add_argument('--total_time', type=float, default=gradients.TOTAL_TIME)
        parser.add_argument('--speed_pps', type=float, default=gradients.SPEED_PPS)
        parser.add_argument('--dt', type=float, default=gradients.DT)
        parser.add_argument('--bump_width', type=float, default=gradients.BUMP_WIDTH)
        args = parser.parse_args(arg_list)

        self.avi_path = avi_path  # to write AVI
        self.contrast: float = args.contrast  # bright to dark stripe ratio; compare this with pos_thres and neg_thres and sigma_thr, e.g. use 1.2 for dot to be 20% brighter than backgreound
        self.dt = args.dt  # frame interval sec
        self.bg = base_synthetic_input.BACKGROUND  # 127
        # moving particle distribution
        self.t_total = args.total_time
        self.speed_pps = args.speed_pps
        self.bump_width=args.bump_width

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
        self.y = np.array(range(self.h))
        self.x = np.array(range(self.w))
        self.last_frame_written_time=0

        logger.info(f'speed(pixels/s): {self.speed_pps}\n'
                    f'contrast(factor): {self.contrast}\n'
                    f'log_contrast(base_e): {np.log(self.contrast)}\n'
                    f'duration(s): {self.t_total}\n'
                    f'dt(s): {self.dt}\n'
                    f'codec: {self.codec}\n')


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
            cv2.destroyAllWindows()
            logger.info(f'finished after {self.frame_number} frames')
            return None, self.times[-1]
        time = self.times[self.frame_number]
        # self.pix_arr.fill(self.bg)
        self.pix_arr = self.im_function(self.y[:, None], self.x[None, :], time)

        if self.preview and self.frame_number % 1 == 0:
            cv2.imshow(self.cv2name, self.pix_arr)
        if self.video_writer is not None and time>self.last_frame_written_time+1./30.:
            self.video_writer.write(cv2.cvtColor(self.pix_arr, cv2.COLOR_GRAY2BGR))
            self.last_frame_written_time=time
        if self.preview and self.frame_number % 50 == 0:
            k = cv2.waitKey(1)
            if k == ord('x'):
                logger.warning('aborted output after {} frames'.format(self.frame_number))
                cv2.destroyAllWindows()
                self.cleanup()
                return None, time
        self.frame_number += 1
        return (self.pix_arr, time)

    def im_function(self, y, x, t):
        # compute high and low pixel values for contrast between high and low and average equal background
        # makes a triangular bump that moves to right. There is a sharp rectangular edge in front of the bump to test high spatial frequencies.

        low = (self.bg * 2) / (self.contrast + 1)
        high = self.contrast * low
        diff = high - low
        w2 = (self.bump_width*self.w) / 2
        p = w2 + t * self.speed_pps # center of bump location
        p2=p+w2*2 # center of sharp edges
        g = np.ones((self.h, self.w)) * low
        x = np.squeeze(x)
        # left side of bump
        ind=(x > p - w2) & (x < p)
        g[:, ind] = high + (-diff / w2) * (p-x[ind])
        # right side of bump
        ind= (x <= p + w2) & (x >= p)
        g[:,ind] = high + (-diff / w2) * (x[ind]-p)

        #square wave
        ind= (x >p2) & (x <= p2+10)
        g[:,ind] = high

        return np.uint8(g)


if __name__ == "__main__":
    m = gradients()
    (fr, time) = m.next_frame()
    with tqdm(total=m.total_frames(), desc='moving-dot', unit='fr') as pbar:  # instantiate progress bar
        while fr is not None:
            (fr, time) = m.next_frame()
            pbar.update(1)
