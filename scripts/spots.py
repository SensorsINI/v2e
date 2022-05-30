# generates barberpole illusion

# use it like this:
# v2e  --output_folder gradients --unique_output --dvs_aedat2=gradients \
# --output_width=346 --output_height=260 --batch=64 --disable_slomo --synthetic_input=scripts.spots

import argparse
from typing import Tuple, Optional

# from scipy import signal
from skimage import draw  # pip install scikit-image
from tqdm import tqdm

from v2ecore.base_synthetic_input import base_synthetic_input
from v2ecore.v2e_utils import *

logger = logging.getLogger(__name__)

def draw_frame(height, width, t, bg, contrast, radius, freq):  # make spots
    gray = bg
    bright = gray * contrast
    dark = gray / contrast

    frame = np.ones((height, width), dtype=np.uint8) * gray

    if t>spots.TOTAL_TIME_S/2: # moving
        tot_time=spots.TOTAL_TIME_S/2
        total_displacement=radius
        speed=total_displacement/tot_time
        dt=t-spots.TOTAL_TIME_S/2
        color=dark
        dx=-total_displacement+ dt*speed
        draw_spot(frame, color, radius/8, width/4+dx, height/4)
        draw_square(frame, color, radius/4, 3*width/4+dx, height/4)
        draw_spot(frame, color, radius/2, width/4+dx, 3*height/4)
        draw_square(frame, color, radius/1, 3*width/4+dx, 3*height/4)
    else: # flashing
        dt=t
        sine = np.sin(dt * freq * np.pi * 2)
        color = gray if np.abs(sine) < .5 else (bright if sine > .5 else dark)
        draw_spot(frame, color, radius/8, width/4, height/4)
        draw_square(frame, color, radius/4, 3*width/4, height/4)
        draw_spot(frame, color, radius/2, width/4, 3*height/4)
        draw_square(frame, color, radius/1, 3*width/4, 3*height/4)

    return frame.astype(np.uint8)


def draw_spot(frame, color, radius, x,y):
    rr, cc = draw.disk((int(y),int(x)), radius)
    frame[rr, cc] = color

def draw_square(frame, color, radius, x,y):
    rr, cc = draw.rectangle((int(y-radius),int(x-radius)),(int(y+radius),int(x+radius)))
    frame[rr, cc] = color


class spots(base_synthetic_input):  # the class name should be the same as the filename, like in Java
    """ Generates moving dots on linear trajectories
    """
    CONTRAST = 1.5  # contrast of spot
    TOTAL_TIME_S = 1  # total time of animation
    DT_S = 100e-6  # timestemp in seconds
    RADIUS_PIX = 60  # radius of spot
    FREQ_HZ = 20  # freq of spot in Hz

    def __init__(self, width: int = 346, height: int = 260, avi_path: Optional[str] = None, preview=False,
                 arg_list=None, parent_args=None) -> None:
        """ Constructs moving-dot class to make frames for v2e

        :param width: width of frames in pixels
        :param height: height in pixels
        :param avi_path: folder to write video to, or None if not needed
        :param preview: set true to show the pix array as cv frame
        :param arg_list: remaining arguments passed from v2e command line
        """
        super().__init__(width, height, avi_path, preview, arg_list)
        parser = argparse.ArgumentParser(arg_list)
        parser.add_argument('--contrast', type=float, default=spots.CONTRAST)
        parser.add_argument('--total_time', type=float, default=spots.TOTAL_TIME_S)
        parser.add_argument('--dt', type=float, default=spots.DT_S)
        parser.add_argument('--freq', type=float, default=spots.FREQ_HZ)
        args = parser.parse_args(arg_list)

        self.avi_path = avi_path  # to write AVI
        self.contrast: float = args.contrast  # spot contrast - compare with pos_thres and neg_thres and sigma_thr,
        # e.g. use 1.2 for dot to be 20% brighter than backgreound
        self.dt = args.dt  # frame interval sec
        self.bg = 64  # 127
        # moving particle distribution
        self.t_total = args.total_time

        self.times = np.arange(0, self.t_total, self.dt)

        self.radius = spots.RADIUS_PIX # radius of largest spot

        self.freq = args.freq

        self.frame_number = 0
        self.out = None
        self.log = sys.stdout
        self.cv2name = 'v2e'
        self.codec = 'HFYU'
        self.preview = preview
        self.y = np.array(range(self.height))
        self.x = np.array(range(self.width))
        self.last_frame_written_time = 0

        logger.info(f'contrast(factor): {self.contrast}\n'
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
        self.pix_arr = draw_frame(self.height, self.width, time, self.bg, self.contrast, self.radius, self.freq)

        if self.preview and self.frame_number % 1 == 0:
            cv2.imshow(self.cv2name, self.pix_arr)
        if self.video_writer is not None and time==0 or time > self.last_frame_written_time + 1. / 1000.:
            self.write_video_frame(self.pix_arr)
            self.last_frame_written_time = time
        if self.preview and self.frame_number%10==0:
            k = cv2.waitKey(1)
            if k == ord('x'):
                logger.warning('aborted output after {} frames'.format(self.frame_number))
                cv2.destroyAllWindows()
                self.cleanup()
                return None, time
        self.frame_number += 1
        return (self.pix_arr, time)



# debug by running main
if __name__ == "__main__":
    m = spots(preview=True)
    (fr, time) = m.next_frame()
    with tqdm(total=m.total_frames(), desc='simulation', unit='fr') as pbar:  # instantiate progress bar
        while fr is not None:
            (fr, time) = m.next_frame()
            pbar.update(1)
