# generates many linearly moving particles

# use it like this:
# v2e --leak_rate=0 --shot=0 --cutoff_hz=300 --sigma_thr=.05 --pos_thr=.2 --neg_thr=.2 --dvs_exposure duration .01 --output_folder "g:\Qsync\particles" --overwrite --dvs_aedat2=particles --output_width=346 --output_height=260 --batch=64 --disable_slomo --synthetic_input=scripts.particles --no_preview

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
def fill_dot(pix_arr: np.ndarray, x: float, y: float, fg: int, bg: int, dot_sigma: float):
    """ Generates intensity values for the 'dot'

    :param pix_arr: the 2d pixel array to fill values to
    :param x: center of dot x in pixels
    :param y: center of dot y in pixels
    :param d: square radius range to generate dot over
    :param fg: the foreground intensity (peak value) of center of dot
    :param bg: the background value outside of dot that we approach at edge of dot
    :param dot_sigma: the sigma of Gaussian, i.e. radius of dot
    """
    x0, y0 = round(x), round(y)
    d=int(dot_sigma * 3)

    for iy in range(-d, +d):
        for ix in range(-d, +d):
            thisx, thisy = int(x0 + ix), int(y0 + iy)
            # bounds check, remember that cv2 uses y for first axis
            if thisx<0 or thisx>=pix_arr.shape[1] or thisy<0 or thisy>=pix_arr.shape[0]:
                continue
            ddx, ddy = thisx - x, thisy - y  # distances of this pixel to float dot location
            dist2 = ddx * ddx + ddy * ddy  # square distance
            v = 10 * np.exp(-dist2 / (dot_sigma * dot_sigma))  # gaussian normalized intensity value
            if v > 1: # make a disk, not a gaussian blob
                v = 1
            elif v < .01:
                v = 0
            v = bg + (fg - bg) * v  # intensity value from 0-1 intensity
            pix_arr[thisy][thisx] = v

class particle():
    def __init__(self, width:int, height:int , time:float):
        self.width=width
        self.height=height
        # generate particle on some edge, moving into the array with random velocity
        edge=np.random.randint(0,4) # nsew
        if edge==0 or edge==1: #north/south
            pos_x=np.random.randint(0,width)
            pos_y=0 if edge==0 else height
        else: # e or w
            pos_y=np.random.randint(0,height)
            pos_x=0 if edge==3 else width
        angle_rad=0
        if edge==1: #n
            angle_rad=np.random.uniform(0,-np.pi)
        elif edge==0: # s
            angle_rad=np.random.uniform(0,np.pi)
        elif edge==3: # e
            angle_rad=np.random.uniform(-np.pi/2,np.pi/2)
        elif edge==2: # w
            angle_rad=np.random.uniform(np.pi/2,3*np.pi/2)


        self.position=np.array([pos_x,pos_y])
        self.speed=np.random.uniform(100,2000)
        self.velocity=np.array([self.speed*np.cos(angle_rad),self.speed*np.sin(angle_rad)])
        self.radius=np.random.uniform(.25,.5)
        self.contrast=np.random.uniform(1.2,1.5)
        self.time=time

    def update(self,time:float):
        dt=time-self.time
        self.position=self.position+dt*self.velocity
        self.time=time

    def is_out_of_bounds(self):
        return self.position[0]<0 or self.position[0]>self.width or self.position[1]<0 or self.position[1]>self.height

    def draw(self, pix_arr):
        bg=100
        fg= int(bg * self.contrast)  # foreground dot brightness
        fill_dot(pix_arr,self.position[0], self.position[1], fg, bg, self.radius)

class particles(): # the class name should be the same as the filename, like in Java
    """ Generates moving dots on linear trajectories
    """

    def __init__(self, width: int = 346, height: int = 260, avi_path: Optional[str] = None, preview=True) -> None:
        """ Constructs moving-dot class to make frames for v2e

        :param width: width of frames in pixels
        :param height: height in pixels
        :param avi_path: folder to write video to, or None if not needed
        :param preview: set true to show the pix array as cv frame
        """
        self.avi_path = avi_path  # to write AVI
        self.num_dots = 100  # number of dots
        self.contrast: float = 1.50  # compare this with pos_thres and neg_thres and sigma_thr, e.g. use 1.2 for dot to be 20% brighter than backgreound
        self.bg: int = 100  # background gray level in range 0-255
        self.dt = 30e-6  # frame interval sec
        self.radius = 100  # of circular motion of dot
        self.dot_sigma: float = 1  # gaussian sigma of dot in pixels
        # moving particle distribution
        self.speed_pps_min = 100  # final speed, pix/s
        self.speed_pps_max = 2000  # final speed, pix/s
        self.num_particles=40 # at any one time
        self.particle_count=0

        self.particles=[]
        for i in range(self.num_particles):
            p=particle(width=width,height=height,time=0)
            self.particles.append(p)
            self.particle_count+=1

        # computed values below here
        # self.t_total = 4 * np.pi * self.radius * self.cycles / self.speed_pps
        self.t_total = 30
        # t_total=cycles*period
        self.times = np.arange(0, self.t_total, self.dt)
        # constant speed
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
        self.pix_arr: np.ndarray = self.bg * np.ones((self.h, self.w), dtype=np.uint8)
        logger.info(f'speed(pixels/s): {self.speed_pps_min} to {self.speed_pps_max}\n'
                    f'dot_sigma(pixels): {self.dot_sigma}\n'
                    f'radius(pixels): {self.radius}\n'
                    f'contrast(factor): {self.contrast}\n'
                    f'log_contrast(base_e): {np.log(self.contrast)}\n'
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
            logger.info(f'finished after {self.frame_number} frames having made {self.particle_count} particles')
            return None, self.times[-1]
        time = self.times[self.frame_number]
        self.pix_arr.fill(self.bg)
        for p in self.particles:
            if p.is_out_of_bounds():
                self.particles.remove(p)
                newp=particle(self.w,self.h,time)
                self.particles.append(newp)
                self.particle_count+=1
                # logger.info(f'made new particle {newp}')
            else:
                p.update(time)
                p.draw(self.pix_arr)

        if self.preview:
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


if __name__ == "__main__":
    m = particles()
    (fr, time) = m.next_frame()
    with tqdm(total=m.total_frames(), desc='moving-dot', unit='fr') as pbar:  # instantiate progress bar
        while fr is not None:
            (fr, time) = m.next_frame()
            pbar.update(1)
