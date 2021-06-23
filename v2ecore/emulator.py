"""
DVS simulator.
Compute events from input frames.

@author: Zhe He
@contact: zhehe@student.ethz.ch
@credits: Yuhuang Hu
@latest updaste: 2019-Jun-13
"""
import atexit
import os
import time
from functools import partial

import cv2
import numpy as np
import logging
import h5py
from engineering_notation import EngNumber  # only from pip

# JAX
import jax.numpy as jnp
from jax import random
from jax import jit
from jax.config import config

from v2ecore.v2e_utils import all_images, read_image, \
    video_writer, checkAddSuffix
from v2ecore.output.aedat2_output import AEDat2Output
from v2ecore.output.ae_text_output import DVSTextOutput

from v2ecore.emulator_utils import lin_log
from v2ecore.emulator_utils import rescale_intensity_frame
from v2ecore.emulator_utils import low_pass_filter
from v2ecore.emulator_utils import subtract_leak_current
from v2ecore.emulator_utils import compute_event_map
from v2ecore.emulator_utils import generate_shot_noise

# configure jax
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

# import rosbag # not yet for python 3

logger = logging.getLogger(__name__)


class EventEmulator(object):
    """compute events based on the input frame.
    - author: Zhe He
    - contact: zhehe@student.ethz.ch
    """

    # todo add refractory period

    def __init__(
            self,
            pos_thres=0.2,
            neg_thres=0.2,
            sigma_thres=0.03,
            cutoff_hz=0,
            leak_rate_hz=0.1,
            refractory_period_s=0,  # todo not yet modeled
            shot_noise_rate_hz=0,  # rate in hz of temporal noise events
            seed=0,
            output_folder: str = None,
            dvs_h5: str = None,
            dvs_aedat2: str = None,
            dvs_text: str = None,
            # change as you like to see 'baseLogFrame',
            # 'lpLogFrame', 'diff_frame'
            show_dvs_model_state: str = None,
            output_width=None,
            output_height=None):
        """
        Parameters
        ----------
        base_frame: np.ndarray
            [height, width]. If None, then it is initialized from first data
        pos_thres: float, default 0.21
            nominal threshold of triggering positive event in log intensity.
        neg_thres: float, default 0.17
            nominal threshold of triggering negative event in log intensity.
        sigma_thres: float, default 0.03
            std deviation of threshold in log intensity.
        cutoff_hz: float,
            3dB cutoff frequency in Hz of DVS photoreceptor
        leak_rate_hz: float
            leak event rate per pixel in Hz,
            from junction leakage in reset switch
        shot_noise_rate_hz: float
            shot noise rate in Hz
        seed: int, default=0
            seed for random threshold variations,
            fix it to nonzero value to get same mismatch every time
        dvs_aedat2, dvs_h5, dvs_text: str
            names of output data files or None
        show_dvs_model_state: str,
            None or 'new_frame' 'baseLogFrame','lpLogFrame0','lpLogFrame1',
            'diff_frame'
        output_width: int,
            width of output in pixels
        output_height: int,
            height of output in pixels
        """

        logger.info(
            "ON/OFF log_e temporal contrast thresholds: "
            "{} / {} +/- {}".format(pos_thres, neg_thres, sigma_thres))

        self.base_log_frame = None
        self.t_previous = None  # time of previous frame

        # thresholds
        self.sigma_thres = sigma_thres
        # initialized to scalar, later overwritten by random value array
        self.pos_thres = pos_thres
        # initialized to scalar, later overwritten by random value array
        self.neg_thres = neg_thres
        self.pos_thres_nominal = pos_thres
        self.neg_thres_nominal = neg_thres

        # non-idealities
        self.cutoff_hz = cutoff_hz
        self.leak_rate_hz = leak_rate_hz
        self.refractory_period_s = refractory_period_s
        self.shot_noise_rate_hz = shot_noise_rate_hz

        # output properties
        self.output_width = output_width
        self.output_height = output_height  # set on first frame
        self.show_input = show_dvs_model_state

        # generate jax key for random process
        if seed == 0:
            # use fractional seconds
            seed = int(time.time()*256)

        self.jax_key = random.PRNGKey(seed)

        if refractory_period_s > 0:
            logger.warning(
                'refractory period not yet implemented; '
                'refractory_period_s={} will be ignored'.format(
                    refractory_period_s))

        self.output_folder = output_folder
        self.dvs_h5 = dvs_h5
        self.dvs_aedat2 = dvs_aedat2
        self.dvs_text = dvs_text
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0
        self.frame_counter = 0

        if self.output_folder:
            if dvs_h5:
                path = os.path.join(self.output_folder, dvs_h5)
                path = checkAddSuffix(path, '.h5')
                logger.info('opening event output dataset file ' + path)
                self.dvs_h5 = h5py.File(path, "w")
                self.dvs_h5_dataset = self.dvs_h5.create_dataset(
                    name="events",
                    shape=(0, 4),
                    maxshape=(None, 4),
                    dtype="uint32",
                    compression="gzip")
            if dvs_aedat2:
                path = os.path.join(self.output_folder, dvs_aedat2)
                path = checkAddSuffix(path, '.aedat')
                logger.info('opening AEDAT-2.0 output file ' + path)
                self.dvs_aedat2 = AEDat2Output(
                    path, output_width=self.output_width,
                    output_height=self.output_height)
            if dvs_text:
                path = os.path.join(self.output_folder, dvs_text)
                path = checkAddSuffix(path, '.txt')
                logger.info('opening text DVS output file ' + path)
                self.dvs_text = DVSTextOutput(path)
        atexit.register(self.cleanup)

    def cleanup(self):
        if self.dvs_h5 is not None:
            self.dvs_h5.close()

        if self.dvs_aedat2 is not None:
            self.dvs_aedat2.close()

        if self.dvs_text is not None:
            self.dvs_text.close()

    def _init(self, first_frame_linear):
        logger.debug(
            'initializing random temporal contrast thresholds '
            'from from base frame')
        # base_frame are memorized lin_log pixel values
        self.base_log_frame = lin_log(first_frame_linear)

        # initialize first stage of 2nd order IIR to first input
        self.lp_log_frame0 = jnp.array(self.base_log_frame, copy=True)
        # 2nd stage is initialized to same,
        # so diff will be zero for first frame
        self.lp_log_frame1 = jnp.array(self.base_log_frame, copy=True)

        # take the variance of threshold into account.
        if self.sigma_thres > 0:
            self.pos_thres = random.normal(
                self.jax_key,
                first_frame_linear.shape)*self.sigma_thres+self.pos_thres
            # to avoid the situation where the threshold is too small.
            self.pos_thres = jnp.where(
                self.pos_thres < 0.01, 0.01, self.pos_thres)

            self.neg_thres = random.normal(
                self.jax_key,
                first_frame_linear.shape)*self.sigma_thres+self.neg_thres
            self.neg_thres = jnp.where(
                self.neg_thres < 0.01, 0.01, self.neg_thres)

        # compute variable for shot-noise
        self.pos_thres_pre_prob = jnp.divide(
            self.pos_thres_nominal, self.pos_thres)
        self.neg_thres_pre_prob = jnp.divide(
            self.neg_thres_nominal, self.neg_thres)

        # If leak is non-zero, then initialize each pixel memorized value
        # some fraction of ON threshold below first frame value, to create leak
        # events from the start; otherwise leak would only gradually
        # grow over time as pixels spike.
        # do this *AFTER* we determine randomly distributed thresholds
        # (and use the actual pixel thresholds)
        # otherwise low threshold pixels will generate
        # a burst of events at the first frame
        if self.leak_rate_hz > 0:
            self.base_log_frame -= random.uniform(
                self.jax_key, first_frame_linear.shape,
                dtype=jnp.float32,
                minval=0, maxval=self.pos_thres)

    def set_dvs_params(self, model: str):
        if model == 'clean':
            self.pos_thres = 0.2
            self.neg_thres = 0.2
            self.sigma_thres = 0.02
            self.cutoff_hz = 0
            self.leak_rate_hz = 0
            self.shot_noise_rate_hz = 0  # rate in hz of temporal noise events
            self.refractory_period_s = 0  # TODO not yet modeled

        elif model == 'noisy':
            self.pos_thres = 0.2
            self.neg_thres = 0.2
            self.sigma_thres = 0.05
            self.cutoff_hz = 30
            self.leak_rate_hz = 0.1
            # rate in hz of temporal noise events
            self.shot_noise_rate_hz = 5.0
            self.refractory_period_s = 0  # TODO not yet modeled
        else:
            #  logger.error(
            #      "dvs_params {} not known: "
            #      "use 'clean' or 'noisy'".format(model))
            logger.warning(
                "dvs_params {} not known: "
                "Using commandline assigned options".format(model))
            #  sys.exit(1)
        logger.info("set DVS model params with option '{}' "
                    "to following values:\n"
                    "pos_thres={}\n"
                    "neg_thres={}\n"
                    "sigma_thres={}\n"
                    "cutoff_hz={}\n"
                    "leak_rate_hz={}\n"
                    "shot_noise_rate_hz={}\n"
                    "refractory_period_s={}".format(
                        model, self.pos_thres, self.neg_thres,
                        self.sigma_thres, self.cutoff_hz,
                        self.leak_rate_hz, self.shot_noise_rate_hz,
                        self.refractory_period_s))

    def reset(self):
        '''resets so that next use will reinitialize the base frame
        '''
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0
        self.base_log_frame = None
        self.lp_log_frame0 = None  # lowpass stage 0
        self.lp_log_frame1 = None  # stage 1
        self.frame_counter = 0

    def _show(self, inp: np.ndarray):
        min = np.min(inp)
        norm = (np.max(inp) - min)
        if norm == 0:
            logger.warning('image is blank, max-min=0')
            norm = 1
        img = ((inp - min) / norm)
        cv2.imshow(__name__+':'+self.show_input, img)
        cv2.waitKey(30)

    def generate_events(self, new_frame, t_frame):
        """Compute events in new frame.

        With JAX Acceleration.

        Parameters
        ----------
        new_frame: np.ndarray
            [height, width]
        t_frame: float
            timestamp of new frame in float seconds

        Returns
        -------
        events: np.ndarray if any events, else None
            [N, 4], each row contains [timestamp, y cordinate,
            x cordinate, sign of event].
            # TODO validate that this order of x and y is correctly documented
        """
        #  base_frame: the change detector input,
        #              stores memorized brightness values
        # new_frame: the new intensity frame input
        # log_frame: the lowpass filtered brightness values
        if self.base_log_frame is None:
            self._init(new_frame)
            self.t_previous = t_frame
            return None
        self.frame_counter += 1

        if t_frame <= self.t_previous:
            raise ValueError(
                "this frame time={} must be later than "
                "previous frame time={}".format(t_frame, self.t_previous))

        # lin-log mapping
        log_new_frame = lin_log(new_frame)

        # Apply nonlinear lowpass filter here.
        # Filter is 2nd order lowpass IIR
        # that uses two internal state variables
        # to store stages of cascaded first order RC filters.
        # Time constant of the filter is proportional to
        # the intensity value (with offset to deal with DN=0)
        delta_time = t_frame - self.t_previous
        # logger.debug('delta_time={}'.format(delta_time))

        inten01 = None  # define for later
        if self.cutoff_hz > 0 or self.shot_noise_rate_hz > 0:  # will use later
            # limit max time constant to ~1/10 of white intensity level
            inten01 = rescale_intensity_frame(new_frame)

        # low pass filter
        self.lp_log_frame0, self.lp_log_frame1 = low_pass_filter(
            log_new_frame=log_new_frame,
            lp_log_frame0=self.lp_log_frame0,
            lp_log_frame1=self.lp_log_frame1,
            inten01=inten01,
            delta_time=delta_time,
            cutoff_hz=self.cutoff_hz)

        # # Noise: add infinite bandwidth white noise to samples
        # # after lowpass filtering,
        # # so that the noise scales up as intensity goes down.
        # # It will model the fact that total noise power is concentrated
        # # to low frequencies when photocurrent (and transconductance)
        # # is smaller.
        #  if self.shot_noise_rate_hz > 0:
        #      noise = (self.shot_noise_rate_hz)*(
        #          np.divide(np.random.randn(
        #              logNewFrame.shape[0],
        #              logNewFrame.shape[1]), inten01 + .1))
        #      # makes the darkest pixels (with DN=0) have sample
        #      # to sample 1-sigma noise of self.shot_noise_rate_hz
        #  else:
        #      noise = 0

        # Leak events: switch in diff change amp leaks at some rate
        # equivalent to some hz of ON events.
        # Actual leak rate depends on threshold for each pixel.
        # We want nominal rate leak_rate_Hz, so
        # R_l=(dI/dt)/Theta_on, so
        # R_l*Theta_on=dI/dt, so
        # dI=R_l*Theta_on*dt
        if self.leak_rate_hz > 0:
            self.base_log_frame = subtract_leak_current(
                base_log_frame=self.base_log_frame,
                leak_rate_hz=self.leak_rate_hz,
                delta_time=delta_time,
                pos_thres_nominal=self.pos_thres_nominal)

        # log intensity (brightness) change from memorized values is computed
        # from the difference between new input
        # (from lowpass of lin-log input) and the memorized value
        diff_frame = self.lp_log_frame1 - self.base_log_frame

        if self.show_input:
            if self.show_input == 'new_frame':
                self._show(new_frame)
            elif self.show_input == 'baseLogFrame':
                self._show(self.base_log_frame)
            elif self.show_input == 'lpLogFrame0':
                self._show(self.lp_log_frame0)
            elif self.show_input == 'lpLogFrame1':
                self._show(self.lp_log_frame1)
            elif self.show_input == 'diff_frame':
                self._show(diff_frame)
            else:
                logger.error("don't know about showing {}".format(
                    self.show_input))

        # generate event map
        pos_evts_frame, neg_evts_frame = compute_event_map(
            diff_frame, self.pos_thres, self.neg_thres)
        num_iters = max(pos_evts_frame.max(), neg_evts_frame.max())

        events = []

        for i in range(num_iters):
            print("I'm here {}/{}".format(i, num_iters))
            events_curr_iters = jnp.zeros((0, 4), dtype=jnp.float32)
            # intermediate timestamps are linearly spaced
            # they start after the t_start to make sure
            # that there is space from previous frame
            # they end at t_end
            # e.g. t_start=0, t_end=1, num_iters=2, i=0,1
            # ts=1*1/2, 2*1/2
            # num_iters+1 matches with the equation in the paper
            ts = self.t_previous + delta_time * (i + 1) / (num_iters+1)

            # for each iteration, compute the ON and OFF event locations
            # for that threshold amount of change or more,
            # these pixels need to output an event in this cycle
            # pos_cord = (pos_frame >= self.pos_thres * (i + 1))
            # neg_cord = (neg_frame >= self.neg_thres * (i + 1))
            # already have the number of events for each pixel in
            # pos_evts_frame, just find bool array of pixels with events in
            # this iteration of max # events
            pos_cord = (pos_evts_frame >= i+1) # it must be >= because we need to make event for each iteration up to total # events for that pixel
            neg_cord = (neg_evts_frame >= i+1)
            # generate events
            #  make a list of coordinates x,y addresses of events
            #  pos_event_xy = np.where(pos_cord)
            pos_event_xy = pos_cord.nonzero()
            num_pos_events = pos_event_xy[0].shape[0]
            #  neg_event_xy = np.where(neg_cord)
            neg_event_xy = neg_cord.nonzero()
            num_neg_events = neg_event_xy[0].shape[0]
            num_events = num_pos_events + num_neg_events

            self.num_events_on += num_pos_events
            self.num_events_off += num_neg_events
            self.num_events_total += num_events

            #  logger.info(
            #      f'frame/iteration: {self.frame_counter}/{i}'
            #      f'#on: {num_pos_events} #off: {num_neg_events}')

            # sort out the positive event and negative event
            if num_pos_events > 0:
                pos_events = np.hstack(
                    (jnp.ones((num_pos_events, 1), dtype=jnp.float32) * ts,
                     pos_event_xy[1][..., np.newaxis],
                     pos_event_xy[0][..., np.newaxis],
                     jnp.ones((num_pos_events, 1), dtype=jnp.float32) * 1))
            else:
                pos_events = jnp.zeros((0, 4), dtype=np.float32)

            if num_neg_events > 0:
                neg_events = np.hstack(
                    (jnp.ones((num_neg_events, 1), dtype=jnp.float32) * ts,
                     neg_event_xy[1][..., np.newaxis],
                     neg_event_xy[0][..., np.newaxis],
                     jnp.ones((num_neg_events, 1), dtype=jnp.float32) * -1))
            else:
                neg_events = jnp.zeros((0, 4), dtype=np.float32)

            events_tmp = jnp.vstack((pos_events, neg_events))

            # randomly order events to prevent bias to one corner
            #  if events_tmp.shape[0] != 0:
            #      np.random.shuffle(events_tmp)

            if num_events > 0:
                events_curr_iters = events_tmp
                #  events.append(events_tmp)

                if self.shot_noise_rate_hz > 0:

                    shot_on_events, shot_off_events = generate_shot_noise(
                        inten01=inten01,
                        base_log_frame=self.base_log_frame,
                        shot_noise_rate_hz=self.shot_noise_rate_hz,
                        delta_time=delta_time,
                        num_iters=num_iters,
                        pos_thres_pre_prob=self.pos_thres_pre_prob,
                        pos_thres=self.pos_thres,
                        neg_thres_pre_prob=self.neg_thres_pre_prob,
                        neg_thres=self.neg_thres,
                        ts=ts,
                        jax_key=self.jax_key)

                    events_curr_iters = jnp.vstack(
                        (events_curr_iters, shot_on_events, shot_off_events))

            # shuffle and append to the events collectors
            random.permutation(self.jax_key, events_curr_iters)
            events.append(events_curr_iters)

            if i == 0:
                # update the base frame only once,
                # after we know how many events per pixel
                # add to memorized brightness values
                # just the events we emitted.
                # don't add the remainder.
                # the next aps frame might have sufficient value
                # to trigger another event or it might not,
                # but we are correct in not storing
                # the current frame brightness
                if num_pos_events > 0:
                    self.base_log_frame += \
                        pos_evts_frame*pos_cord*self.pos_thres

                if num_neg_events > 0:
                    self.base_log_frame -= \
                        neg_evts_frame*neg_cord*self.neg_thres

        if len(events) > 0:
            events = jnp.vstack(events)
            if self.dvs_h5 is not None:
                # convert data to uint32 (microsecs) format
                temp_events = np.array(events)
                temp_events[:, 0] = temp_events[:, 0] * 1e6
                temp_events[temp_events[:, 3] == -1, 3] = 0
                temp_events = temp_events.astype(np.uint32)

                # save events
                self.dvs_h5_dataset.resize(
                   self.dvs_h5_dataset.shape[0] + temp_events.shape[0],
                   axis=0)

                self.dvs_h5_dataset[-temp_events.shape[0]:] = temp_events
                self.dvs_h5.flush()
            if self.dvs_aedat2 is not None:
                self.dvs_aedat2.appendEvents(events)
            if self.dvs_text is not None:
                self.dvs_text.appendEvents(events)

        self.t_previous = t_frame
        if len(events) > 0:
            return events
        else:
            return None
