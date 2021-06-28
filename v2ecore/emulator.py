"""
DVS simulator.
Compute events from input frames.
"""
import atexit
import os
import random

import cv2
import numpy as np
import logging
import h5py

import torch

from v2ecore.v2e_utils import checkAddSuffix
from v2ecore.output.aedat2_output import AEDat2Output
from v2ecore.output.ae_text_output import DVSTextOutput

from v2ecore.emulator_utils import lin_log
from v2ecore.emulator_utils import rescale_intensity_frame
from v2ecore.emulator_utils import low_pass_filter
from v2ecore.emulator_utils import subtract_leak_current
from v2ecore.emulator_utils import compute_event_map
from v2ecore.emulator_utils import generate_shot_noise

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
            output_height=None,
            device="cuda"):
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

        # torch device
        self.device = device

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
        if seed != 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

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

                # for events
                self.dvs_h5_dataset = self.dvs_h5.create_dataset(
                    name="events",
                    shape=(0, 4),
                    maxshape=(None, 4),
                    dtype="uint32",
                    compression="gzip")
            else:
                self.dvs_h5_dataset = None
            self.frame_h5_dataset = None
            self.frame_ts_dataset = None
            self.frame_ev_idx_dataset = None

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

    def prepare_storage(self, n_frames, frame_ts):
        # extra prepare for frame storage
        if self.dvs_h5:
            # for frame
            self.frame_h5_dataset = self.dvs_h5.create_dataset(
                name="frame",
                shape=(n_frames, self.output_height, self.output_width),
                dtype="uint8",
                compression="gzip")

            frame_ts_arr = np.array(frame_ts, dtype=np.float32)*1e6
            self.frame_ts_dataset = self.dvs_h5.create_dataset(
                name="frame_ts",
                shape=(n_frames,),
                data=frame_ts_arr.astype(np.uint32),
                dtype="uint32",
                compression="gzip")
            # corresponding event idx
            self.frame_ev_idx_dataset = self.dvs_h5.create_dataset(
                name="frame_idx",
                shape=(n_frames,),
                dtype="uint64",
                compression="gzip")
        else:
            self.frame_h5_dataset = None
            self.frame_ts_dataset = None
            self.frame_ev_idx_dataset = None

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
        self.lp_log_frame0 = self.base_log_frame.clone().detach()
        # 2nd stage is initialized to same,
        # so diff will be zero for first frame
        self.lp_log_frame1 = self.base_log_frame.clone().detach()

        # take the variance of threshold into account.
        if self.sigma_thres > 0:
            self.pos_thres = torch.normal(
                self.pos_thres, self.sigma_thres,
                size=first_frame_linear.shape,
                dtype=torch.float32).to(self.device)

            # to avoid the situation where the threshold is too small.
            self.pos_thres = torch.clamp(self.pos_thres, min=0.01)

            self.neg_thres = torch.normal(
                self.neg_thres, self.sigma_thres,
                size=first_frame_linear.shape,
                dtype=torch.float32).to(self.device)
            self.neg_thres = torch.clamp(self.neg_thres, min=0.01)

        # compute variable for shot-noise
        self.pos_thres_pre_prob = torch.div(
            self.pos_thres_nominal, self.pos_thres)
        self.neg_thres_pre_prob = torch.div(
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
            self.base_log_frame -= torch.rand(
                first_frame_linear.shape,
                dtype=torch.float32, device=self.device)*self.pos_thres

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
        inp = np.array(inp.cpu().data.numpy())
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
        # like a DAVIS, write frame into the file if it's HDF5
        if self.frame_h5_dataset is not None:
            # save frame data
            self.frame_h5_dataset[self.frame_counter] = \
                new_frame.astype(np.uint8)

        # update frame counter
        self.frame_counter += 1

        # convert into torch tensor
        new_frame = torch.tensor(new_frame, dtype=torch.float32,
                                 device=self.device)
        # base_frame: the change detector input,
        #              stores memorized brightness values
        # new_frame: the new intensity frame input
        # log_frame: the lowpass filtered brightness values
        if self.base_log_frame is None:
            self._init(new_frame)
            self.t_previous = t_frame
            return None

        if t_frame <= self.t_previous:
            raise ValueError(
                "this frame time={} must be later than "
                "previous frame time={}".format(t_frame, self.t_previous))

        # lin-log mapping
        log_new_frame = lin_log(new_frame)

        # compute time difference between this and the previous frame
        delta_time = t_frame - self.t_previous
        # logger.debug('delta_time={}'.format(delta_time))

        inten01 = None  # define for later
        if self.cutoff_hz > 0 or self.shot_noise_rate_hz > 0:  # will use later
            # Time constant of the filter is proportional to
            # the intensity value (with offset to deal with DN=0)
            # limit max time constant to ~1/10 of white intensity level
            inten01 = rescale_intensity_frame(new_frame.clone().detach())

        # Apply nonlinear lowpass filter here.
        # Filter is a 1st order lowpass IIR (can be 2nd order)
        # that uses two internal state variables
        # to store stages of cascaded first order RC filters.
        # Time constant of the filter is proportional to
        # the intensity value (with offset to deal with DN=0)
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

        # update the base frame, after we know how many events per pixel
        # add to memorized brightness values just the events we emitted.
        # don't add the remainder.
        # the next aps frame might have sufficient value to trigger
        # another event or it might not, but we are correct in not storing
        # the current frame brightness
        self.base_log_frame += pos_evts_frame*self.pos_thres
        self.base_log_frame -= neg_evts_frame*self.neg_thres

        # all events
        events = []

        # event timestamps at each iteration
        # intermediate timestamps are linearly spaced
        # they start after the t_start to make sure
        # that there is space from previous frame
        # they end at t_end
        # e.g. t_start=0, t_end=1, num_iters=2, i=0,1
        # ts=1*1/2, 2*1/2
        #  ts = self.t_previous + delta_time * (i + 1) / num_iters
        ts = torch.linspace(
            start=self.t_previous+delta_time/num_iters,
            end=self.t_previous+delta_time,
            steps=num_iters, dtype=torch.float32, device=self.device)

        # NOISE: add temporal noise here by
        # simple Poisson process that has a base noise rate
        # self.shot_noise_rate_hz.
        # If there is such noise event,
        # then we output event from each such pixel

        # the shot noise rate varies with intensity:
        # for lowest intensity the rate rises to parameter.
        # the noise is reduced by factor
        # SHOT_NOISE_INTEN_FACTOR for brightest intensities

        # This was in the loop, here we calculate loop-independent quantities
        if self.shot_noise_rate_hz > 0:
            SHOT_NOISE_INTEN_FACTOR = 0.25
            shot_noise_factor = (
                (self.shot_noise_rate_hz/2)*delta_time/num_iters) * \
                ((SHOT_NOISE_INTEN_FACTOR-1)*inten01+1)
            # =1 for inten=0 and SHOT_NOISE_INTEN_FACTOR for inten=1

            # probability for each pixel is
            # dt*rate*nom_thres/actual_thres.
            # That way, the smaller the threshold,
            # the larger the rate
            shot_ON_prob_this_sample = \
                shot_noise_factor*self.pos_thres_pre_prob
            shot_OFF_prob_this_sample = \
                shot_noise_factor*self.neg_thres_pre_prob

            # for shot noise
            rand01 = torch.rand(
                size=[num_iters]+list(inten01.shape),
                dtype=torch.float32,
                device=self.device)  # draw samples

        for i in range(num_iters):
            # events for this iteration
            events_curr_iter = []

            # already have the number of events for each pixel in
            # pos_evts_frame, just find bool array of pixels with events in
            # this iteration of max # events

            # it must be >= because we need to make event for
            # each iteration up to total # events for that pixel
            pos_cord = (pos_evts_frame >= i+1)
            neg_cord = (neg_evts_frame >= i+1)

            # generate events
            #  make a list of coordinates x,y addresses of events
            pos_event_xy = pos_cord.nonzero(as_tuple=True)
            neg_event_xy = neg_cord.nonzero(as_tuple=True)

            # update event stats
            num_pos_events = pos_event_xy[0].shape[0]
            num_neg_events = neg_event_xy[0].shape[0]
            num_events = num_pos_events + num_neg_events

            self.num_events_on += num_pos_events
            self.num_events_off += num_neg_events
            self.num_events_total += num_events

            # sort out the positive event and negative event
            if num_pos_events > 0:
                pos_events = torch.ones(
                    (num_pos_events, 4), dtype=torch.float32,
                    device=self.device)
                pos_events[:, 0] *= ts[i]
                pos_events[:, 1] = pos_event_xy[1]
                pos_events[:, 2] = pos_event_xy[0]

                events_curr_iter.append(pos_events)

            if num_neg_events > 0:
                neg_events = torch.ones(
                    (num_neg_events, 4), dtype=torch.float32,
                    device=self.device)
                neg_events[:, 0] *= ts[i]
                neg_events[:, 1] = neg_event_xy[1]
                neg_events[:, 2] = neg_event_xy[0]
                neg_events[:, 3] *= -1

                events_curr_iter.append(neg_events)

            # generate shot noise
            if self.shot_noise_rate_hz > 0:
                shot_on_events, shot_off_events, self.base_log_frame = \
                    generate_shot_noise(
                        shot_noise_factor=shot_noise_factor,
                        rand01=rand01[i],
                        base_log_frame=self.base_log_frame,
                        shot_ON_prob_this_sample=shot_ON_prob_this_sample,
                        shot_OFF_prob_this_sample=shot_OFF_prob_this_sample,
                        pos_thres=self.pos_thres,
                        neg_thres=self.neg_thres,
                        ts=ts[i])

                # add to event stats
                self.num_events_on += shot_on_events.shape[0]
                self.num_events_off += shot_off_events.shape[0]
                self.num_events_total += \
                    shot_on_events.shape[0]+shot_off_events.shape[0]

                # append noisy events
                events_curr_iter.append(shot_on_events)
                events_curr_iter.append(shot_off_events)

            # stack all events at this iteration
            events_curr_iter = torch.vstack(events_curr_iter)

            # shuffle and append to the events collectors
            if events_curr_iter.shape[0] > 0:
                idx = torch.randperm(events_curr_iter.shape[0])
                events_curr_iter = events_curr_iter[idx].view(
                    events_curr_iter.size())
                events.append(events_curr_iter)

            # the update is moved out from the loop
            #  if i == 0:
            #      if num_pos_events > 0:
            #          self.base_log_frame += \
            #              pos_evts_frame*pos_cord*self.pos_thres
            #
            #      if num_neg_events > 0:
            #          self.base_log_frame -= \
            #              neg_evts_frame*neg_cord*self.neg_thres

        if len(events) > 0:
            events = torch.vstack(events).cpu().data.numpy()
            if self.dvs_h5 is not None:
                # convert data to uint32 (microsecs) format
                temp_events = np.array(events, dtype=np.float32)
                temp_events[:, 0] = temp_events[:, 0] * 1e6
                temp_events[temp_events[:, 3] == -1, 3] = 0
                temp_events = temp_events.astype(np.uint32)

                # save events
                self.dvs_h5_dataset.resize(
                   self.dvs_h5_dataset.shape[0] + temp_events.shape[0],
                   axis=0)

                self.dvs_h5_dataset[-temp_events.shape[0]:] = temp_events

            if self.dvs_aedat2 is not None:
                self.dvs_aedat2.appendEvents(events)
            if self.dvs_text is not None:
                self.dvs_text.appendEvents(events)

        if self.frame_ev_idx_dataset is not None:
            # save frame event idx
            # determine after the events are added
            self.frame_ev_idx_dataset[self.frame_counter-1] = \
                self.dvs_h5_dataset.shape[0]

        self.t_previous = t_frame
        if len(events) > 0:
            return events
        else:
            return None
