"""
DVS simulator.
Compute events from input frames.
"""
import atexit
import logging
import math
import os
import pickle
import random
from typing import Optional

import cv2
import h5py
import numpy as np
import torch  # https://pytorch.org/docs/stable/torch.html
from screeninfo import get_monitors

from v2ecore.emulator_utils import compute_event_map, compute_photoreceptor_noise_voltage
from v2ecore.emulator_utils import generate_shot_noise
from v2ecore.emulator_utils import lin_log
from v2ecore.emulator_utils import low_pass_filter
from v2ecore.emulator_utils import rescale_intensity_frame
from v2ecore.emulator_utils import subtract_leak_current
from v2ecore.output.ae_text_output import DVSTextOutput
from v2ecore.output.aedat2_output import AEDat2Output
from v2ecore.output.aedat4_output import AEDat4Output
from v2ecore.v2e_utils import checkAddSuffix, v2e_quit, video_writer

# import rosbag # not yet for python 3

logger = logging.getLogger(__name__)


class EventEmulator(object):
    """compute events based on the input frame.
    - author: Tobi Delbruck, Yuhuang Hu, Zhe He
    - contact: tobi@ini.uzh.ch
    """

    # frames that can be displayed and saved to video with their plotting/display settings
    l255 = np.log(255)
    gr = (0, 255)  # display as 8 bit int gray image
    lg = (0, l255)  # display as log image with max ln(255)
    slg = (
        -l255 / 8,
        l255 / 8)  # display as signed log image with 1/8 of full scale for better visibility of faint contrast
    MODEL_STATES = {'new_frame': gr, 'log_new_frame': lg,
                    'lp_log_frame': lg, 'scidvs_highpass': slg, 'photoreceptor_noise_arr': slg, 'cs_surround_frame': lg,
                    'c_minus_s_frame': slg, 'base_log_frame': slg, 'diff_frame': slg}

    MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING = 1e-5

    SINGLE_PIXEL_STATES_FILENAME='pixel-states.dat'
    SINGLE_PIXEL_MAX_SAMPLES=10000

    # scidvs adaptation
    def scidvs_dvdt(self, v, tau=None):
        """

        Parameters
        ----------
            the input 'voltage',
        v:Tensor
            actually log intensity in base e units
        tau:Optional[Tensor]
            if None, tau is set internally

        Returns
        -------
        the time derivative of the signal

        """
        if tau is None:
            tau = EventEmulator.SCIDVS_TAU_S  # time constant for small signals = C/g
        # C = 100e-15
        # g = C/tau
        efold = 1 / 0.7  # efold of sinh conductance in log_e units, based on 1/kappa
        dvdt = torch.div(1,tau) * torch.sinh(v / efold)
        return dvdt

    SCIDVS_GAIN: float = 2  # gain after highpass
    SCIDVS_TAU_S: float = .01  # small signal time constant in seconds
    SCIDVS_TAU_COV: float = 0.5  # each pixel has its own time constant. The tau's have log normal distribution with this sigma

    def __init__(
            self,
            pos_thres: float = 0.2,
            neg_thres: float = 0.2,
            sigma_thres: float = 0.03,
            cutoff_hz: float = 0.0,
            leak_rate_hz: float = 0.1,
            refractory_period_s: float = 0.0,
            shot_noise_rate_hz: float = 0.0,  # rate in hz of temporal noise events
            photoreceptor_noise: bool = False,
            leak_jitter_fraction: float = 0.1,
            noise_rate_cov_decades: float = 0.1,
            seed: int = 0,
            output_folder: str = None,
            dvs_h5: str = None,
            dvs_aedat2: str = None,
            dvs_aedat4: str = None,
            dvs_text: str = None,
            # change as you like to see 'baseLogFrame',
            # 'lpLogFrame', 'diff_frame'
            show_dvs_model_state: str = None,
            save_dvs_model_state: bool = False,
            output_width: int = None,
            output_height: int = None,
            device: str = "cuda",
            cs_lambda_pixels: float = None,
            cs_tau_p_ms: float = None,
            hdr: bool = False,
            scidvs: bool = False,
            record_single_pixel_states=None,
            label_signal_noise=False
    ):
        """
        Parameters
        ----------
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
        photoreceptor_noise: bool
            model photoreceptor noise to create the desired shot noise rate
        seed: int, default=0
            seed for random threshold variations,
            fix it to nonzero value to get same mismatch every time
        dvs_aedat2, dvs_aedat4, dvs_h5, dvs_text: str
            names of output data files or None
        show_dvs_model_state: List[str],
            None or 'new_frame','diff_frame' etc; see EventEmulator.MODEL_STATES
        output_folder: str
            Path to optional model state videos
        output_width: int,
            width of output in pixels
        output_height: int,
            height of output in pixels
        device: str
            device, either 'cpu' or 'cuda' (selected automatically by caller depending on GPU availability)
        cs_lambda_pixels: float
            space constant of surround in pixels, or None to disable surround inhibition
        cs_tau_p_ms: float
            time constant of lowpass filter of surround in ms or 0 to make surround 'instantaneous'
        hdr: bool
            Treat input as HDR floating point logarithmic gray scale with 255 input scaled as ln(255)=5.5441
        scidvs: bool
            Simulate the high gain adaptive photoreceptor SCIDVS pixel
        record_single_pixel_states: tuple
            Record this pixel states to 'pixel_states.npy'
        label_signal_noise: bool
            Record signal and noise event labels to a CSV file
        """

        self.no_events_warning_count = 0
        logger.info(
            "ON/OFF log_e temporal contrast thresholds: "
            "{} / {} +/- {}".format(pos_thres, neg_thres, sigma_thres))

        self.reset()
        self.t_previous = 0  # time of previous frame

        self.dont_show_list = []  # list of frame types to not show and not print warnings for except for once
        self.show_list = []  # list of named windows shown for internal states
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
        self.photoreceptor_noise = photoreceptor_noise
        self.photoreceptor_noise_vrms: Optional[float] = None
        self.photoreceptor_noise_arr: Optional[
            np.ndarray] = None  # separate noise source that is lowpass filtered to provide intensity-independent noise to add to intensity-dependent filtered photoreceptor output
        if photoreceptor_noise:
            if shot_noise_rate_hz == 0:
                logger.warning(
                    '--photoreceptor_noise is specified but --shot_noise_rate_hz is 0; set a finite rate of shot noise events per pixel')
                v2e_quit(1)
            if cutoff_hz == 0:
                logger.warning(
                    '--photoreceptor_noise is specified but --cutoff_hz is zero; set a finite photoreceptor cutoff frequency')
                v2e_quit(1)
            self.photoreceptor_noise_samples = []

        self.leak_jitter_fraction = leak_jitter_fraction
        self.noise_rate_cov_decades = noise_rate_cov_decades

        self.SHOT_NOISE_INTEN_FACTOR = 0.25 # this factor models the slight increase of shot noise with intensity

        # output properties
        self.output_folder = output_folder
        self.output_width = output_width
        self.output_height = output_height  # set on first frame
        self.show_dvs_model_state = show_dvs_model_state
        self.save_dvs_model_state = save_dvs_model_state
        self.video_writers: dict[str, video_writer] = {}  # list of avi file writers for saving model state videos

        # generate jax key for random process
        if seed != 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # h5 output
        self.output_folder = output_folder
        self.dvs_h5 = dvs_h5
        self.dvs_h5_dataset = None
        self.frame_h5_dataset = None
        self.frame_ts_dataset = None
        self.frame_ev_idx_dataset = None

        # aedat or text output
        self.dvs_aedat2 = dvs_aedat2
        self.dvs_aedat4 = dvs_aedat4
        self.dvs_text = dvs_text

        # event stats
        self.num_events_total = 0
        self.num_events_on = 0
        self.num_events_off = 0
        self.frame_counter = 0

        # csdvs
        self.cs_steps_warning_printed = False
        self.cs_steps_taken = []
        self.cs_alpha_warning_printed = False
        self.cs_tau_p_ms = cs_tau_p_ms
        self.cs_lambda_pixels = cs_lambda_pixels
        self.cs_surround_frame: Optional[torch.Tensor] = None  # surround frame state
        self.csdvs_enabled = False  # flag to run center surround DVS emulation
        if self.cs_lambda_pixels is not None:
            self.csdvs_enabled = True
            # prepare kernels
            self.cs_tau_h_ms = 0 \
                if (self.cs_tau_p_ms is None or self.cs_tau_p_ms == 0) \
                else self.cs_tau_p_ms / (self.cs_lambda_pixels ** 2)
            lat_res = 1 / (self.cs_lambda_pixels ** 2)
            trans_cond = 1 / self.cs_lambda_pixels
            logger.debug(
                f'lateral resistance R={lat_res:.2g}Ohm, transverse transconductance g={trans_cond:.2g} Siemens, Rg={(lat_res * trans_cond):.2f}')
            self.cs_k_hh = torch.tensor([[[[0, 1, 0],
                                           [1, -4, 1],
                                           [0, 1, 0]]]], dtype=torch.float32).to(self.device)
            # self.cs_k_pp = torch.tensor([[[[0, 0, 0],
            #                                [0, 1, 0],
            #                                [0, 0, 0]]]], dtype=torch.float32).to(self.device)
            logger.info(f'Center-surround parameters:\n\t'
                        f'cs_tau_p_ms: {self.cs_tau_p_ms}\n\t'
                        f'cs_tau_h_ms:  {self.cs_tau_h_ms}\n\t'
                        f'cs_lambda_pixels:  {self.cs_lambda_pixels:.2f}\n\t'
                        )

        # label signal and noise events
        self.label_signal_noise=label_signal_noise

        # record pixel
        self.record_single_pixel_states=record_single_pixel_states
        self.single_pixel_sample_count=0
        if self.record_single_pixel_states is None:
            self.single_pixel_states=None
        else:
            if not (type(self.record_single_pixel_states) is tuple):
                raise ValueError(f'--record_single_pixel_states {self.record_single_pixel_states} should be a tuple, e.g. (10,20)')
            if len(self.record_single_pixel_states)!=2:
                raise ValueError(f'--record_single_pixel_states {self.record_single_pixel_states} should have two pixel addresses (x,y)')
            for i in self.record_single_pixel_states:
                if not (type(i) is int):
                    raise ValueError(f'--record_single_pixel_states {self.record_single_pixel_states} should have two integer-value pixel addresses (x,y)')
            self.single_pixel_states={
                'time':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'new_frame':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'base_log_frame':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'lp_log_frame':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'log_new_frame':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'pos_thres':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'neg_thres':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'diff_frame':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'final_neg_evts_frame':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
                'final_pos_evts_frame':np.empty(self.SINGLE_PIXEL_MAX_SAMPLES)*np.nan,
            } # dict to be filled with arrays of states (and time array)

        self.log_input = hdr
        if self.log_input:
            logger.info('Treating input as log-encoded HDR input')

        self.scidvs = scidvs
        if self.scidvs:
            logger.info('Modeling potential SCIDVS pixel with nonlinear CR highpass amplified log intensity')

        try:
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

            if dvs_aedat2:
                path = os.path.join(self.output_folder, dvs_aedat2)
                path = checkAddSuffix(path, '.aedat')
                logger.info('opening AEDAT-2.0 output file ' + path)
                self.dvs_aedat2 = AEDat2Output(
                    path, output_width=self.output_width,
                    output_height=self.output_height, label_signal_noise=self.label_signal_noise)

            if dvs_aedat4:
                path = os.path.join(self.output_folder, dvs_aedat4)
                path = checkAddSuffix(path, '.aedat4')
                logger.info('opening AEDAT-4.0 output file ' + path)
                self.dvs_aedat4 = AEDat4Output(
                    path)

            if dvs_text:
                path = os.path.join(self.output_folder, dvs_text)
                path = checkAddSuffix(path, '.txt')
                logger.info('opening text DVS output file ' + path)
                self.dvs_text = DVSTextOutput(path,label_signal_noise=self.label_signal_noise)



        except Exception as e:
            logger.error(f'Output file exception "{e}" (maybe you need to specify a supported DVS camera type?)')
            raise e

        self.screen_width = 1600
        self.screen_height = 1200
        try:
            mi = get_monitors()
            for m in mi:
                if m.is_primary:
                    self.screen_width = int(m.width)
                    self.screen_height = int(m.height)
        except Exception as e:
            logger.warning(f'cannot get screen size for window placement: {e}')

        if self.show_dvs_model_state is not None and len(self.show_dvs_model_state) == 1 and self.show_dvs_model_state[
            0] == 'all':
            logger.info(f'will show all model states that exist from {EventEmulator.MODEL_STATES.keys()}')
            self.show_dvs_model_state = EventEmulator.MODEL_STATES.keys()

        self.show_norms = {}  # dict of named tuples (min,max) for each displayed model state that adapts to fit displayed values into 0-1 range for rendering

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

            frame_ts_arr = np.array(frame_ts, dtype=np.float32) * 1e6
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
        if len(self.cs_steps_taken) > 1:
            mean_staps = np.mean(self.cs_steps_taken)
            std_steps = np.std(self.cs_steps_taken)
            median_steps = np.median(self.cs_steps_taken)
            logger.info(
                f'CSDVS steps statistics: mean+std= {mean_staps:.0f} + {std_steps:.0f} (median= {median_steps:.0f})')
        if self.dvs_h5 is not None:
            self.dvs_h5.close()

        if self.dvs_aedat2 is not None:
            self.dvs_aedat2.close()

        if self.dvs_aedat4 is not None:
            self.dvs_aedat4.close()

        if self.dvs_text is not None:
            try:
                self.dvs_text.close()
            except:
                pass

        for vw in self.video_writers:
            logger.info(f'closing video AVI {vw}')
            self.video_writers[vw].release()

        if not self.record_single_pixel_states is None:
            self.save_recorded_single_pixel_states()

    def save_recorded_single_pixel_states(self):
        try:
            with open(self.SINGLE_PIXEL_STATES_FILENAME,'wb') as outfile:
                pickle.dump(self.single_pixel_states, outfile, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f'saved single pixel states with {self.single_pixel_sample_count} samples to {self.SINGLE_PIXEL_STATES_FILENAME}')
        except Exception as e:
            logger.error(f'could not save pickled pixel states, got {e}')

    def _init(self, first_frame_linear):
        """

        Parameters:
        ----------
        first_frame_linear: np.ndarray
            the first frame, used to initialize data structures

        Returns:
            new instance
        -------

        """
        logger.debug(
            'initializing random temporal contrast thresholds '
            'from from base frame')
        # base_frame are memorized lin_log pixel values
        self.diff_frame = None

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

        if self.scidvs and EventEmulator.SCIDVS_TAU_COV > 0:
            self.scidvs_tau_arr = EventEmulator.SCIDVS_TAU_S * (
                torch.exp(torch.normal(0, EventEmulator.SCIDVS_TAU_COV, size=first_frame_linear.shape,
                                       dtype=torch.float32).to(self.device)))

        # If leak is non-zero, then initialize each pixel memorized value
        # some fraction of ON threshold below first frame value, to create leak
        # events from the start; otherwise leak would only gradually
        # grow over time as pixels spike.
        # do this *AFTER* we determine randomly distributed thresholds
        # (and use the actual pixel thresholds)
        # otherwise low threshold pixels will generate
        # a burst of events at the first frame
        if self.leak_rate_hz > 0:
            # no justification for this subtraction after having the
            # new leak rate model
            #  self.base_log_frame -= torch.rand(
            #      first_frame_linear.shape,
            #      dtype=torch.float32, device=self.device)*self.pos_thres

            # set noise rate array, it's a log-normal distribution
            self.noise_rate_array = torch.randn(
                first_frame_linear.shape, dtype=torch.float32,
                device=self.device)
            self.noise_rate_array = torch.exp(
                math.log(10) * self.noise_rate_cov_decades * self.noise_rate_array)

        # refractory period
        if self.refractory_period_s > 0:
            self.timestamp_mem = torch.zeros(
                first_frame_linear.shape, dtype=torch.float32,
                device=self.device) - self.refractory_period_s

    def set_dvs_params(self, model: str):
        if model == 'clean':
            self.pos_thres = 0.2
            self.neg_thres = 0.2
            self.sigma_thres = 0.02
            self.cutoff_hz = 0
            self.leak_rate_hz = 0
            self.leak_jitter_fraction = 0
            self.noise_rate_cov_decades = 0
            self.shot_noise_rate_hz = 0  # rate in hz of temporal noise events
            self.refractory_period_s = 0

        elif model == 'noisy':
            self.pos_thres = 0.2
            self.neg_thres = 0.2
            self.sigma_thres = 0.05
            self.cutoff_hz = 30
            self.leak_rate_hz = 0.1
            # rate in hz of temporal noise events
            self.shot_noise_rate_hz = 5.0
            self.refractory_period_s = 0
            self.leak_jitter_fraction = 0.1
            self.noise_rate_cov_decades = 0.1
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

        # add names of new states to potentially show with --show_model_states all
        self.new_frame: Optional[np.ndarray] = None # new frame that comes in [height, width]
        self.log_new_frame: Optional[np.ndarray] = None #  [height, width]
        self.lp_log_frame: Optional[np.ndarray] = None  # lowpass stage 0
        self.lp_log_frame: Optional[np.ndarray] = None  # stage 1
        self.cs_surround_frame: Optional[np.ndarray] = None
        self.c_minus_s_frame: Optional[np.ndarray] = None
        self.base_log_frame: Optional[np.ndarray] = None # memorized log intensities at change detector
        self.diff_frame: Optional[np.ndarray] = None  # [height, width]
        self.scidvs_highpass: Optional[np.ndarray] = None
        self.scidvs_previous_photo: Optional[np.ndarray] = None
        self.scidvs_tau_arr: Optional[np.ndarray] = None

        self.frame_counter = 0

    def _show(self, inp: torch.Tensor, name: str):
        """
        Shows the ndarray in window, and save frame to avi file if self.save_dvs_model_state==True.
        The displayed image is normalized according to its type (grayscale, log, or signed log).
        Parameters
        ----------
        inp: the array
        name: label for window

        Returns
        -------
        None
        """

        img = np.array(inp.cpu().data.numpy())
        (min, max) = EventEmulator.MODEL_STATES[name]

        img = (img - min) / (max - min)

        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        if not name in self.show_list:
            d = len(self.show_list) * 200
            # (x,y,w,h)=cv2.getWindowImageRect(name)
            cv2.moveWindow(name, int(self.screen_width / 8 + d), int(self.screen_height / 8 + d / 2))
            self.show_list.append(name)
            if self.save_dvs_model_state:
                fn = os.path.join(self.output_folder, name + '.avi')
                vw = video_writer(fn, self.output_height, self.output_width)
                self.video_writers[name] = vw
        cv2.putText(img, f'fr:{self.frame_counter} t:{self.t_previous:.4f}s', org=(0, self.output_height),
                    fontScale=1.3, color=(0, 0, 0), fontFace=cv2.FONT_HERSHEY_PLAIN, thickness=1)
        cv2.putText(img, f'fr:{self.frame_counter} t:{self.t_previous:.4f}s', org=(1, self.output_height - 1),
                    fontScale=1.3, color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_PLAIN, thickness=1)
        cv2.imshow(name, img)
        if self.save_dvs_model_state:
            self.video_writers[name].write(
                cv2.cvtColor((img * 255).astype(np.uint8),
                             cv2.COLOR_GRAY2BGR))

    def generate_events(self, new_frame, t_frame):
        """Compute events in new frame.

        Parameters
        ----------
        new_frame: np.ndarray
            [height, width], NOTE y is first dimension, like in matlab the column, x is 2nd dimension, i.e. row.
        t_frame: float
            timestamp of new frame in float seconds

        Returns
        -------
        events: np.ndarray if any events, else None
            [N, 4], each row contains [timestamp, x coordinate, y coordinate, sign of event (+1 ON, -1 OFF)].
            NOTE x,y, NOT y,x.
        """

        # base_frame: the change detector input,
        #              stores memorized brightness values
        # new_frame: the new intensity frame input
        # log_frame: the lowpass filtered brightness values

        # like a DAVIS, write frame into the file if it's HDF5
        if self.frame_h5_dataset is not None:
            # save frame data
            self.frame_h5_dataset[self.frame_counter] = \
                new_frame.astype(np.uint8)

        # update frame counter
        self.frame_counter += 1

        if t_frame < self.t_previous:
            raise ValueError(
                "this frame time={} must be later than "
                "previous frame time={}".format(t_frame, self.t_previous))

        # compute time difference between this and the previous frame
        delta_time = t_frame - self.t_previous
        # logger.debug('delta_time={}'.format(delta_time))

        if self.log_input and new_frame.dtype != np.float32:
            logger.warning('log_frame is True but input frome is not np.float32 datatype')

        # convert into torch tensor
        self.new_frame = torch.tensor(new_frame, dtype=torch.float64,
                                      device=self.device)
        # lin-log mapping, if input is not already float32 log input
        self.log_new_frame = lin_log(self.new_frame) if not self.log_input else self.new_frame

        inten01 = None  # define for later
        if self.cutoff_hz > 0 or self.shot_noise_rate_hz > 0:  # will use later
            # Time constant of the filter is proportional to
            # the intensity value (with offset to deal with DN=0)
            # limit max time constant to ~1/10 of white intensity level
            inten01 = rescale_intensity_frame(self.new_frame.clone().detach())  # TODO assumes 8 bit

        # Apply nonlinear lowpass filter here.
        # Filter is a 1st order lowpass IIR (can be 2nd order)
        # that uses two internal state variables
        # to store stages of cascaded first order RC filters.
        # Time constant of the filter is proportional to
        # the intensity value (with offset to deal with DN=0)
        if self.base_log_frame is None:
            # initialize 1st order IIR to first input
            self.lp_log_frame = self.log_new_frame
            self.photoreceptor_noise_arr = torch.zeros_like(self.lp_log_frame)

        self.lp_log_frame = low_pass_filter(
            log_new_frame=self.log_new_frame,
            lp_log_frame=self.lp_log_frame,
            inten01=inten01,
            delta_time=delta_time,
            cutoff_hz=self.cutoff_hz)

        # add photoreceptor noise if we are using photoreceptor noise to create shot noise
        if self.photoreceptor_noise and not self.base_log_frame is None:  # only add noise after the initial values are memorized and we can properly lowpass filter the noise
            self.photoreceptor_noise_vrms = compute_photoreceptor_noise_voltage(
                shot_noise_rate_hz=self.shot_noise_rate_hz, f3db=self.cutoff_hz, sample_rate_hz=1 / delta_time,
                pos_thr=self.pos_thres_nominal, neg_thr=self.neg_thres_nominal, sigma_thr=self.sigma_thres)
            noise = self.photoreceptor_noise_vrms * torch.randn(self.log_new_frame.shape, dtype=torch.float32,
                                                                device=self.device)
            self.photoreceptor_noise_arr = low_pass_filter(noise, self.photoreceptor_noise_arr, None, delta_time,
                                                           self.cutoff_hz)
            self.photoreceptor_noise_samples.append(
                self.photoreceptor_noise_arr[0, 0].cpu().item())  # todo debugging can remove
            # std=np.std(self.photoreceptor_noise_samples)

        # surround computations by time stepping the diffuser
        if self.csdvs_enabled:
            self._update_csdvs(delta_time)

        if self.base_log_frame is None:
            self._init(new_frame)
            if not self.csdvs_enabled:
                self.base_log_frame = self.lp_log_frame
            else:
                self.base_log_frame = self.lp_log_frame - self.cs_surround_frame  # init base log frame (input to diff) to DC value, TODO check might not be correct to avoid transient

            return None  # on first input frame we just setup the state of all internal nodes of pixels

        if self.scidvs:
            if self.scidvs_highpass is None:
                self.scidvs_highpass = torch.zeros_like(self.lp_log_frame)
                self.scidvs_previous_photo = torch.clone(self.lp_log_frame).detach()
            self.scidvs_highpass += (self.lp_log_frame - self.scidvs_previous_photo) \
                                    - delta_time * self.scidvs_dvdt(self.scidvs_highpass,self.scidvs_tau_arr)
            self.scidvs_previous_photo = torch.clone(self.lp_log_frame)

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
                pos_thres=self.pos_thres,
                leak_jitter_fraction=self.leak_jitter_fraction,
                noise_rate_array=self.noise_rate_array)

        # log intensity (brightness) change from memorized values is computed
        # from the difference between new input
        # (from lowpass of lin-log input) and the memorized value

        # take input from either photoreceptor or amplified high pass nonlinear filtered scidvs
        photoreceptor = EventEmulator.SCIDVS_GAIN * self.scidvs_highpass if self.scidvs else self.lp_log_frame

        if not self.csdvs_enabled:
            self.diff_frame = photoreceptor + self.photoreceptor_noise_arr - self.base_log_frame
        else:
            self.c_minus_s_frame = photoreceptor + self.photoreceptor_noise_arr - self.cs_surround_frame
            self.diff_frame = self.c_minus_s_frame - self.base_log_frame

        if not self.show_dvs_model_state is None:
            for s in self.show_dvs_model_state:
                if not s in self.dont_show_list:
                    f = getattr(self, s, None)
                    if f is None:
                        logger.error(f'{s} does not exist so we cannot show it')
                        self.dont_show_list.append(s)
                    else:
                        self._show(f, s)  # show the frame f with name s
            k = cv2.waitKey(30)
            if k == 27 or k == ord('x'):
                v2e_quit()

        # generate event map
        # print(f'\ndiff_frame max={torch.max(self.diff_frame)} pos_thres mean={torch.mean(self.pos_thres)} expect {int(torch.max(self.diff_frame)/torch.mean(self.pos_thres))} max events')
        pos_evts_frame, neg_evts_frame = compute_event_map(
            self.diff_frame, self.pos_thres, self.neg_thres)
        max_num_events_any_pixel = max(pos_evts_frame.max(),
                                       neg_evts_frame.max())  # max number of events in any pixel for this interframe
        max_num_events_any_pixel=max_num_events_any_pixel.cpu().numpy().item() # turn singleton tensor to scalar
        if max_num_events_any_pixel > 100:
            logger.warning(f'Too many events generated for this frame: num_iter={max_num_events_any_pixel}>100 events')

        # to assemble all events
        events = torch.empty((0, 4), dtype=torch.float32, device=self.device)  # ndarray shape (N,4) where N is the number of events are rows are [t,x,y,p]
        # event timestamps at each iteration
        # min_ts_steps timestamps are linearly spaced
        # they start after the self.t_previous to make sure
        # that there is interval from previous frame
        # they end at t_frame.
        # delta_time=t_frame - self.t_previous
        # e.g. t_start=0, t_end=1, min_ts_steps=2, i=0,1
        # ts=1*1/2, 2*1/2
        #  ts = self.t_previous + delta_time * (i + 1) / min_ts_steps
        # if min_ts_steps==1, then there is only a single timestamp at t_frame
        min_ts_steps=max_num_events_any_pixel if max_num_events_any_pixel>0 else 1
        ts_step = delta_time / min_ts_steps
        ts = torch.linspace(
            start=self.t_previous+ts_step,
            end=t_frame,
            steps=min_ts_steps, dtype=torch.float32, device=self.device)
        # print(f'ts={ts}')

        # record final events update
        final_pos_evts_frame = torch.zeros(
            pos_evts_frame.shape, dtype=torch.int32, device=self.device)
        final_neg_evts_frame = torch.zeros(
            neg_evts_frame.shape, dtype=torch.int32, device=self.device)

        if max_num_events_any_pixel == 0 and self.no_events_warning_count<100:
            logger.warning(f'no signal events generated for frame #{self.frame_counter:,} at t={t_frame:.4f}s')
            self.no_events_warning_count+=1
            # max_num_events_any_pixel = 1
        else: # there are signal events to generate
            for i in range(max_num_events_any_pixel):
                # events for this iteration

                # already have the number of events for each pixel in
                # pos_evts_frame, just find bool array of pixels with events in
                # this iteration of max # events

                # it must be >= because we need to make event for
                # each iteration up to total # events for that pixel
                pos_cord = (pos_evts_frame >= i + 1)
                neg_cord = (neg_evts_frame >= i + 1)


                # filter events with refractory_period
                # only filter when refractory_period_s is large enough
                # otherwise, pass everything
                # TODO David Howe figured out that the reference level was resetting to the log photoreceptor value at event generation,
                # NOT at the value at the end of the refractory period.
                # Brian McReynolds thinks that this effect probably only makes a significant difference if the temporal resolution of the signal
                # is high enough so that dt is less than one refractory period.
                if self.refractory_period_s > ts_step:
                    pos_time_since_last_spike = (
                            pos_cord * ts[i] - self.timestamp_mem)
                    neg_time_since_last_spike = (
                            neg_cord * ts[i] - self.timestamp_mem)

                    # filter the events
                    pos_cord = (
                            pos_time_since_last_spike > self.refractory_period_s)
                    neg_cord = (
                            neg_time_since_last_spike > self.refractory_period_s)

                    # assign new history
                    self.timestamp_mem = torch.where(
                        pos_cord, ts[i], self.timestamp_mem)
                    self.timestamp_mem = torch.where(
                        neg_cord, ts[i], self.timestamp_mem)

                # update event count frames with the shot noise
                final_pos_evts_frame += pos_cord
                final_neg_evts_frame += neg_cord

                # generate events
                # make a list of coordinates x,y addresses of events
                # torch.nonzero(as_tuple=True)
                # Returns a tuple of 1-D tensors, one for each dimension in input,
                # each containing the indices (in that dimension) of all non-zero elements of input .

                # pos_event_xy and neg_event_xy each return two 1-d tensors each with same length of the number of events
                #   Tensor 0 is list of y addresses (first dimension in pos_cord input)
                #   Tensor 1 is list of x addresses
                pos_event_xy = pos_cord.nonzero(as_tuple=True)
                neg_event_xy = neg_cord.nonzero(as_tuple=True)

                events_curr_iter = self.get_event_list_from_coords(pos_event_xy, neg_event_xy, ts[i])

                # shuffle and append to the events collectors
                if events_curr_iter is not None:
                    idx = torch.randperm(events_curr_iter.shape[0])
                    events_curr_iter = events_curr_iter[idx].view(events_curr_iter.size())
                    events=torch.cat((events,events_curr_iter))

                # end of iteration over max_num_events_any_pixel

        # NOISE: add shot temporal noise here by
        # simple Poisson process that has a base noise rate
        # self.shot_noise_rate_hz.
        # If there is such noise event,
        # then we output event from each such pixel. Note this is too simplified to model
        # alternating ON/OFF noise; see --photoreceptor_noise option for that type of noise
        # Advantage here is to be able to label signal and noise events.

        # the shot noise rate varies with intensity:
        # for lowest intensity the rate rises to parameter.
        # the noise is reduced by factor
        # SHOT_NOISE_INTEN_FACTOR for brightest intensities

        shot_on_cord, shot_off_cord = None, None

        num_signal_events=len(events)
        signnoise_label=torch.ones(num_signal_events,dtype=torch.bool, device=self.device) if self.label_signal_noise else None # all signal so far

        # This was in the loop, here we calculate loop-independent quantities
        if self.shot_noise_rate_hz > 0 and not self.photoreceptor_noise:
            # generate all the noise events for this entire input frame; there could be (but unlikely) several per pixel but only 1 on or off event is returned here
            shot_on_cord, shot_off_cord = generate_shot_noise(
                shot_noise_rate_hz=self.shot_noise_rate_hz,
                delta_time=delta_time,
                shot_noise_inten_factor=self.SHOT_NOISE_INTEN_FACTOR,
                inten01=inten01,
                pos_thres_pre_prob=self.pos_thres_pre_prob,
                neg_thres_pre_prob=self.neg_thres_pre_prob)

            # noise_on_xy and noise_off_xy each are two 1-d tensors each with same length of the number of events
            #   Tensor 0 is list of y addresses (first dimension in pos_cord input)
            #   Tensor 1 is list of x addresses
            shot_on_xy = shot_on_cord.nonzero(as_tuple=True)
            shot_off_xy = shot_off_cord.nonzero(as_tuple=True)

            # give noise events the last timestamp generated for any signal event from this frame
            shot_noise_events = self.get_event_list_from_coords(shot_on_xy, shot_off_xy, ts[-1])

            # append the shot noise events and shuffle in, keeping track of labels if labeling
            # append to the signal events but don't shuffle since this causes nonmonotonic timestamps
            if shot_noise_events is not None:
                num_shot_noise_events=len(shot_noise_events)
                events=torch.cat((events, shot_noise_events), dim=0) # stack signal events before noise events, [N,4]
                num_total_events=len(events)
                # idx = torch.randperm(num_total_events)  # makes timestamps nonmonotonic
                # events = events[idx].view(events.size())
                if self.label_signal_noise:
                    noise_label=torch.zeros((num_shot_noise_events),dtype=torch.bool, device=self.device)
                    signnoise_label=torch.cat((signnoise_label,noise_label))
                    signnoise_label=signnoise_label[idx].view(signnoise_label.size())

        # update base log frame according to the final
        # number of output events
        # update the base frame, after we know how many events per pixel
        # add to memorized brightness values just the events we emitted.
        # don't add the remainder.
        # the next aps frame might have sufficient value to trigger
        # another event, or it might not, but we are correct in not storing
        # the current frame brightness
        #  self.base_log_frame += pos_evts_frame*self.pos_thres
        #  self.base_log_frame -= neg_evts_frame*self.neg_thres

        self.base_log_frame += final_pos_evts_frame * self.pos_thres # TODO should this be self.lp_log_frame ? I.e. output of lowpass photoreceptor?
        self.base_log_frame -= final_neg_evts_frame * self.neg_thres

        # however, if we made a shot noise event, then just memorize the log intensity at this point, so that the pixels are reset and forget the log intensity input
        if not self.photoreceptor_noise and self.shot_noise_rate_hz>0:
            self.base_log_frame[shot_on_xy]=self.lp_log_frame[shot_on_xy]
            self.base_log_frame[shot_off_xy]=self.lp_log_frame[shot_off_xy]


        if len(events) > 0:
            events = events.cpu().data.numpy() # # ndarray shape (N,4) where N is the number of events are rows are [t,x,y,p]
            timestamps=events[:,0]
            if np.any(np.diff(timestamps)<0):
                idx=np.argwhere(np.diff(timestamps)<0)
                logger.warning(f'nonmonotonic timestamp(s) at indices {idx}')
            if signnoise_label is not None:
                signnoise_label=signnoise_label.cpu().numpy()
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
                self.dvs_aedat2.appendEvents(events, signnoise_label=signnoise_label)

            if self.dvs_aedat4 is not None:
                self.dvs_aedat4.appendEvents(events, signnoise_label=signnoise_label)
                
            if self.dvs_text is not None:
                if self.label_signal_noise:
                    self.dvs_text.appendEvents(events, signnoise_label=signnoise_label)
                else:
                    self.dvs_text.appendEvents(events)

        if self.frame_ev_idx_dataset is not None:
            # save frame event idx
            # determine after the events are added
            self.frame_ev_idx_dataset[self.frame_counter - 1] = \
                self.dvs_h5_dataset.shape[0]

        if not self.record_single_pixel_states is None:
            if self.single_pixel_sample_count<self.SINGLE_PIXEL_MAX_SAMPLES:
                k=self.single_pixel_sample_count
                if k%250==0:
                    logger.info(f'recorded {k} single pixel states')
                self.single_pixel_states['time'][k]=t_frame
                self.single_pixel_states['new_frame'][k]=new_frame[self.record_single_pixel_states]
                self.single_pixel_states['base_log_frame'][k]=self.base_log_frame[self.record_single_pixel_states]
                self.single_pixel_states['lp_log_frame'][k]=self.lp_log_frame[self.record_single_pixel_states]
                self.single_pixel_states['log_new_frame'][k]=self.log_new_frame[self.record_single_pixel_states]
                if type(self.pos_thres) is float:
                    self.single_pixel_states['pos_thres'][k]=self.pos_thres
                else:
                    self.single_pixel_states['pos_thres'][k]=self.pos_thres[self.record_single_pixel_states]
                if type(self.neg_thres) is float:
                    self.single_pixel_states['neg_thres'][k]=self.neg_thres
                else:
                    self.single_pixel_states['neg_thres'][k]=self.neg_thres[self.record_single_pixel_states]
                self.single_pixel_states['diff_frame'][k]=self.diff_frame[self.record_single_pixel_states]
                self.single_pixel_states['final_neg_evts_frame'][k]=final_neg_evts_frame[self.record_single_pixel_states]
                self.single_pixel_states['final_pos_evts_frame'][k]=final_pos_evts_frame[self.record_single_pixel_states]
                self.single_pixel_sample_count+=1
            else:
                self.save_recorded_single_pixel_states()
                self.record_single_pixel_states=None
        # assign new time
        self.t_previous = t_frame
        if len(events) > 0:

            # debug TODO remove
            tsout = events[:, 0]
            tsoutdiff = np.diff(tsout)
            if (np.any(tsoutdiff < 0)):
                print('nonmonotonic timestamp in events')

            return events # ndarray shape (N,4) where N is the number of events are rows are [t,x,y,p]. Confirmed by Tobi Oct 2023
        else:
            return None

    def get_event_list_from_coords(self, pos_event_xy, neg_event_xy, ts):
        """ Gets event list from ON and OFF event coordinate lists.
        :param pos_event_xy: Tensor[2,n] where n is number of ON events, [0,n] are y addresses and [1,n] are x addresses
        :param neg_event_xy: Tensor[2,m] where m is number of ON events, [0,m] are y addresses and [1,m] are x addresses
        :param ts: the timestamp given to all events (scalar)
        :returns: Tensor[n+m,4] of AER [t, x, y, p]
        """
        # update event stats
        num_pos_events = pos_event_xy[0].shape[0]
        num_neg_events = neg_event_xy[0].shape[0]
        num_events = num_pos_events + num_neg_events
        events_curr_iter=None
        if num_events > 0:
            # following will update stats for all events (signal and shot noise)
            self.num_events_on += num_pos_events
            self.num_events_off += num_neg_events
            self.num_events_total += num_events

            # events_curr_iter is 2d array [N,4] with 2nd dimension [t,x,y,p]
            events_curr_iter = torch.ones(  # set all elements 1 so that polarities start out positive ON events
                (num_events, 4), dtype=torch.float32,
                device=self.device)
            events_curr_iter[:, 0] *= ts  # put all timestamps into events

            # pos_event cords
            # events_curr_iter is 2d array [N,4] with 2nd dimension [t,x,y,p]. N is the number of events from this frame
            # we replace the x's (element 1) and y's (element 2) with the on event coordinates in the first num_pos_coord entries of events_curr_iter
            events_curr_iter[:num_pos_events, 1] = pos_event_xy[1]  # tensor 1 of pos_event_xy is x addresses
            events_curr_iter[:num_pos_events, 2] = pos_event_xy[0]  # tensor 0 of pos_event_xy is y addresses

            # neg event cords
            # we replace the x's (element 1) and y's (element 2) with the off event coordinates in the remaining entries num_pos_events: entries of events_curr_iter
            events_curr_iter[num_pos_events:, 1] = neg_event_xy[1]
            events_curr_iter[num_pos_events:, 2] = neg_event_xy[0]
            events_curr_iter[num_pos_events:, 3] = -1  # neg events polarity is -1 so flip the signs
        return events_curr_iter

    def _update_csdvs(self, delta_time):
        if self.cs_surround_frame is None:
            self.cs_surround_frame = self.lp_log_frame.clone().detach()  # detach makes true clone decoupled from torch computation tree
        else:
            # we still need to simulate dynamics even if "instantaneous", unfortunately it will be really slow with Euler stepping and
            # no gear-shifting
            # TODO change to compute steady-state 'instantaneous' solution by better method than Euler stepping
            abs_min_tau_p = 1e-9
            tau_p = abs_min_tau_p if (
                    self.cs_tau_p_ms is None or self.cs_tau_p_ms == 0) else self.cs_tau_p_ms * 1e-3
            tau_h = abs_min_tau_p / (self.cs_lambda_pixels ** 2) if (
                    self.cs_tau_h_ms is None or self.cs_tau_h_ms == 0) else self.cs_tau_h_ms * 1e-3
            min_tau = min(tau_p, tau_h)
            # if min_tau < abs_min_tau_p:
            #     min_tau = abs_min_tau_p
            NUM_STEPS_PER_TAU = 5
            num_steps = int(np.ceil((delta_time / min_tau) * NUM_STEPS_PER_TAU))
            actual_delta_time = delta_time / num_steps
            if num_steps > 1000 and not self.cs_steps_warning_printed:
                if self.cs_tau_p_ms == 0:
                    logger.warning(
                        f'You set time constant cs_tau_p_ms to zero which set the minimum tau of {abs_min_tau_p}s')
                logger.warning(
                    f'CSDVS timestepping of diffuser could take up to {num_steps} '
                    f'steps per frame for Euler delta time {actual_delta_time:.3g}s; '
                    f'simulation of each frame will terminate when max change is smaller than {EventEmulator.MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING}')
                self.cs_steps_warning_printed = True

            alpha_p = actual_delta_time / tau_p
            alpha_h = actual_delta_time / tau_h
            if alpha_p >= 1 or alpha_h >= 1:
                logger.error(
                    f'CSDVS update alpha (of IIR update) is too large; simulation would explode: '
                    f'alpha_p={alpha_p:.3f} alpha_h={alpha_h:.3f}')
                self.cs_alpha_warning_printed = True
                v2e_quit(1)
            if alpha_p > .25 or alpha_h > .25:
                logger.warning(
                    f'CSDVS update alpha (of IIR update) is too large; simulation will be inaccurate: '
                    f'alpha_p={alpha_p:.3f} alpha_h={alpha_h:.3f}')
                self.cs_alpha_warning_printed = True
            p_ten = torch.unsqueeze(torch.unsqueeze(self.lp_log_frame, 0), 0)
            h_ten = torch.unsqueeze(torch.unsqueeze(self.cs_surround_frame, 0), 0)
            padding = torch.nn.ReplicationPad2d(1)
            max_change = 2 * EventEmulator.MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING
            steps = 0
            while steps < num_steps and max_change > EventEmulator.MAX_CHANGE_TO_TERMINATE_EULER_SURROUND_STEPPING:
                if not self.show_dvs_model_state is None and steps % 100 == 0:
                    cv2.pollKey()  # allow movement of windows and resizing
                diff = p_ten - h_ten
                p_term = alpha_p * diff
                # For the conv2d, unfortunately the zero padding pulls down the border pixels,
                # so we use replication padding to reduce this effect on border.
                # TODO check if possible to implement some form of open circuit resistor termination condition by correct padding
                h_conv = torch.conv2d(padding(h_ten.float()), self.cs_k_hh.float())
                h_term = alpha_h * h_conv
                change_ten = p_term + h_term  # change_ten is the change in the diffuser voltage
                max_change = torch.max(
                    torch.abs(change_ten)).item()  # find the maximum absolute change in any diffuser pixel
                h_ten += change_ten
                steps += 1

            self.cs_steps_taken.append(steps)
            self.cs_surround_frame = torch.squeeze(h_ten)


if __name__ == "__main__":
    # define a emulator
    emulator = EventEmulator(
        pos_thres=0.2,
        neg_thres=0.2,
        sigma_thres=0.03,
        cutoff_hz=200,
        leak_rate_hz=1,
        shot_noise_rate_hz=10,
        device="cuda",
    )

    cap = cv2.VideoCapture(
        os.path.join(os.environ["HOME"], "v2e_tutorial_video.avi"))

    # num of frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS: {}".format(fps))
    num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Num of frames: {}".format(num_of_frames))

    duration = num_of_frames / fps
    delta_t = 1 / fps
    current_time = 0.

    print("Clip Duration: {}s".format(duration))
    print("Delta Frame Tiem: {}s".format(delta_t))
    print("=" * 50)

    new_events = None

    idx = 0
    # Only Emulate the first 10 frame
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is True and idx < 10:
            # convert it to Luma frame
            luma_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("=" * 50)
            print("Current Frame {} Time {}".format(idx, current_time))
            print("-" * 50)

            # # emulate events
            new_events = emulator.generate_events(luma_frame, current_time)

            # update time
            current_time += delta_t

            # print event stats
            if new_events is not None:
                num_events = new_events.shape[0]
                start_t = new_events[0, 0]
                end_t = new_events[-1, 0]
                event_time = (new_events[-1, 0] - new_events[0, 0])
                event_rate_kevs = (num_events / delta_t) / 1e3

                print("Number of Events: {}\n"
                      "Duration: {}\n"
                      "Start T: {:.5f}\n"
                      "End T: {:.5f}\n"
                      "Event Rate: {:.2f}KEV/s".format(
                    num_events, event_time, start_t, end_t,
                    event_rate_kevs))
            idx += 1
            print("=" * 50)
        else:
            break

    cap.release()
