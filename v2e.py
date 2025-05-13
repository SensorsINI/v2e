#!/usr/bin/env python
"""
Python code for extracting frames from video file and synthesizing fake DVS
events from this video after SuperSloMo has generated interpolated
frames from the original video frames.

@author: Tobi Delbruck, Yuhuang Hu, Zhe He
@contact: tobi@ini.uzh.ch, yuhuang.hu@ini.uzh.ch, zhehe@student.ethz.ch
"""
# todo refractory period for pixel

import glob
import argparse
import importlib
import sys

import argcomplete
import cv2
import numpy as np
import os
from tempfile import TemporaryDirectory
from engineering_notation import EngNumber as eng  # only from pip
from tqdm import tqdm

import torch

import v2ecore.desktop as desktop
from v2ecore.base_synthetic_input import base_synthetic_input
from v2ecore.v2e_utils import all_images, read_image, \
    check_lowpass, v2e_quit
from v2ecore.v2e_utils import set_output_dimension
from v2ecore.v2e_utils import set_output_folder
from v2ecore.v2e_utils import ImageFolderReader
from v2ecore.v2e_args import v2e_args, write_args_info, SmartFormatter
from v2ecore.v2e_args import v2e_check_dvs_exposure_args
from v2ecore.v2e_args import NO_SLOWDOWN
from v2ecore.renderer import EventRenderer, ExposureMode
from v2ecore.slomo import SuperSloMo
from v2ecore.emulator import EventEmulator
from v2ecore.v2e_utils import inputVideoFileDialog
import logging
import time
from typing import Optional, Any

logging.basicConfig()
root = logging.getLogger()
LOGGING_LEVEL=logging.INFO
root.setLevel(LOGGING_LEVEL)  # todo move to info for production
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
logging.addLevelName(
    logging.DEBUG, "\033[1;36m%s\033[1;0m" % logging.getLevelName(
        logging.DEBUG)) # cyan foreground
logging.addLevelName(
    logging.INFO, "\033[1;34m%s\033[1;0m" % logging.getLevelName(
        logging.INFO)) # blue foreground
logging.addLevelName(
    logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(
        logging.WARNING)) # red foreground
logging.addLevelName(
    logging.ERROR, "\033[38;5;9m%s\033[1;0m" % logging.getLevelName(
        logging.ERROR)) # red background
logger = logging.getLogger(__name__)

# torch device
torch_device:str = torch.device('cuda' if torch.cuda.is_available() else 'cpu').type
logger.info(f'torch device is {torch_device}')
if torch_device=='cpu':
    logger.warning('CUDA GPU acceleration of pytorch operations is not available; '
                   'see https://pytorch.org/get-started/locally/ '
                   'to generate the correct conda install command to enable GPU-accelerated CUDA.')

# may only apply to windows
try:
    #  from scripts.regsetup import description
    from gooey import Gooey  # pip install Gooey
except Exception as e:
    logger.info(f"{e}: Gooey GUI builder not available, "
                   f"will use command line arguments.\n"
                   f"Install with 'pip install Gooey if you want a no-arg GUI to invoke v2e'. See README")


def get_args():
    """ proceses input arguments
    :returns: (args_namespace,other_args,command_line) """
    parser = argparse.ArgumentParser(
        description='v2e: generate simulated DVS events from video.',
        epilog='Run with no --input to open file dialog', allow_abbrev=True,
        formatter_class=SmartFormatter)

    parser = v2e_args(parser)

    #  parser.add_argument(
    #      "--rotate180", type=bool, default=False,
    #      help="rotate all output 180 deg.")
    # https://kislyuk.github.io/argcomplete/#global-completion
    # Shellcode (only necessary if global completion is not activated -
    # see Global completion below), to be put in e.g. .bashrc:
    # eval "$(register-python-argcomplete v2e.py)"
    argcomplete.autocomplete(parser)

    (args_namespace,other_args) = parser.parse_known_args() # change to known arguments so that synthetic input module can take arguments
    command_line=''
    for a in sys.argv:
        command_line=command_line+' '+a
    return (args_namespace,other_args,command_line)


def main():
    try:
        ga = Gooey(get_args, program_name="v2e", default_size=(575, 600))
        logger.info(
            "Use --ignore-gooey to disable GUI and "
            "run with command line arguments")
        ga()
    except Exception as e:
        logger.info(
            f'{e}: Gooey package GUI not available, using command line arguments. \n'
            f'You can try to install with "pip install Gooey"')

    (args,other_args,command_line) = get_args()

    # set input file
    input_file = args.input
    synthetic_input:str = args.synthetic_input

    if synthetic_input is not None and input_file is not None:
        logger.error(f'Both input_file {input_file} and synthetic_input {synthetic_input} are specified - you can only specify one of them')
        v2e_quit(1)

    if synthetic_input is None and input_file is None:
        try:
            input_file = inputVideoFileDialog()
            if input_file is None:
                logger.info('no file selected, quitting')
                v2e_quit()
        except Exception as e:
            logger.error(f'no input file specified and cannot show input file dialog; are you running without graphical display? ({e})')
            v2e_quit(1)

    # Set output folder
    output_folder = set_output_folder(
        args.output_folder,
        input_file,
        args.unique_output_folder if not args.overwrite else False,
        args.overwrite,
        args.output_in_place if (not synthetic_input) else False,
        logger)

    # Set output width and height based on the arguments
    output_width, output_height = set_output_dimension(
        args.output_width, args.output_height,
        args.dvs128, args.dvs240, args.dvs346,
        args.dvs640, args.dvs1024,
        logger)

    # Visualization
    avi_frame_rate = args.avi_frame_rate
    dvs_vid = args.dvs_vid  if not args.skip_video_output else None
    dvs_vid_full_scale = args.dvs_vid_full_scale
    vid_orig = args.vid_orig if not args.skip_video_output else None
    vid_slomo = args.vid_slomo if not args.skip_video_output else None
    preview = not args.no_preview


    # setup synthetic input classes and method
    synthetic_input_module = None
    synthetic_input_class = None
    synthetic_input_instance:Optional[base_synthetic_input] = None
    synthetic_input_next_frame_method = None
    if synthetic_input is not None:
        try:
            synthetic_input_module = importlib.import_module(synthetic_input)
            if '.' in synthetic_input:
                classname=synthetic_input[synthetic_input.rindex('.')+1:]
            else:
                classname=synthetic_input
            synthetic_input_class:synthetic_input = getattr(
                synthetic_input_module, classname)
            vid_path=os.path.join(output_folder, vid_orig) if not vid_orig is None else None
            synthetic_input_instance:base_synthetic_input = synthetic_input_class(
                width=output_width, height=output_height,
                preview=not args.no_preview, arg_list=other_args, avi_path=vid_path,parent_args=args) #TODO output folder might not be unique, could write to first output folder

            if not isinstance(synthetic_input_instance,base_synthetic_input):
                logger.error(f'synthetic input instance of {synthetic_input} is of type {type(synthetic_input_instance)}, but it should be a sublass of synthetic_input;'
                             f'there is no guarentee that it implements the necessary methods')
            synthetic_input_next_frame_method = getattr(
                synthetic_input_class, 'next_frame')
            synthetic_input_total_frames_method = getattr(
                synthetic_input_class, 'total_frames')


            srcNumFramesToBeProccessed=synthetic_input_instance.total_frames()

            logger.info(
                f'successfully instanced {synthetic_input_instance} with method {synthetic_input_next_frame_method}:'
                '{synthetic_input_module.__doc__}')

        except ModuleNotFoundError as e:
            logger.error(f'Could not import {synthetic_input}: {e}')
            v2e_quit(1)
        except AttributeError as e:
            logger.error(f'{synthetic_input} method incorrect?: {e}')
            v2e_quit(1)

    # check to make sure there are no other arguments that might be bogus misspelled arguments in case
    # we don't have synthetic input class to pass these to
#     if synthetic_input_instance is None and len(other_args)>0:
#         logger.error(f'There is no synthetic input class specified but there are extra arguments {other_args} that are probably incorrect')
#         v2e_quit(1)



    # Writing the info file
    infofile = write_args_info(args, output_folder,other_args,command_line)

    fh = logging.FileHandler(infofile,mode='a')
    fh.setLevel(LOGGING_LEVEL)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)




    num_frames = 0
    srcNumFramesToBeProccessed = 0
    srcDurationToBeProcessed=float("NaN")

   # input file checking
    #  if (not input_file or not os.path.isfile(input_file)
    #      or not os.path.isdir(input_file)) \
    #          and not base_synthetic_input:
    if (not synthetic_input):
        if not os.path.isfile(input_file) and not os.path.isdir(input_file):
            logger.error('input file {} does not exist'.format(input_file))
            v2e_quit(1)
        if os.path.isdir(input_file):
            if len(os.listdir(input_file))==0:
                logger.error(f'input folder {input_file} is empty')
                v2e_quit(1)




    # define video parameters
    # the input start and stop time, may be round to actual
    # frame timestamp
    input_start_time = args.start_time
    input_stop_time = args.stop_time
    def is_float(element: Any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    if not input_start_time is None and not input_stop_time is None and is_float(input_start_time) and is_float(input_stop_time) and input_stop_time<=input_start_time:
        logger.error(f'stop time {input_stop_time} must be later than start time {input_start_time}')
        v2e_quit(1)

    input_slowmotion_factor: float = args.input_slowmotion_factor
    input_frame_rate:Optional[float] =args.input_frame_rate
    timestamp_resolution: float = args.timestamp_resolution
    auto_timestamp_resolution: bool = args.auto_timestamp_resolution
    disable_slomo: bool = args.disable_slomo
    slomo = None  # make it later on

    if not disable_slomo and auto_timestamp_resolution is False \
            and timestamp_resolution is None:
        logger.error(
            'if --auto_timestamp_resolution=False, '
            'then --timestamp_resolution must be set to '
            'some desired DVS event timestamp resolution in seconds, '
            'e.g. 0.01')
        v2e_quit()

    if auto_timestamp_resolution is True \
            and timestamp_resolution is not None:
        logger.info(
            f'auto_timestamp_resolution=True and '
            f'timestamp_resolution={timestamp_resolution}: '
            f'Limiting automatic upsampling to maximum timestamp interval.')

    # DVS pixel thresholds
    pos_thres = args.pos_thres
    neg_thres = args.neg_thres
    sigma_thres = args.sigma_thres
    record_single_pixel_states=args.record_single_pixel_states

    # Cutoff and noise frequencies
    cutoff_hz = args.cutoff_hz
    leak_rate_hz = args.leak_rate_hz
    if leak_rate_hz > 0 and sigma_thres == 0:
        logger.warning(
            'leak_rate_hz>0 but sigma_thres==0, '
            'so all leak events will be synchronous')
    shot_noise_rate_hz = args.shot_noise_rate_hz



    # Event saving options
    dvs_h5 = args.dvs_h5
    dvs_aedat2 = args.dvs_aedat2
    dvs_aedat4 = args.dvs_aedat4
    dvs_text = args.dvs_text
    # signal noise output CSV file
    label_signal_noise=args.label_signal_noise
    if label_signal_noise and dvs_text is None and dvs_aedat2 is None and dvs_aedat4 is None:
        logger.error('if you specify --label_signal_noise you must specify --dvs_text and/or --dvs_aedat2 and/or --dvs_aedat4')
        v2e_quit(1)
    if label_signal_noise and args.photoreceptor_noise:
        logger.error('if you specify --label_signal_noise you cannot use --photoreceptor_noise option')
        v2e_quit(1)
    if label_signal_noise and shot_noise_rate_hz==0:
        logger.error('You specified --label_signal_noise, but --shot_noise_rate=0 and there will be no noise events')
        v2e_quit(1)

    # Debug feature: if show slomo stats
    slomo_stats_plot = args.slomo_stats_plot
    #  rotate180 = args.rotate180  # never used, consider removing
    batch_size = args.batch_size

    # DVS exposure
    exposure_mode, exposure_val, area_dimension = \
        v2e_check_dvs_exposure_args(args)
    if exposure_mode == ExposureMode.DURATION:
        dvsFps = 1. / exposure_val


    time_run_started = time.time()

    slomoTimestampResolutionS = None

    if synthetic_input is None:
        logger.info("opening video input file " + input_file)

        if os.path.isdir(input_file):
            if input_frame_rate is None:
                logger.error(
                    "When the video is presented as a folder, "
                    "The user must set --input_frame_rate manually")
                v2e_quit(1)

            cap = ImageFolderReader(input_file, args.input_frame_rate)
            srcFps = cap.frame_rate
            srcNumFrames = cap.num_frames

        else:
            cap = cv2.VideoCapture(input_file)
            srcFps = cap.get(cv2.CAP_PROP_FPS)
            srcNumFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if input_frame_rate is not None:
                logger.info(f'Input video frame rate {srcFps}Hz is overridden by command line argument --input_frame_rate={args.input_frame_rate}')
                srcFps=args.input_frame_rate

        if cap is not None:
            # set the output width and height from first image in folder, but only if they were not already set
            set_size = False
            if output_height is None and hasattr(cap,'frame_height'):
                set_size = True
                output_height = cap.frame_height
            if output_width is None and hasattr(cap,'frame_width'):
                set_size = True
                output_width = cap.frame_width
            if set_size:
                logger.warning(
                    f'From input frame automatically set DVS output_width={output_width} and/or output_height={output_height}. '
                    f'This may not be desired behavior. \nCheck DVS camera sizes arguments.')
                time.sleep(5);
            elif output_height is None or output_width is None:
                logger.warning(
                    'Could not read video frame size from video input and so could not automatically set DVS output size. \nCheck DVS camera sizes arguments.')

        # Check frame rate and number of frames
        if srcFps == 0:
            logger.error(
                'source {} fps is 0; v2e needs to have a timescale '
                'for input video'.format(input_file))
            v2e_quit()

        if srcNumFrames < 2:
            logger.warning(
                'num frames is less than 2, probably cannot be determined '
                'from cv2.CAP_PROP_FRAME_COUNT')

        srcTotalDuration = (srcNumFrames-1)/srcFps
        # the index of the frames, from 0 to srcNumFrames-1
        start_frame = int(srcNumFrames*(input_start_time/srcTotalDuration)) \
            if input_start_time else 0
        stop_frame = int(srcNumFrames*(input_stop_time/srcTotalDuration)) \
            if input_stop_time else srcNumFrames-1
        srcNumFramesToBeProccessed = stop_frame-start_frame+1
        # the duration to be processed, should subtract 1 frame when
        # calculating duration
        srcDurationToBeProcessed = (srcNumFramesToBeProccessed-1)/srcFps

        # redefining start and end time using the time calculated
        # from the frames, the minimum resolution there is
        start_time = start_frame/srcFps
        stop_time = stop_frame/srcFps

        srcFrameIntervalS = (1./srcFps)/input_slowmotion_factor

        slowdown_factor = NO_SLOWDOWN  # start with factor 1 for upsampling
        if disable_slomo:
            logger.warning(
                'slomo interpolation disabled by command line option; '
                'output DVS timestamps will have source frame interval '
                'resolution')
            # time stamp resolution equals to source frame interval
            slomoTimestampResolutionS = srcFrameIntervalS
        elif not auto_timestamp_resolution:
            slowdown_factor = int(
                np.ceil(srcFrameIntervalS/timestamp_resolution))
            if slowdown_factor < NO_SLOWDOWN:
                slowdown_factor = NO_SLOWDOWN
                logger.warning(
                    'timestamp resolution={}s is >= source '
                    'frame interval={}s, will not upsample'
                    .format(timestamp_resolution, srcFrameIntervalS))
            elif slowdown_factor > 100 and cutoff_hz == 0:
                logger.warning(
                    f'slowdown_factor={slowdown_factor} is >100 but '
                    'cutoff_hz={cutoff_hz}. We have observed that '
                    'numerical errors in SuperSloMo can cause noise '
                    'that makes fake events at the upsampling rate. '
                    'Recommend to set physical cutoff_hz, '
                    'e.g. --cutoff_hz=200 (or leave the default cutoff_hz)')
            slomoTimestampResolutionS = srcFrameIntervalS/slowdown_factor

            logger.info(
                f'--auto_timestamp_resolution is False, '
                f'srcFps={srcFps}Hz '
                f'input_slowmotion_factor={input_slowmotion_factor}, '
                f'real src FPS={srcFps*input_slowmotion_factor}Hz, '
                f'srcFrameIntervalS={eng(srcFrameIntervalS)}s, '
                f'timestamp_resolution={eng(timestamp_resolution)}s, '
                f'so SuperSloMo will use slowdown_factor={slowdown_factor} '
                f'and have '
                f'slomoTimestampResolutionS={eng(slomoTimestampResolutionS)}s')

            if slomoTimestampResolutionS > timestamp_resolution:
                logger.warning(
                    'Upsampled src frame intervals of {}s is larger than\n '
                    'the desired DVS timestamp resolution of {}s'
                    .format(slomoTimestampResolutionS, timestamp_resolution))

            check_lowpass(cutoff_hz, 1/slomoTimestampResolutionS, logger)
        else:  # auto_timestamp_resolution
            if timestamp_resolution is not None:
                slowdown_factor = int(
                    np.ceil(srcFrameIntervalS/timestamp_resolution))

                logger.info(
                    f'--auto_timestamp_resolution=True and '
                    f'timestamp_resolution={eng(timestamp_resolution)}s: '
                    f'source video will be automatically upsampled but '
                    f'with at least upsampling factor of {slowdown_factor}')
            else:
                logger.info(
                    '--auto_timestamp_resolution=True and '
                    'timestamp_resolution is not set: '
                    'source video will be automatically upsampled to '
                    'limit maximum interframe motion to 1 pixel')

        # the SloMo model, set no SloMo model if no slowdown
        if not disable_slomo and \
                (auto_timestamp_resolution or slowdown_factor != NO_SLOWDOWN):
            slomo = SuperSloMo(
                model=args.slomo_model,
                auto_upsample=auto_timestamp_resolution,
                upsampling_factor=slowdown_factor,
                video_path=None if args.skip_video_output else output_folder,
                vid_orig=None if args.skip_video_output else vid_orig,
                vid_slomo=None if args.skip_video_output else vid_slomo,
                preview=preview, batch_size=batch_size)

    if not synthetic_input and not auto_timestamp_resolution:
        logger.info(
            f'\n events will have timestamp resolution '
            f'{eng(slomoTimestampResolutionS)}s,')
        if exposure_mode == ExposureMode.DURATION \
                and dvsFps > (1 / slomoTimestampResolutionS):
            logger.warning(
                'DVS video frame rate={}Hz is larger than '
                'the effective DVS frame rate of {}Hz; '
                'DVS video will have blank frames'.format(
                    dvsFps, (1 / slomoTimestampResolutionS)))

    if not synthetic_input:
        logger.info(
            'Source video {} has total {} frames with total duration {}s. '
            '\nSource video is {}fps with slowmotion_factor {} '
            '(frame interval {}s),'
            '\nWill convert {} frames {} to {}\n'
            '(From {}s to {}s, duration {}s)'
            .format(input_file, srcNumFrames, eng(srcTotalDuration),
                    eng(srcFps), eng(input_slowmotion_factor),
                    eng(srcFrameIntervalS),
                    stop_frame-start_frame+1, start_frame, stop_frame,
                    start_time, stop_time, (stop_time-start_time)))

        if exposure_mode == ExposureMode.DURATION:
            dvsNumFrames = np.floor(
                dvsFps*srcDurationToBeProcessed/input_slowmotion_factor)
            dvsDuration = dvsNumFrames/dvsFps
            dvsPlaybackDuration = dvsNumFrames/avi_frame_rate
            start_time = start_frame/srcFps
            stop_time = stop_frame/srcFps  # todo something replicated here, already have start and stop times

            logger.info('v2e DVS video will have constant-duration frames \n'
                        'at {}fps (accumulation time {}s), '
                        '\nDVS video will have {} frames with duration {}s '
                        'and playback duration {}s\n'
                        .format(eng(dvsFps), eng(1 / dvsFps),
                                dvsNumFrames, eng(dvsDuration),
                                eng(dvsPlaybackDuration)))
        elif exposure_mode==ExposureMode.SOURCE:
            logger.info(f'v2e DVS video will have constant-duration frames \n'
                        f'at the source video {eng(srcFps)} fps (accumulation time {eng(srcFrameIntervalS)}s)')
        else:
            logger.info(
                'v2e DVS video will have constant-count '
                'frames with {} events), '
                .format(exposure_val))

    # check one more time that we have an output width and height
    if output_width is None or output_height is None:
        logger.error("Either or both of output_width or output_height is None,\n"
                     "which means that they were not specified or could not be inferred from the input video. \n "
                     "Please see options for DVS camera sizes. \nYou can try the option --dvs346 for DAVIS346 camera as one well-supported option.")
        v2e_quit(1)
    num_pixels=output_width*output_height

    hdr: bool = args.hdr
    if hdr:
        logger.info('Treating input as HDR logarithmic video')

    scidvs:bool=args.scidvs
    if scidvs:
        logger.info('Simulating SCIDVS pixel')

    emulator = EventEmulator(
        pos_thres=pos_thres, neg_thres=neg_thres,
        sigma_thres=sigma_thres, cutoff_hz=cutoff_hz,
        leak_rate_hz=leak_rate_hz, shot_noise_rate_hz=shot_noise_rate_hz, photoreceptor_noise=args.photoreceptor_noise,
        leak_jitter_fraction=args.leak_jitter_fraction,
        noise_rate_cov_decades=args.noise_rate_cov_decades,
        refractory_period_s=args.refractory_period,
        seed=args.dvs_emulator_seed,
        output_folder=output_folder, dvs_h5=dvs_h5, dvs_aedat2=dvs_aedat2, dvs_aedat4 = dvs_aedat4,
        dvs_text=dvs_text, show_dvs_model_state=args.show_dvs_model_state,
        save_dvs_model_state=args.save_dvs_model_state,
        output_width=output_width, output_height=output_height,
        device=torch_device,
        cs_lambda_pixels=args.cs_lambda_pixels, cs_tau_p_ms=args.cs_tau_p_ms,
        hdr=hdr,
        scidvs=scidvs,
        record_single_pixel_states=record_single_pixel_states,
        label_signal_noise=label_signal_noise
    )

    if args.dvs_params is not None:
        logger.warning(
            f'--dvs_param={args.dvs_params} option overrides your '
            f'selected options for threshold, threshold-mismatch, '
            f'leak and shot noise rates')
        emulator.set_dvs_params(args.dvs_params)

    eventRenderer = EventRenderer(
        output_path=output_folder,
        dvs_vid=dvs_vid, preview=preview, full_scale_count=dvs_vid_full_scale,
        exposure_mode=exposure_mode,
        exposure_value=exposure_val,
        area_dimension=area_dimension,
        avi_frame_rate=args.avi_frame_rate)

    if synthetic_input_next_frame_method is not None:
        # array to batch events for rendering to DVS frames
        events = np.zeros((0, 4), dtype=np.float32)
        (fr, fr_time) = synthetic_input_instance.next_frame()
        num_frames+=1
        i = 0
        with tqdm(total=synthetic_input_instance.total_frames(),
                  desc='dvs', unit='fr') as pbar:
            with torch.no_grad():
                while fr is not None:
                    newEvents = emulator.generate_events(fr, fr_time)
                    pbar.update(1)
                    i += 1
                    if newEvents is not None and newEvents.shape[0] > 0 \
                            and not args.skip_video_output:
                        events = np.append(events, newEvents, axis=0)
                        events = np.array(events)
                        if i % batch_size == 0:
                            eventRenderer.render_events_to_frames(
                                events, height=output_height,
                                width=output_width)
                            events = np.zeros((0, 4), dtype=np.float32)
                    (fr, fr_time) = synthetic_input_instance.next_frame()
                    num_frames+=1
            # process leftover events
            if len(events) > 0 and not args.skip_video_output:
                eventRenderer.render_events_to_frames(
                    events, height=output_height, width=output_width)
    else:  # video file folder or (avi/mp4) file input
        # timestamps of DVS start at zero and end with
        # span of video we processed
        srcVideoRealProcessedDuration = (stop_time-start_time) / \
            input_slowmotion_factor
        num_frames = srcNumFramesToBeProccessed
        inputHeight = None
        inputWidth = None
        inputChannels = None
        if start_frame > 0:
            logger.info('skipping to frame {}'.format(start_frame))
            for i in tqdm(range(start_frame), unit='fr', desc='src'):
                if isinstance(cap,ImageFolderReader):
                    if i<start_frame-1:
                        ret,_=cap.read(skip=True)
                    else:
                        ret, _ = cap.read()
                else:
                    ret, _ = cap.read()
                if not ret:
                    raise ValueError(
                        'something wrong, got to end of file before '
                        'reaching start_frame')

        logger.info(
            'processing frames {} to {} from video input'.format(
                start_frame, stop_frame))

        c_l=0
        c_r=None
        c_t=0
        c_b=None
        if args.crop is not None:
            c=args.crop
            if len(c)!=4:
                logger.error(f'--crop must have 4 elements (you specified --crop={args.crop}')
                v2e_quit(1)

            c_l=c[0] if c[0] > 0 else 0
            c_r=-c[1] if c[1]>0 else None
            c_t=c[2] if c[2]>0 else 0
            c_b=-c[3] if c[3]>0 else None
            logger.info(f'cropping video by (left,right,top,bottom)=({c_l},{c_r},{c_t},{c_b})')


        with TemporaryDirectory() as source_frames_dir:
            if os.path.isdir(input_file):  # folder input
                inputWidth = cap.frame_width
                inputHeight = cap.frame_height
                inputChannels = cap.frame_channels
            else:
                inputWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                inputHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                inputChannels = 1 if int(cap.get(cv2.CAP_PROP_MONOCHROME)) \
                    else 3
            logger.info(
                'Input video {} has W={} x H={} frames each with {} channels'
                .format(input_file, inputWidth, inputHeight, inputChannels))

            if (output_width is None) and (output_height is None):
                output_width = inputWidth
                output_height = inputHeight
                logger.warning(
                    'output size ({}x{}) was set automatically to '
                    'input video size\n    Are you sure you want this? '
                    'It might be slow.\n Consider using\n '
                    '    --output_width=346 --output_height=260\n '
                    'to match Davis346.'
                    .format(output_width, output_height))

                # set emulator output width and height for the last time
                emulator.output_width = output_width
                emulator.output_height = output_height

            logger.info(
                f'*** Stage 1/3: '
                f'Resizing {srcNumFramesToBeProccessed} input frames '
                f'to output size '
                f'(with possible RGB to luma conversion)')
            for inputFrameIndex in tqdm(
                    range(srcNumFramesToBeProccessed),
                    desc='rgb2luma', unit='fr'):
                # read frame
                ret, inputVideoFrame = cap.read()
                num_frames+=1
                if ret==False:
                    logger.warning(f'could not read frame {inputFrameIndex} from {cap}')
                    continue
                if inputVideoFrame is None or np.shape(inputVideoFrame) == ():
                    logger.warning(f'empty video frame number {inputFrameIndex} in {cap}')
                    continue
                if not ret or inputFrameIndex + start_frame > stop_frame:
                    break

                if args.crop is not None:
                    # crop the frame, indices are y,x, UL is 0,0
                    if c_l+(c_r if c_r is not None else 0)>=inputWidth:
                        logger.error(f'left {c_l}+ right crop {c_r} is larger than image width {inputWidth}')
                        v2e_quit(1)
                    if c_t+(c_b if c_b is not None else 0)>=inputHeight:
                        logger.error(f'top {c_t}+ bottom crop {c_b} is larger than image height {inputHeight}')
                        v2e_quit(1)

                    inputVideoFrame= inputVideoFrame[c_t:c_b, c_l:c_r] # https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python

                if output_height and output_width and \
                        (inputHeight != output_height or
                         inputWidth != output_width):
                    dim = (output_width, output_height)
                    (fx, fy) = (float(output_width) / inputWidth,
                                float(output_height) / inputHeight)
                    inputVideoFrame = cv2.resize(
                        src=inputVideoFrame, dsize=dim, fx=fx, fy=fy,
                        interpolation=cv2.INTER_AREA)
                if inputChannels == 3:  # color
                    if inputFrameIndex == 0:  # print info once
                        logger.info(
                            '\nConverting input frames from RGB color to luma')
                    # TODO would break resize if input is gray frames
                    # convert RGB frame into luminance.
                    inputVideoFrame = cv2.cvtColor(
                        inputVideoFrame, cv2.COLOR_BGR2GRAY)  # much faster

                    # TODO add vid_orig output if not using slomo


                # save frame into numpy records
                save_path = os.path.join(
                    source_frames_dir, str(inputFrameIndex).zfill(8) + ".npy")
                np.save(save_path, inputVideoFrame)
                # print("Writing source frame {}".format(save_path), end="\r")
            cap.release()

            with TemporaryDirectory() as interpFramesFolder:
                interpTimes = None
                # make input to slomo
                if slomo is not None and (auto_timestamp_resolution or slowdown_factor != NO_SLOWDOWN):
                    # interpolated frames are stored to tmpfolder as
                    # 1.png, 2.png, etc

                    logger.info(
                        f'*** Stage 2/3: SloMo upsampling from '
                        f'{source_frames_dir}')
                    interpTimes, avgUpsamplingFactor = slomo.interpolate(
                        source_frames_dir, interpFramesFolder,
                        (output_width, output_height))
                    avgTs = srcFrameIntervalS / avgUpsamplingFactor
                    logger.info(
                        'SloMo average upsampling factor={:5.2f}; '
                        'average DVS timestamp resolution={}s'
                        .format(avgUpsamplingFactor, eng(avgTs)))
                    # check for undersampling wrt the
                    # photoreceptor lowpass filtering

                    if cutoff_hz > 0:
                        logger.info('Using auto_timestamp_resolution. '
                                       'checking if cutoff hz is ok given '
                                       'sample rate {}'.format(1/avgTs))
                        check_lowpass(cutoff_hz, 1/avgTs, logger)

                    # read back to memory
                    interpFramesFilenames = all_images(interpFramesFolder)
                    # number of frames
                    n = len(interpFramesFilenames)
                else:
                    logger.info(
                        f'*** Stage 2/3:turning npy frame files to png '
                        f'from {source_frames_dir}')
                    interpFramesFilenames = []
                    n = 0
                    src_files = sorted(
                        glob.glob("{}".format(source_frames_dir) + "/*.npy"))
                    for frame_idx, src_file_path in tqdm(
                            enumerate(src_files), desc='npy2png', unit='fr'):
                        src_frame = np.load(src_file_path)
                        tgt_file_path = os.path.join(
                            interpFramesFolder, str(frame_idx) + ".png")
                        interpFramesFilenames.append(tgt_file_path)
                        n += 1
                        cv2.imwrite(tgt_file_path, src_frame)
                    interpTimes = np.array(range(n))

                # compute times of output integrated frames
                nFrames = len(interpFramesFilenames)
                # interpTimes is in units of 1 per input frame,
                # normalize it to src video time range
                f = srcVideoRealProcessedDuration/(
                    np.max(interpTimes)-np.min(interpTimes))
                # compute actual times from video times
                interpTimes = f*interpTimes
                # debug
                if slomo_stats_plot:
                    from matplotlib import pyplot as plt  # TODO debug
                    dt = np.diff(interpTimes)
                    fig = plt.figure()
                    ax1 = fig.add_subplot(111)
                    ax1.set_title(
                        'Slo-Mo frame interval stats (close to continue)')
                    ax1.plot(interpTimes)
                    ax1.plot(interpTimes, 'x')
                    ax1.set_xlabel('frame')
                    ax1.set_ylabel('frame time (s)')
                    ax2 = ax1.twinx()
                    ax2.plot(dt*1e3)
                    ax2.set_ylabel('frame interval (ms)')
                    logger.info('close plot to continue')
                    fig.show()

                # array to batch events for rendering to DVS frames
                events = np.zeros((0, 4), dtype=np.float32)

                logger.info(
                    f'*** Stage 3/3: emulating DVS events from '
                    f'{nFrames} frames')

                # parepare extra steps for data storage
                # right before event emulation
                if args.ddd_output:
                    emulator.prepare_storage(nFrames, interpTimes)

                # generate events from frames and accumulate events to DVS frames for output DVS video
                with tqdm(total=nFrames, desc='dvs', unit='fr') as pbar:
                    with torch.no_grad():
                        for i in range(nFrames):
                            fr = read_image(interpFramesFilenames[i])
                            newEvents = emulator.generate_events(
                                fr, interpTimes[i])

                            pbar.update(1)
                            if newEvents is not None and \
                                    newEvents.shape[0] > 0 \
                                    and not args.skip_video_output:
                                events = np.append(events, newEvents, axis=0)
                                events = np.array(events)
                                if i % batch_size == 0:
                                    eventRenderer.render_events_to_frames(
                                        events, height=output_height,
                                        width=output_width)
                                    events = np.zeros((0, 4), dtype=np.float32)
                    # process leftover events
                    if len(events) > 0 and not args.skip_video_output:
                        eventRenderer.render_events_to_frames(
                            events, height=output_height, width=output_width)

    # Clean up
    eventRenderer.cleanup()
    emulator.cleanup()
    if slomo is not None:
        slomo.cleanup()
    if synthetic_input_instance is not None:
        synthetic_input_instance.cleanup()

    if num_frames == 0:
        logger.error('no frames read from file')

    totalTime = (time.time()-time_run_started)
    framePerS = num_frames / totalTime
    sPerFrame = totalTime / num_frames if num_frames>0 else None
    throughputStr = (str(eng(framePerS)) + 'fr/s') \
        if framePerS > 1 else (str(eng(sPerFrame)) + 's/fr')
    timestr ='done processing {} frames in {}s ({})\n **************** see output folder {}'.format(num_frames,
            eng(totalTime),
            throughputStr,
            output_folder)
    logger.info('generated total {} events ({} on, {} off)'
                .format(eng(emulator.num_events_total),
                        eng(emulator.num_events_on),
                        eng(emulator.num_events_off)))
    total_time = emulator.t_previous
    rate_total= emulator.num_events_total / total_time
    rate_on_total= emulator.num_events_on / total_time
    rate_off_total= emulator.num_events_off / total_time
    rate_per_pixel=rate_total/num_pixels
    rate_on_per_pixel=rate_on_total/num_pixels
    rate_off_per_pixel=rate_off_total/num_pixels
    logger.info(
        f'Avg event rate for N={num_pixels} px and total time ={total_time:.3f} s'
        f'\n\tTotal: {eng(rate_total)}Hz ({eng(rate_on_total)}Hz on, {eng(rate_off_total)}Hz off)'
        f'\n\tPer pixel:  {eng(rate_per_pixel)}Hz ({eng(rate_on_per_pixel)}Hz on, {eng(rate_off_per_pixel)}Hz off)')
    if totalTime>60:
        try:
            from plyer import notification
            logger.info(f'generating desktop notification')
            notification.notify(title='v2e done', message=timestr,timeout=3)
        except Exception as e:
            logger.info(f'could not show notification: {e}')

    # try to show desktop
    # suppress folder opening if it's not necessary
    if not output_folder is None:
        try:
            logger.info(f'showing {output_folder} in desktop')
            desktop.open(os.path.abspath(output_folder))
        except Exception as e:
            logger.warning(
                '{}: could not open {} in desktop'.format(e, output_folder))
    logger.info(timestr)
    sys.exit(0)


if __name__ == "__main__":
    main()
    v2e_quit()
