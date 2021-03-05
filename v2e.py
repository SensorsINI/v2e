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

import argcomplete
import cv2
import numpy as np
import os
from tempfile import TemporaryDirectory
from engineering_notation import EngNumber as eng  # only from pip
from tqdm import tqdm


import v2ecore.desktop as desktop
from v2ecore.v2e_utils import all_images, read_image, \
    check_lowpass, v2e_quit
from v2ecore.v2e_utils import set_output_dimension
from v2ecore.v2e_utils import set_output_folder
from v2ecore.v2e_utils import ImageFolderReader
from v2ecore.v2e_args import v2e_args, write_args_info
from v2ecore.v2e_args import v2e_check_dvs_exposure_args
from v2ecore.v2e_args import NO_SLOWDOWN
from v2ecore.renderer import EventRenderer, ExposureMode
from v2ecore.slomo import SuperSloMo
from v2ecore.emulator import EventEmulator
from v2ecore.v2e_utils import inputVideoFileDialog
import logging

logging.basicConfig()
root = logging.getLogger()
root.setLevel(logging.INFO)  # todo move to info for production
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
logging.addLevelName(
    logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(
        logging.WARNING))
logging.addLevelName(
    logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(
        logging.ERROR))
logger = logging.getLogger(__name__)

# may only apply to windows
try:
    #  from scripts.regsetup import description
    from gooey import Gooey  # pip install Gooey
except Exception as e:
    logger.warning(f"{e}: Gooey GUI builder not available, "
                   f"will use command line arguments.\n"
                   f"Install with 'pip install Gooey'. See README")


def get_args():
    parser = argparse.ArgumentParser(
        description='v2e: generate simulated DVS events from video.',
        epilog='Run with no --input to open file dialog', allow_abbrev=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = v2e_args(parser)

    #  parser.add_argument(
    #      "--rotate180", type=bool, default=False,
    #      help="rotate all output 180 deg.")
    # https://kislyuk.github.io/argcomplete/#global-completion
    # Shellcode (only necessary if global completion is not activated -
    # see Global completion below), to be put in e.g. .bashrc:
    # eval "$(register-python-argcomplete v2e.py)"
    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    return args


def main():
    try:
        ga = Gooey(get_args, program_name="v2e", default_size=(575, 600))
        logger.info(
            "Use --ignore-gooey to disable GUI and "
            "run with command line arguments")
        ga()
    except Exception as e:
        logger.warning(
            f'{e}: Gooey GUI not available, using command line arguments. \n'
            f'You can try to install with "pip install Gooey"')

    args = get_args()

    # Set output width and height based on the arguments
    output_width, output_height = set_output_dimension(
        args.output_width, args.output_height,
        args.dvs128, args.dvs240, args.dvs346,
        args.dvs640, args.dvs1024,
        logger)

    # setup synthetic input classes and method
    synthetic_input = args.synthetic_input
    synthetic_input_module = None
    synthetic_input_class = None
    synthetic_input_instance = None
    synthetic_input_next_frame_method = None
    if synthetic_input is not None:
        try:
            synthetic_input_module = importlib.import_module(synthetic_input)
            synthetic_input_class = getattr(
                synthetic_input_module, synthetic_input)
            synthetic_input_instance = synthetic_input_class(
                width=output_width, height=output_height,
                preview=not args.no_preview)
            synthetic_input_next_frame_method = getattr(
                synthetic_input_class, 'next_frame')

            logger.info(
                f'successfully instanced {synthetic_input_instance} with'
                'method {synthetic_input_next_frame_method}:'
                '{synthetic_input_module.__doc__}')

        except ModuleNotFoundError as e:
            logger.error(f'Could not import {synthetic_input}: {e}')
            v2e_quit(1)
        except AttributeError as e:
            logger.error(f'{synthetic_input} method incorrect?: {e}')
            v2e_quit(1)

    # set input file
    input_file = args.input
    if synthetic_input is None and not input_file:
        input_file = inputVideoFileDialog()
        if not input_file:
            logger.info('no file selected, quitting')
            v2e_quit()

    # Set output folder
    output_folder = set_output_folder(
        args.output_folder,
        input_file,
        args.unique_output_folder if not args.overwrite else False,
        args.overwrite,
        args.output_in_place
        if (not synthetic_input and args.output_folder is None) else False,
        logger)

    # input file checking
    #  if (not input_file or not os.path.isfile(input_file)
    #      or not os.path.isdir(input_file)) \
    #          and not synthetic_input:
    if not input_file and not synthetic_input:
        logger.error('input file {} does not exist'.format(input_file))
        v2e_quit(1)

    num_frames = 0
    srcNumFramesToBeProccessed = 0

    # define video parameters
    # the input start and stop time, may be round to actual
    # frame timestamp
    input_start_time = args.start_time
    input_stop_time = args.stop_time

    input_slowmotion_factor: float = args.input_slowmotion_factor
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

    # Cutoff and noise frequencies
    cutoff_hz = args.cutoff_hz
    leak_rate_hz = args.leak_rate_hz
    if leak_rate_hz > 0 and sigma_thres == 0:
        logger.warning(
            'leak_rate_hz>0 but sigma_thres==0, '
            'so all leak events will be synchronous')
    shot_noise_rate_hz = args.shot_noise_rate_hz

    # Visualization
    avi_frame_rate = args.avi_frame_rate
    dvs_vid = args.dvs_vid
    dvs_vid_full_scale = args.dvs_vid_full_scale
    vid_orig = args.vid_orig
    vid_slomo = args.vid_slomo
    preview = not args.no_preview

    # Event saving options
    dvs_h5 = args.dvs_h5
    dvs_aedat2 = args.dvs_aedat2
    dvs_text = args.dvs_text

    # Debug feature: if show slomo stats
    slomo_stats_plot = args.slomo_stats_plot
    #  rotate180 = args.rotate180  # never used, consider removing
    batch_size = args.batch_size

    # DVS exposure
    exposure_mode, exposure_val, area_dimension = \
        v2e_check_dvs_exposure_args(args)
    if exposure_mode == ExposureMode.DURATION:
        dvsFps = 1. / exposure_val

    # Writing the info file
    infofile = write_args_info(args, output_folder)

    fh = logging.FileHandler(infofile)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # TODO: fix this, this is not reasonable
    import time
    time_run_started = time.time()

    slomoTimestampResolutionS = None

    if synthetic_input is None:
        logger.info("opening video input file " + input_file)

        if os.path.isdir(input_file):
            if args.input_frame_rate is None:
                logger.error(
                    "When the video is presented as a folder, "
                    "The user has to set --input_frame_rate manually")
                v2e_quit(1)

            cap = ImageFolderReader(input_file, args.input_frame_rate)
            srcFps = cap.frame_rate
            srcNumFrames = cap.num_frames
        else:
            cap = cv2.VideoCapture(input_file)
            srcFps = cap.get(cv2.CAP_PROP_FPS)
            srcNumFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
                video_path=output_folder,
                vid_orig=vid_orig, vid_slomo=vid_slomo,
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
            dvsNumFrames = np.math.floor(
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
        else:
            logger.info(
                'v2e DVS video will have constant-count '
                'frames with {} events), '
                .format(exposure_val))

    emulator = EventEmulator(
        pos_thres=pos_thres, neg_thres=neg_thres,
        sigma_thres=sigma_thres, cutoff_hz=cutoff_hz,
        leak_rate_hz=leak_rate_hz, shot_noise_rate_hz=shot_noise_rate_hz,
        seed=args.dvs_emulator_seed,
        output_folder=output_folder, dvs_h5=dvs_h5, dvs_aedat2=dvs_aedat2,
        dvs_text=dvs_text, show_dvs_model_state=args.show_dvs_model_state, output_width=output_width, output_height=output_height)

    if args.dvs_params is not None:
        logger.warning(f'--dvs_param={args.dvs_params} option overrides your selected options for threshold, threshold-mismatch, leak and shot noise rates')
        emulator.set_dvs_params(args.dvs_params)

    eventRenderer = EventRenderer(
        output_path=output_folder,
        dvs_vid=dvs_vid, preview=preview, full_scale_count=dvs_vid_full_scale,
        exposure_mode=exposure_mode,
        exposure_value=exposure_val,
        area_dimension=area_dimension)

    if synthetic_input_next_frame_method is not None:
        # array to batch events for rendering to DVS frames
        events = np.zeros((0, 4), dtype=np.float32)
        (fr, time) = synthetic_input_instance.next_frame()
        i = 0
        with tqdm(total=synthetic_input_instance.total_frames(),
                  desc='dvs', unit='fr') as pbar:
            while fr is not None:
                newEvents = emulator.generate_events(fr, time)
                pbar.update(1)
                i += 1
                if newEvents is not None and newEvents.shape[0] > 0 \
                        and not args.skip_video_output:
                    events = np.append(events, newEvents, axis=0)
                    events = np.array(events)
                    if i % batch_size == 0:
                        eventRenderer.render_events_to_frames(
                            events, height=output_height, width=output_width)
                        events = np.zeros((0, 4), dtype=np.float32)
                (fr, time) = synthetic_input_instance.next_frame()
            # process leftover events
            if len(events) > 0 and not args.skip_video_output:
                eventRenderer.render_events_to_frames(
                    events, height=output_height, width=output_width)
    else:  # file input
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
                ret, _ = cap.read()
                if not ret:
                    raise ValueError(
                        'something wrong, got to end of file before '
                        'reaching start_frame')

        logger.info(
            'processing frames {} to {} from video input'.format(
                start_frame, stop_frame))

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

                if not ret or inputFrameIndex + start_frame > stop_frame:
                    break

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

                # save frame into numpy records
                save_path = os.path.join(
                    source_frames_dir, str(inputFrameIndex).zfill(8) + ".npy")
                np.save(save_path, inputVideoFrame)
                # print("Writing source frame {}".format(save_path), end="\r")
            cap.release()

            with TemporaryDirectory() as interpFramesFolder:
                interpTimes = None
                # make input to slomo
                if auto_timestamp_resolution or slowdown_factor != NO_SLOWDOWN:
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
                        logger.warning('Using auto_timestamp_resolution. '
                                       'checking if cutoff hz is ok given '
                                       'samplee rate {}'.format(1/avgTs))
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
                with tqdm(total=nFrames, desc='dvs', unit='fr') as pbar:
                    for i in range(nFrames):
                        fr = read_image(interpFramesFilenames[i])
                        newEvents = emulator.generate_events(
                            fr, interpTimes[i])

                        pbar.update(1)
                        if newEvents is not None and newEvents.shape[0] > 0 \
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

    if num_frames == 0:
        logger.error('no frames read from file')
        v2e_quit()

    totalTime = (time.time()-time_run_started)
    framePerS = num_frames / totalTime
    sPerFrame = 1 / framePerS
    throughputStr = (str(eng(framePerS)) + 'fr/s') \
        if framePerS > 1 else (str(eng(sPerFrame)) + 's/fr')
    logger.info(
        'done processing {} frames in {}s ({})\n see output folder {}'
        .format(num_frames,
                eng(totalTime),
                throughputStr,
                output_folder))
    logger.info('generated total {} events ({} on, {} off)'
                .format(eng(emulator.num_events_total),
                        eng(emulator.num_events_on),
                        eng(emulator.num_events_off)))
    logger.info(
        'avg event rate {}Hz ({}Hz on, {}Hz off)'
        .format(
            eng(emulator.num_events_total / srcDurationToBeProcessed),
            eng(emulator.num_events_on / srcDurationToBeProcessed),
            eng(emulator.num_events_off / srcDurationToBeProcessed)))

    # try to show desktop
    # suppress folder opening if it's not necessary
    if not args.skip_video_output and not args.no_preview:
        try:
            desktop.open(os.path.abspath(output_folder))
        except Exception as e:
            logger.warning(
                '{}: could not open {} in desktop'.format(e, output_folder))


if __name__ == "__main__":
    main()
    v2e_quit()
