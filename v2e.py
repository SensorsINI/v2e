#!/usr/bin/env python
"""
Python code for extracting frames from video file and synthesizing fake DVS
events from this video after SuperSloMo has generated interpolated
frames from the original video frames.

@author: Tobi Delbruck Zhe He
@contact: tobi@ini.uzh.ch, zhehe@student.ethz.ch
@latest update: Apr 2020
"""
# todo  add batch mode for slomo to speed up
# todo refractory period for pixel

import argparse
from pathlib import Path

import argcomplete
import cv2
import numpy as np
import os
from tempfile import TemporaryDirectory
from engineering_notation import EngNumber  # only from pip
from tqdm import tqdm


import v2e.desktop as desktop
from v2e.v2e_utils import all_images, read_image, OUTPUT_VIDEO_FPS, \
    check_lowpass, v2e_quit
from v2e.v2e_args import v2e_args, write_args_info, v2e_check_dvs_exposure_args
from v2e.renderer import EventRenderer, ExposureMode
from v2e.slomo import SuperSloMo
from v2e.emulator import EventEmulator
from v2e.v2e_utils import inputVideoFileDialog


import logging

logging.basicConfig()
root = logging.getLogger()
root.setLevel(logging.INFO)
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
logging.addLevelName(
    logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(
        logging.WARNING))
logging.addLevelName(
    logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(
        logging.ERROR))
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description='v2e: generate simulated DVS events from video.',
    epilog='Run with no --input to open file dialog', allow_abbrev=True,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser = v2e_args(parser)
parser.add_argument(
    "--rotate180", type=bool, default=False,
    help="rotate all output 180 deg.")

# https://kislyuk.github.io/argcomplete/#global-completion
# Shellcode (only necessary if global completion is not activated -
# see Global completion below), to be put in e.g. .bashrc:
# eval "$(register-python-argcomplete v2e.py)"
argcomplete.autocomplete(parser)
args = parser.parse_args()


if __name__ == "__main__":
    overwrite = args.overwrite
    output_folder = args.output_folder
    f = not overwrite and os.path.exists(output_folder) \
        and os.listdir(output_folder)
    if f:
        logger.error(
            'output folder {} already exists\n it holds files {}\n '
            '- use --overwrite'.format(os.path.abspath(output_folder), f))
        v2e_quit()

    if not os.path.exists(output_folder):
        logger.info('making output folder {}'.format(output_folder))
        os.mkdir(output_folder)

    if (args.output_width is not None) ^ (args.output_width is not None):
        logger.error(
            'provide both or neither of output_width and output_height')
        v2e_quit()
    input_file = args.input
    if not input_file:
        input_file = inputVideoFileDialog()
        if not input_file:
            logger.info('no file selected, quitting')
            v2e_quit()

    output_width: int = args.output_width
    output_height: int = args.output_height
    if (output_width is None) ^ (output_height is None):
        logger.error('set neither or both of output_width and output_height')
        v2e_quit()

    # input file checking
    if not input_file or not Path(input_file).exists():
        logger.error('input file {} does not exist'.format(input_file))
        v2e_quit()

    start_time = args.start_time
    stop_time = args.stop_time
    slowdown_factor = args.slowdown_factor
    pos_thres = args.pos_thres
    neg_thres = args.neg_thres
    sigma_thres = args.sigma_thres
    cutoff_hz = args.cutoff_hz
    leak_rate_hz = args.leak_rate_hz
    if leak_rate_hz > 0 and sigma_thres == 0:
        logger.warning(
            'leak_rate_hz>0 but sigma_thres==0, '
            'so all leak events will be synchronous')
    shot_noise_rate_hz = args.shot_noise_rate_hz
    dvs_vid = args.dvs_vid
    dvs_vid_full_scale = args.dvs_vid_full_scale
    dvs_h5 = args.dvs_h5
    # dvs_np = args.dvs_np
    dvs_aedat2 = args.dvs_aedat2
    dvs_text = args.dvs_text
    vid_orig = args.vid_orig
    vid_slomo = args.vid_slomo
    preview = not args.no_preview
    rotate180 = args.rotate180
    segment_size = args.segment_size
    batch_size = args.batch_size
    exposure_mode,exposure_val,area_dimension=v2e_check_dvs_exposure_args(args)

    infofile = write_args_info(args, output_folder)

    fh = logging.FileHandler(infofile)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    import time
    time_run_started = time.time()

    logger.info("opening video input " + input_file)

    cap = cv2.VideoCapture(input_file)
    srcFps = cap.get(cv2.CAP_PROP_FPS)
    if srcFps == 0:
        logger.error('source {} fps is 0'.format(input_file))
        v2e_quit()
    srcFrameIntervalS = 1. / srcFps
    slomoTimestampResolutionS = srcFrameIntervalS / slowdown_factor
    # https://stackoverflow.com/questions/25359288/how-to-know-total-number-of-frame-in-a-file-with-cv2-in-python
    srcNumFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if srcNumFrames < 2:
        logger.warning(
            'num frames is less than 2, probably cannot be determined '
            'from cv2.CAP_PROP_FRAME_COUNT')

    check_lowpass(cutoff_hz, srcFps*slowdown_factor, logger)

    # only works with batch_size=1 now
    slomo = SuperSloMo(
        model=args.slomo_model, slowdown_factor=args.slowdown_factor,
        video_path=output_folder, vid_orig=vid_orig, vid_slomo=vid_slomo,
        preview=preview, batch_size=batch_size)

    srcTotalDuration = (srcNumFrames - 1) * srcFrameIntervalS
    start_frame = int(srcNumFrames * (start_time / srcTotalDuration)) \
        if start_time else 0
    stop_frame = int(srcNumFrames * (stop_time / srcTotalDuration)) \
        if stop_time else srcNumFrames
    srcNumFramesToBeProccessed = stop_frame-start_frame+1
    srcDurationToBeProcessed = srcNumFramesToBeProccessed/srcFps

    if exposure_mode==ExposureMode.DURATION:
        dvsFps = 1./exposure_val
        dvsNumFrames = np.math.floor(dvsFps * srcDurationToBeProcessed)
        dvsDuration = dvsNumFrames / dvsFps
        dvsPlaybackDuration = dvsNumFrames / OUTPUT_VIDEO_FPS
        logger.info('\n\n{} has {} frames with duration {}s, '
                    '\nsource video is {}fps (frame interval {}s),'
                    '\n slomo will have {}fps,'
                    '\n events will have timestamp resolution {}s,'
                    '\n v2e DVS video will have {}fps (accumulation time {}s), '
                    '\n DVS video will have {} frames with duration {}s '
                    'and playback duration {}s\n'
                    .format(input_file, srcNumFrames, EngNumber(srcTotalDuration),
                            EngNumber(srcFps), EngNumber(srcFrameIntervalS),
                            EngNumber(srcFps * slowdown_factor),
                            EngNumber(slomoTimestampResolutionS),
                            EngNumber(dvsFps), EngNumber(1 / dvsFps),
                            dvsNumFrames, EngNumber(dvsDuration),
                            EngNumber(dvsPlaybackDuration)))
    else:
        logger.info('\n\n{} has {} frames with duration {}s, '
                    '\nsource video is {}fps (frame interval {}s),'
                    '\n slomo will have {}fps,'
                    '\n events will have timestamp resolution {}s,'
                    '\n v2e DVS video will have constant count frames with {} events), '
                    .format(input_file, srcNumFrames, EngNumber(srcTotalDuration),
                            EngNumber(srcFps), EngNumber(srcFrameIntervalS),
                            EngNumber(srcFps * slowdown_factor),
                            EngNumber(slomoTimestampResolutionS),
                            exposure_val))

    emulator = EventEmulator(
        pos_thres=pos_thres, neg_thres=neg_thres,
        sigma_thres=sigma_thres, cutoff_hz=cutoff_hz,
        leak_rate_hz=leak_rate_hz, shot_noise_rate_hz=shot_noise_rate_hz,
        output_folder=output_folder, dvs_h5=dvs_h5, dvs_aedat2=dvs_aedat2,
        dvs_text=dvs_text)

    if args.dvs_params:
        emulator.set_dvs_params(args.dvs_params)

    eventRenderer = EventRenderer(
        output_path=output_folder,
        dvs_vid=dvs_vid, preview=preview, full_scale_count=dvs_vid_full_scale,
        exposure_mode=exposure_mode, exposure_value=exposure_val,area_dimension=area_dimension)

    ts0 = 0
    ts1 = srcFrameIntervalS  # timestamps of src frames
    num_frames = 0
    inputHeight = None
    inputWidth = None
    inputChannels = None
    if start_frame > 0:
        logger.info('skipping to frame {}'.format(start_frame))
        for i in range(start_frame):
            ret, _ = cap.read()
            if not ret:
                raise ValueError(
                    'something wrong, got to end of file before '
                    'reaching start_frame')

    logger.info(
        'processing frames {} to {} from video input'.format(
            start_frame, stop_frame))
    batchFrames = []
    # step over input by batch_size steps
    for frameNumber in tqdm(
            range(start_frame, stop_frame, segment_size),
            unit='fr', desc='v2e'):
        # each time add batch_size frames to previous frame
        # which we made first frame at end of interpolating and
        # generating events
        for i in range(segment_size):
            if cap.isOpened():
                ret, inputVideoFrame = cap.read()
            else:
                break
            if not ret:
                break
            num_frames += 1
            if len(batchFrames) == 0:  # first frame, just initialize sizes
                logger.info(
                    'input frames have shape {}'.format(inputVideoFrame.shape))
                inputHeight = inputVideoFrame.shape[0]
                inputWidth = inputVideoFrame.shape[1]
                inputChannels = inputVideoFrame.shape[2]
                if (output_width is None) and (output_height is None):
                    output_width = inputWidth
                    output_height = inputHeight
                    logger.warning(
                        'output size ({}x{}) was set automatically to '
                        'input video size\n    Are you sure you want this? '
                        'It might be slow.\n    Consider using '
                        '--output_width and --output_height'
                            .format(output_width, output_height))
            if output_height and output_width and \
                    (inputHeight != output_height or
                     inputWidth != output_width):
                dim = (output_width, output_height)
                (fx, fy) = (float(output_width)/inputWidth,
                            float(output_height)/inputHeight)
                inputVideoFrame = cv2.resize(
                    src=inputVideoFrame, dsize=dim, fx=fx, fy=fy,
                    interpolation=cv2.INTER_AREA)
            if inputChannels == 3:  # color
                if len(batchFrames) == 0:  # print info once
                    logger.info(
                        'converting input frames from RGB color to luma')
                # TODO would break resize if input is gray frames
                # convert RGB frame into luminance.
                inputVideoFrame = cv2.cvtColor(
                    inputVideoFrame, cv2.COLOR_BGR2GRAY)  # much faster
                # frame = (0.2126 * frame[:, :, 0] +
                #          0.7152 * frame[:, :, 1] +
                #          0.0722 * frame[:, :, 2])
            batchFrames.append(inputVideoFrame)
        if len(batchFrames) < 2:
            continue  # need at least 2 frames

        with TemporaryDirectory() as interpFramesFolder:
            # make input to slomo
            slomoInputFrames = np.asarray(batchFrames)

            # interpolated frames are stored to tmpfolder as 1.png, 2.png, etc
            slomo.interpolate(slomoInputFrames, interpFramesFolder)
            # read back to memory
            interpFramesFilenames = all_images(interpFramesFolder)
            # number of interpolated frames, will be 1 if slowdown_factor==1
            n = len(interpFramesFilenames)
            events = np.empty((0, 4), float)
            # Interpolating the 2 frames f0 to f1 results in
            # n frames f0 fi0 fi1 ... fin-2 f1
            # The endpoint frames are same as input.
            # If we pass these to emulator repeatedly,
            # then the f1 frame from past loop is the same as
            # the f0 frame in the next iteration.
            # For emulation, we should pass in to the emulator
            # only up to the last interpolated frame,
            # since the next iteration will pass in the f1
            # from previous iteration.

            # compute times of output integrated frames
            interpTimes = np.linspace(
                start=ts0, stop=ts1, num=n+1, endpoint=False)
            if n == 1:  # no slowdown
                fr = read_image(interpFramesFilenames[0])
                newEvents = emulator.generate_events(fr, ts0, ts1)
            else:
                for i in range(n):  # for each interpolated frame
                    fr = read_image(interpFramesFilenames[i])
                    newEvents = emulator.generate_events(
                        fr, interpTimes[i], interpTimes[i + 1])
                    if newEvents is not None and newEvents.shape[0] > 0:
                        events = np.append(events, newEvents, axis=0)
            events = np.array(events)  # remove first None element
            eventRenderer.render_events_to_frames(
                events, height=output_height, width=output_width)
            ts0 = ts1
            ts1 += srcFrameIntervalS

        # save last frame of input as 1st frame of new batch
        batchFrames = [inputVideoFrame]

    cap.release()
    if num_frames == 0:
        logger.error('no frames read from file')
        v2e_quit()
    totalTime = (time.time() - time_run_started)
    framePerS = num_frames/totalTime
    sPerFrame = 1/framePerS
    throughputStr = (str(EngNumber(framePerS))+'fr/s') \
        if framePerS > 1 else (str(EngNumber(sPerFrame))+'s/fr')
    logger.info('done processing {} frames in {}s ({})\n see output folder {}'
        .format(
        num_frames,
        EngNumber(totalTime),
        throughputStr,
        output_folder))
    logger.info('generated total {} events ({} on, {} off)'
                .format(EngNumber(emulator.num_events_total),
                        EngNumber(emulator.num_events_on),
                        EngNumber(emulator.num_events_off)))
    logger.info(
        'avg event rate {}Hz ({}Hz on, {}Hz off)'
            .format(EngNumber(emulator.num_events_total/srcDurationToBeProcessed),
                    EngNumber(emulator.num_events_on/srcDurationToBeProcessed),
                    EngNumber(emulator.num_events_off/srcDurationToBeProcessed)))
    try:
        desktop.open(os.path.abspath(output_folder))
    except Exception as e:
        logger.warning(
            '{}: could not open {} in desktop'.format(e, output_folder))
    eventRenderer.cleanup()
    slomo.cleanup()
    v2e_quit()
