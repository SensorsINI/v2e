#!/usr/bin/env python
"""
Python code for extracting frames from video file and synthesizing fake DVS
events from this video after SuperSloMo has generated interpolated frames from the original video frames.

@author: Tobi Delbruck Zhe He
@contact: tobi@ini.uzh.ch, zhehe@student.ethz.ch
@latest update: Apr 2020
"""
# todo  add batch mode for slomo to speed up
# todo refractory period for pixel

import argparse
import sys
from pathlib import Path

import argcomplete
import cv2
import numpy as np
import os
from tempfile import TemporaryDirectory
from engineering_notation import EngNumber  # only from pip
from tqdm import tqdm
import src.desktop as desktop

from src.v2e_utils import all_images, read_image, OUTPUT_VIDEO_FPS, v2e_args
from src.renderer import EventRenderer
from src.slomo import SuperSloMo
from src.emulator import EventEmulator
import logging

from src.v2e_utils import inputVideoFileDialog

logging.basicConfig()
root = logging.getLogger()
root.setLevel(logging.INFO)
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
logging.addLevelName( logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName( logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
logger=logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='v2e: generate simulated DVS events from video.',
                                 epilog='Run with no --input to open file dialog', allow_abbrev=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser=v2e_args(parser)
parser.add_argument("--rotate180", type=bool, default=False,
                    help="rotate all output 180 deg.")
# https://kislyuk.github.io/argcomplete/#global-completion
# Shellcode (only necessary if global completion is not activated - see Global completion below), to be put in e.g. .bashrc:
# eval "$(register-python-argcomplete v2e.py)"
argcomplete.autocomplete(parser)
args = parser.parse_args()

if __name__ == "__main__":
    overwrite=args.overwrite
    output_folder=args.output_folder
    f=not overwrite and os.path.exists(output_folder) and os.listdir(output_folder)
    if f:
        logger.error('output folder {} already exists\n it holds files {}\n - use --overwrite'.format(os.path.abspath(output_folder),f))
        quit()

    if not os.path.exists(output_folder):
        logger.info('making output folder {}'.format(output_folder))
        os.mkdir(output_folder)

    if (args.output_width != None) ^ (args.output_width != None):
        logger.error('provide both or neither of output_width and output_height')
        quit()
    input_file = args.input
    if not input_file:
        input_file = inputVideoFileDialog()
        if not input_file:
            logger.info('no file selected, quitting')
            quit()

    output_width: int = args.output_width
    output_height: int = args.output_height
    if (output_width is None) ^ (output_height is None):
        logger.error('set neither or both of output_width and output_height')
        quit()

    arguments_list = 'arguments:\n'
    for arg, value in args._get_kwargs():
        arguments_list += "{}:\t{}\n".format(arg, value)
    logger.info(arguments_list)

    with open(os.path.join(args.output_folder, "info.txt"), "w") as f:
        f.write(arguments_list)

    start_time=args.start_time
    stop_time=args.stop_time
    slowdown_factor = args.slowdown_factor
    pos_thres = args.pos_thres
    neg_thres = args.neg_thres
    sigma_thres = args.sigma_thres
    cutoff_hz=args.cutoff_hz
    leak_rate_hz=args.leak_rate_hz
    shot_noise_rate_hz=args.shot_noise_rate_hz
    dvs_vid = args.dvs_vid
    dvs_vid_full_scale = args.dvs_vid_full_scale
    dvs_h5 = args.dvs_h5
    # dvs_np = args.dvs_np
    dvs_aedat2 = args.dvs_aedat2
    dvs_text = args.dvs_text
    vid_orig = args.vid_orig
    vid_slomo = args.vid_slomo
    preview=not args.no_preview
    rotate180=args.rotate180

    import time
    time_run_started = time.time()

    # input file checking
    if not input_file or not Path(input_file).exists():
        logger.error('input file {} does not exist'.format(input_file))
        quit()

    logger.info("opening video input " + input_file)

    cap = cv2.VideoCapture(input_file)
    srcFps = cap.get(cv2.CAP_PROP_FPS)
    if srcFps == 0:
        logger.error('source {} fps is 0'.format(input_file))
        quit()
    srcFrameIntervalS = 1. / srcFps
    slomoTimestampResolutionS = srcFrameIntervalS / slowdown_factor
    # https://stackoverflow.com/questions/25359288/how-to-know-total-number-of-frame-in-a-file-with-cv2-in-python
    srcNumFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if srcNumFrames < 2:
        logger.warning('num frames is less than 2, probably cannot be determined from cv2.CAP_PROP_FRAME_COUNT')

    slomo = SuperSloMo(model=args.slomo_model, slowdown_factor=args.slowdown_factor, video_path=output_folder, vid_orig=vid_orig, vid_slomo=vid_slomo, preview=preview,rotate=rotate180)

    srcTotalDuration= (srcNumFrames - 1) * srcFrameIntervalS
    start_frame=int(srcNumFrames * (start_time / srcTotalDuration)) if start_time else 0
    stop_frame=int(srcNumFrames * (stop_time / srcTotalDuration)) if stop_time else srcNumFrames
    srcNumFramesToBeProccessed=stop_frame-start_frame+1
    srcDurationToBeProcessed=srcNumFramesToBeProccessed/srcFps
    dvsFps=args.frame_rate if args.frame_rate else srcFps
    dvsNumFrames= np.math.floor(dvsFps * srcDurationToBeProcessed)
    dvsDuration= dvsNumFrames / dvsFps
    dvsPlaybackDuration= dvsNumFrames / OUTPUT_VIDEO_FPS
    logger.info('\n\n{} has {} frames with duration {}s, '
                 '\nsource video is {}fps (frame interval {}s),'
                 '\n slomo will have {}fps,'
                 '\n events will have timestamp resolution {}s,'
                 '\n v2e DVS video will have {}fps (accumulation time {}s), '
                 '\n DVS video will have {} frames with duration {}s and playback duration {}s\n'
                 .format(input_file, srcNumFrames, EngNumber(srcTotalDuration),
                         EngNumber(srcFps), EngNumber(srcFrameIntervalS),
                         EngNumber(srcFps * slowdown_factor),
                         EngNumber(slomoTimestampResolutionS),
                         EngNumber(dvsFps), EngNumber(1 / dvsFps),
                         dvsNumFrames, EngNumber(dvsDuration), EngNumber(dvsPlaybackDuration))
                 )

    emulator = EventEmulator(pos_thres=pos_thres, neg_thres=neg_thres, sigma_thres=sigma_thres, cutoff_hz=cutoff_hz,leak_rate_hz=leak_rate_hz, shot_noise_rate_hz=shot_noise_rate_hz, output_folder=output_folder, dvs_h5=dvs_h5, dvs_aedat2=dvs_aedat2, dvs_text=dvs_text)
    eventRenderer = EventRenderer(frame_rate_hz=dvsFps,output_path=output_folder, dvs_vid=dvs_vid, preview=preview, rotate180=rotate180, full_scale_count=dvs_vid_full_scale)

    frame0 = None
    frame1 = None  # rotating buffers for slomo
    ts0 = 0
    ts1 = srcFrameIntervalS  # timestamps of src frames
    num_frames = 0
    inputHeight=None
    inputWidth=None
    inputChannels=None
    if start_frame>0:
        logger.info('skipping to frame {}'.format(start_frame))
        for i in range(start_frame):
            ret, _ = cap.read()
            if not ret: raise ValueError('something wrong, got to end of file before reaching start_frame')
    logger.info('processing frames {} to {} from video input'.format(start_frame,stop_frame))
    for frameNumber in tqdm(range(start_frame,stop_frame),unit='fr',desc='v2e'):
    # while (cap.isOpened()):
        if cap.isOpened():
            ret, frame = cap.read()
        else:
            break
        if not ret:
            break
        num_frames += 1
        if frame1 is None: # first frame, just initialize sizes
            logger.info('input frames have shape {}'.format(frame.shape))
            inputHeight = frame.shape[0]
            inputWidth = frame.shape[1]
            inputChannels= frame.shape[2]
            if (output_width is None) and (output_height is None):
                output_width = inputWidth
                output_height = inputHeight
                logger.warning('output size ({}x{}) was set automatically to input video size\n    Are you sure you want this? It might be slow.\n    Consider using --output_width and --output_height'.format(output_width,output_height))
        if output_height and output_width and (inputHeight != output_height or inputWidth != output_width):
            dim = (output_width, output_height)
            (fx, fy) = (float(output_width) / inputWidth, float(output_height) / inputHeight)
            frame = cv2.resize(src=frame, dsize=dim, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        if  inputChannels == 3: # color
            if frame1 is None: # print info once
                logger.info('converting input frames from RGB color to luma')
#todo would break resize if input is gray frames
            # convert RGB frame into luminance.
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # much faster
                # frame = (0.2126 * frame[:, :, 0] +
                #          0.7152 * frame[:, :, 1] +
                #          0.0722 * frame[:, :, 2])
        frame0 = frame1  # new first frame is old 2nd frame
        frame1 = frame.astype(np.uint8)  # new 2nd frame is latest input
        if frame0 is None:
            continue  # didn't get two frames yet
        with TemporaryDirectory() as interpFramesFolder:
            twoFrames = np.stack([frame0, frame1], axis=0)
            slomo.interpolate(twoFrames, interpFramesFolder)  # interpolated frames are stored to tmpfolder as 1.png, 2.png, etc
            interpFramesFilenames = all_images(interpFramesFolder)  # read back to memory
            n = len(interpFramesFilenames)  # number of interpolated frames, will be 1 if slowdown_factor==1
            events = np.empty((0, 4), float)
            # Interpolating the 2 frames f0 to f1 results in n frames f0 fi0 fi1 ... fin-2 f1
            # The endpoint frames are same as input.
            # If we pass these to emulator repeatedly,
            # then the f1 frame from past loop is the same as the f0 frame in the next iteration.
            # For emulation, we should pass in to the emulator only up to the last interpolated frame,
            # since the next iteration will pass in the f1 from previous iteration.

            # compute times of output integrated frames
            interpTimes = np.linspace(start=ts0, stop=ts1, num=n, endpoint=True)
            if n==1: # no slowdown
                fr = read_image(interpFramesFilenames[0])
                newEvents = emulator.accumulate_events(fr, ts0, ts1)
            else:
                for i in range(n -1):  # for each interpolated frame up to last; use n-1 because we get last interpolated frame as first frame next time
                    fr = read_image(interpFramesFilenames[i])
                    newEvents = emulator.accumulate_events(fr, interpTimes[i], interpTimes[i + 1])
                    if not newEvents is None: events = np.append(events, newEvents, axis=0)
            events = np.array(events)  # remove first None element
            eventRenderer.renderEventsToFrames(events, height=output_height, width=output_width)
            ts0 = ts1
            ts1 += srcFrameIntervalS


    cap.release()
    if num_frames == 0:
        logger.error('no frames read from file')
        quit()
    totalTime=(time.time() - time_run_started)
    framePerS=num_frames/totalTime
    sPerFrame=1/framePerS
    throughputStr=(str(EngNumber(framePerS))+'fr/s') if framePerS>1 else (str(EngNumber(sPerFrame))+'s/fr')
    logger.info('done processing {} frames in {}s ({})\n see output folder {}'.format(
                 num_frames,
                 EngNumber(totalTime),
                 throughputStr,
                 output_folder))
    logger.info('generated total {} events ({} on, {} off)'.format(EngNumber(emulator.num_events_total),EngNumber(emulator.num_events_on),EngNumber(emulator.num_events_off)))
    logger.info('avg event rate {}Hz ({}Hz on, {}Hz off)'.format(EngNumber(emulator.num_events_total/srcDurationToBeProcessed),EngNumber(emulator.num_events_on/srcDurationToBeProcessed),EngNumber(emulator.num_events_off/srcDurationToBeProcessed)))
    try:
        desktop.open(os.path.abspath(output_folder))
    except Exception as e:
        logger.warning('{}: could not open {} in desktop'.format(e, output_folder))
    sys.exit()

