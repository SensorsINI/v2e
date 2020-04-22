#!/usr/bin/env python
"""
Python code for extracting frames from video file and synthesizing fake DVS
events from this video after SuperSloMo has generated interpolated frames from the original video frames.

@author: Tobi Delbruck Zhe He
@contact: tobi@ini.uzh.ch, zhehe@student.ethz.ch
@latest update: Apr 2020
"""
# todo  h5ddd, solve bug with gap in events, add batch mode for slomo to speed up
# todo lowpass filter photoceptor
# todo refractory period for pixel
# todo leak events
# todo shot noise jitter

import argparse
from pathlib import Path

import argcomplete
import cv2
import numpy as np
import os
from tempfile import TemporaryDirectory
from engineering_notation import EngNumber  # only from pip
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
# import webbrowser
import src.desktop

from src.v2e_utils import all_images, read_image, OUTPUT_VIDEO_FPS
from src.renderer import EventRenderer
from src.slomo import SuperSloMo
from src.emulator import EventEmulator
import logging
logging.basicConfig()
root = logging.getLogger()
root.setLevel(logging.DEBUG)
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
logging.addLevelName( logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName( logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
logger=logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='v2e: generate simulated DVS events from video.',
                                 epilog='Run with no --input to open file dialog', allow_abbrev=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# https://kislyuk.github.io/argcomplete/#global-completion
# Shellcode (only necessary if global completion is not activated - see Global completion below), to be put in e.g. .bashrc:
# eval "$(register-python-argcomplete v2e.py)"
parser.add_argument("-i","--input", type=str, help="input video file; leave empty for file chooser dialog")
parser.add_argument("--start_time", type=float, default=None, help="start at this time in seconds in video")
parser.add_argument("--stop_time", type=float, default=None, help="stop at this time in seconds in video")
parser.add_argument("--pos_thres", type=float, default=0.21,
                    help="threshold in log_e intensity change to trigger a positive event")
parser.add_argument("--neg_thres", type=float, default=0.17,
                    help="threshold in log_e intensity change to trigger a negative event")
parser.add_argument("--sigma_thres", type=float, default=0.03,
                    help="1-std deviation threshold variation in log_e intensity change")
parser.add_argument("--cutoff_hz", type=float, default=300,
                    help="photoreceptor first order IIR lowpass cutoff-off 3dB frequency in Hz - see https://ieeexplore.ieee.org/document/4444573")
parser.add_argument("--leak_rate_hz", type=float, default=0.05,
                    help="leak event rate per pixel in Hz - see https://ieeexplore.ieee.org/abstract/document/7962235")
parser.add_argument("--slowdown_factor", type=int, default=10,
                    help="slow motion factor; if the input video has frame rate fps, then the DVS events will have time resolution of 1/(fps*slowdown_factor)")
parser.add_argument("--output_height", type=int, default=None,
                    help="height of output DVS data in pixels. If None, same as input video.")
parser.add_argument("--output_width", type=int, default=None,
                    help="width of output DVS data in pixels. If None, same as input video.")
parser.add_argument("--rotate180", action="store_true",
                    help="rotate all output 180 deg")
parser.add_argument("--slomo_model", type=str, default="input/SuperSloMo39.ckpt", help="path of slomo_model checkpoint")
parser.add_argument("-o", "--output_folder", type=str, required=True, help="folder to store outputs")
parser.add_argument("--frame_rate", type=int, default=300,
                    help="equivalent frame rate of --dvs_vid output video; the events will be accummulated as this sample rate; DVS frames will be accumulated for duration 1/frame_rate")
parser.add_argument("--dvs_vid", type=str, default="dvs-video.avi", help="output DVS events as AVI video at frame_rate")
parser.add_argument("--dvs_h5", type=str, default=None, help="output DVS events as hdf5 event database")
# parser.add_argument("--dvs_np", type=str, default=None, help="output DVS events as numpy event file")
parser.add_argument("--dvs_aedat2", type=str, default=None, help="output DVS events as AEDAT-2.0 event file for jAER")
parser.add_argument("--dvs_text", type=str, default=None, help="output DVS events as text file with one event per line [timestamp (float s), x, y, polarity (0,1)]")
parser.add_argument("--vid_orig", type=str, default="video_orig.avi", help="output src video at same rate as slomo video (with duplicated frames)")
parser.add_argument("--vid_slomo", type=str, default="video_slomo.avi", help="output slomo of src video slowed down by slowdown_factor")
parser.add_argument("-p","--preview", action="store_true", help="show preview in cv2 windows")
parser.add_argument("--overwrite", action="store_true", help="overwrites files in existing folder (checks existance of non-empty output_folder)")
argcomplete.autocomplete(parser)
args = parser.parse_args()


def inputFileDialog():
    root = tk.Tk()
    root.tk.call('tk', 'scaling', 4.0)  # doesn't help on hdpi screen
    root.withdraw()
    os.chdir('./input')
    filetypes=[("Video files", ".avi .mp4 .wmv"),('Any type','*')]
    filepath = filedialog.askopenfilename(filetypes=filetypes)
    os.chdir('..')
    return filepath

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
        input_file =inputFileDialog()
        if not input_file:
            logger.info('no file selected, quitting')
            quit()

    output_width = args.output_width
    output_height = args.output_height
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
    dvs_vid = args.dvs_vid
    dvs_h5 = args.dvs_h5
    # dvs_np = args.dvs_np
    dvs_aedat2 = args.dvs_aedat2
    dvs_text = args.dvs_text
    vid_orig = args.vid_orig
    vid_slomo = args.vid_slomo
    preview=args.preview
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
    eventRenderer = EventRenderer(pos_thres=args.pos_thres, neg_thres=args.neg_thres, sigma_thres=args.sigma_thres,
                                  output_path=output_folder,dvs_vid=dvs_vid,preview=preview,rotate=rotate180)
    emulator = EventEmulator(None, pos_thres=pos_thres, neg_thres=neg_thres, sigma_thres=sigma_thres, cutoff_hz=cutoff_hz,leak_rate_hz=leak_rate_hz, output_folder=output_folder, dvs_h5=dvs_h5, dvs_aedat2=dvs_aedat2, dvs_text=dvs_text)

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
                 '\n v2e DVS video will have {}fps (accumulation time {}), '
                 '\n DVS video will have {} frames with duration {}s and playback duration {}s\n'
                 .format(input_file, srcNumFrames, EngNumber(srcTotalDuration),
                         EngNumber(srcFps), EngNumber(srcFrameIntervalS),
                         EngNumber(srcFps * slowdown_factor),
                         EngNumber(slomoTimestampResolutionS),
                         EngNumber(dvsFps), EngNumber(1 / dvsFps),
                         dvsNumFrames, EngNumber(dvsDuration), EngNumber(dvsPlaybackDuration))
                 )
    frame0 = None
    frame1 = None  # rotating buffers for slomo
    ts0 = 0
    ts1 = srcFrameIntervalS  # timestamps of src frames
    num_frames = 0
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
            frame = cv2.resize(src=frame, dsize=dim, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)  # todo check that this scales to full output size. It doesn't; leaves border at top/bottom with noise on edge that creates events
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
            n = len(interpFramesFilenames)  # number of interpolated frames
            events = np.empty((0, 4), float)
            # Interpolating the 2 frames f0 to f1 results in n frames f0 fi0 fi1 ... fin-2 f1
            # The endpoint frames are same as input.
            # If we pass these to emulator repeatedly,
            # then the f1 frame from past loop is the same as the f0 frame in the next iteration.
            # For emulation, we should pass in to the emulator only up to the last interpolated frame,
            # since the next iteration will pass in the f1 from previous iteration.

            # compute times of output integrated frames
            interpTimes = np.linspace(start=ts0, stop=ts1, num=n, endpoint=True)
            for i in range(n -1):  # for each interpolated frame up to last; use n-1 because we get last interpolated frame as first frame next time
                fr = read_image(interpFramesFilenames[i])
                newEvents = emulator.compute_events(fr, interpTimes[i],
                                                    interpTimes[i + 1])  # todo something wrong here with count
                if not newEvents is None: events = np.append(events, newEvents, axis=0)
            dvsFrameTimestamps = np.linspace(
                start=ts0,
                stop=ts1,
                num=int(srcFrameIntervalS * dvsFps) if dvsFps else int(srcFrameIntervalS * srcFps),
                endpoint=True
            ) # output_ts are the timestamps of the DVS video output frames. They come from destFps
            events = np.array(events)  # remove first None element
            eventRenderer.renderEventsToFrames(events, height=output_height, width=output_width, frame_ts=dvsFrameTimestamps)
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
        src.desktop.open(output_folder)
    except:
        logger.warning('could not open {} in desktop'.format(output_folder))
    quit()

