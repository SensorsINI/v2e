"""
Python code for extracting frames from video file and synthesizing fake DVS
events from this video after SuperSloMo has generated interpolated frames from the original video frames.

@author: Zhe He
@contact: zhehe@student.ethz.ch
@latest update: 2019-Jul-4th
"""

import argparse

import argcomplete
import cv2
import numpy as np
import os
import logging
from tempfile import TemporaryDirectory
from engineering_notation import EngNumber  # only from pip
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

from src.v2e_utils import all_images,read_image
from src.renderer import EventRenderer
from src.slomo import SuperSloMo
from src.emulator import EventEmulator

logging.basicConfig()
root = logging.getLogger()
root.setLevel(logging.DEBUG)
parser = argparse.ArgumentParser(description='v2e: generate simulated DVS events from video.',
                                 epilog='Run with no --input to open file dialog', allow_abbrev=True)
argcomplete.autocomplete(parser)
# https://kislyuk.github.io/argcomplete/#global-completion
# Shellcode (only necessary if global completion is not activated - see Global completion below), to be put in e.g. .bashrc:
# eval "$(register-python-argcomplete v2e.py)"
parser.add_argument("--input", type=str, help="input video file; leave empty for file chooser dialog")
parser.add_argument("--pos_thres", type=float, default=0.21,
                    help="threshold in log_e intensity change to trigger a positive event")
parser.add_argument("--neg_thres", type=float, default=0.17,
                    help="threshold in log_e intensity change to trigger a negative event")
parser.add_argument("--sigma_thres", type=float, default=0.03,
                    help="1-std deviation threshold variation in log_e intensity change")
parser.add_argument("--slowdown_factor", type=int, default=10,
                    help="slow motion factor; if the input video has frame rate fps, then the DVS events will have time resolution of 1/(fps*slowdown_factor)")
parser.add_argument("--output_height", type=int, default=None,
                    help="height of output DVS data in pixels. If None, same as input video.")
parser.add_argument("--output_width", type=int, default=None,
                    help="width of output DVS data in pixels. If None, same as input video.")
parser.add_argument("--slomo_model", type=str, required=True, help="path of slomo_model checkpoint")
parser.add_argument("--frame_rate", type=int, required=True,
                    help="frame rate of output video; the events will be accummulated as this sample rate; DVS frames will be accumulated for duration 1/frame_rate")
parser.add_argument("--output_folder", type=str, required=True, help="folder to store output video")
args = parser.parse_args()

slomo = 'arguments:\n'
for arg, value in args._get_kwargs():
    slomo += "{}:\t{}\n".format(arg, value)
logging.info(slomo)


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
    output_folder=args.output_folder
    if not os.path.exists(output_folder):
        logging.info('making output folder {}'.format(output_folder))
        os.mkdir(output_folder)

    if (args.output_width != None) ^ (args.output_width != None):
        logging.error('provide both or neither of output_width and output_height')
        quit()
    input_file = args.input
    if not input_file:
        input_file =inputFileDialog()
        if not input_file:
            logging.info('no file selected, quitting')
            quit()

    output_width = args.output_width
    output_height = args.output_height
    if (output_width is None) ^ (output_height is None):
        logging.error('set neither or both of output_width and output_height')
        quit()

    slowdown_factor = args.slowdown_factor
    pos_thres = args.pos_thres
    neg_thres = args.neg_thres
    sigma_thres = args.sigma_thres
    import time

    start_time = time.time()

    logging.info("opening video input " + input_file)
    cap = cv2.VideoCapture(input_file)
    srcFps = cap.get(cv2.CAP_PROP_FPS)
    if srcFps == 0:
        logging.error('source fps is 0')
        quit()
    srcFrameIntervalS = 1. / srcFps
    slomoTimestampResolutionS = srcFrameIntervalS / slowdown_factor
    srcNumFrames = int(cap.get(
        cv2.CAP_PROP_FRAME_COUNT))  # https://stackoverflow.com/questions/25359288/how-to-know-total-number-of-frame-in-a-file-with-cv2-in-python
    if srcNumFrames < 1:
        logging.warning('num frames in file cannot be determined from cv2.CAP_PROP_FRAME_COUNT')
    else:
        logging.info('{} has {} frames'.format(input_file, srcNumFrames))
    destFps=args.frame_rate if args.frame_rate else srcFps

    logging.info('\nsource video fps is {}Hz (frame interval {}s),\n slomo will have {} fps,\n events will have timestamp resolution {}s,\n v2e DVS video will have {} fps (accumulation time {})'.format(srcFps, EngNumber(1. / srcFps), srcFps * slowdown_factor, EngNumber(slomoTimestampResolutionS),destFps,EngNumber(1/destFps)))
    slomo = SuperSloMo(model=args.slomo_model, slowdown_factor=args.slowdown_factor, video_path=output_folder)
    eventRenderer = EventRenderer(pos_thres=args.pos_thres, neg_thres=args.neg_thres, sigma_thres=args.sigma_thres,
                                  video_path=output_folder)
    emulator = EventEmulator(None, pos_thres=pos_thres, neg_thres=neg_thres, sigma_thres=sigma_thres)

    frame0 = None
    frame1 = None  # rotating buffers for slomo
    ts0 = 0
    ts1 = srcFrameIntervalS  # timestamps of src frames
    num_frames = 0
    logging.info('reading frames from video input')
    for frameNumber in tqdm(range(srcNumFrames),unit='fr',desc='v2e'):
    # while (cap.isOpened()):
        if cap.isOpened():
            ret, frame = cap.read()
        else:
            break
        if not ret:
            break
        num_frames += 1
        if frame1 is None:
            logging.info('input frames have shape {}'.format(frame.shape))
            inputHeight = frame.shape[0]
            inputWidth = frame.shape[1]
            if (output_width is None) and (output_height is None):
                output_width = inputWidth
                output_height = inputHeight
        if frame.shape[2] == 3:
            if frame1 is None:
                logging.info('converting input frames from RGB color to luma')
            # convert RGB frame into luminance.
            frame = (0.2126 * frame[:, :, 0] +
                     0.7152 * frame[:, :, 1] +
                     0.0722 * frame[:, :, 2])
        if output_height and output_width:
            dim = (output_width, output_height)
            frame = cv2.resize(frame, dim)  # todo check that default linear interpolation is OK
        frame0 = frame1  # new first frame is old 2nd frame
        frame1 = frame.astype(np.uint8)  # new 2nd frame is latest input
        ts0 = ts1
        ts1 += srcFrameIntervalS  # todo check init here
        if frame0 is None: continue  # didn't get two frames yet
        # compute times of output integrated frames, using frame rate if supplied, otherwise to match input frame rate
        with TemporaryDirectory() as interpFramesFolder:
            twoFrames = np.stack([frame0, frame1], axis=0)
            slomo.interpolate(twoFrames, interpFramesFolder)  # interpolated frames are stored to tmpfolder as 1.png, 2.png, etc
            interpFramesFilenames = all_images(interpFramesFolder)  # read back to memory todo this is dumb
            n = len(interpFramesFilenames)  # number of interpolated frames
            interpTimes = np.linspace(start=ts0, stop=ts1, num=n,
                                      endpoint=True)  # slowdown_factor intermediate timestamps
            events = np.empty((0, 4), float)
            for i in range(n - 1):  # for each frame up to last
                fr = read_image(interpFramesFilenames[i])
                newEvents = emulator.compute_events(fr, interpTimes[i],
                                                    interpTimes[i + 1])  # todo something wrong here with count
                if not newEvents is None: events = np.append(events, newEvents, axis=0)
            output_ts = np.linspace(
                start=ts0,
                stop=ts1,
                num=int(srcFrameIntervalS * destFps) if destFps else int(srcFrameIntervalS * srcFps),
                endpoint=False
            )
            events = np.array(events)  # remove first None element
            eventFrames = eventRenderer.renderEventsToFrames(events, height=output_height, width=output_width,
                                                             frame_ts=output_ts)

    cap.release()
    if num_frames == 0:
        logging.error('no frames read from file')
        quit()
    totalTime=(time.time() - start_time)
    framePerS=num_frames/totalTime
    sPerFrame=1/framePerS
    throughputStr=(str(EngNumber(framePerS))+'fr/s') if framePerS>1 else (str(EngNumber(sPerFrame))+'s/fr')
    logging.info('done processing {} frames in {}s ({})\n see output folder {}'.format(
                 num_frames,
                 EngNumber(totalTime),
                 throughputStr,
                 output_folder))
    quit()

