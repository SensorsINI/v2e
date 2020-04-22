""" Render event frames from DVS recorded file in DDD20 dataset.

This script is more for validation that runtime use.
We use it to see if the emulator does a good job in capturing reality of DVS camera,
by comparing the real DVS events with v2e events from DAVIS APS frames.

@author: Zhe He, Yuhuang Hu, Tobi Delbruck
"""
from pathlib import Path

import numpy as np
import argparse
import os

from tempfile import TemporaryDirectory

from engineering_notation import EngNumber

import src.desktop
import argcomplete
import tkinter as tk
from tkinter import filedialog

from ddd20_utils.ddd_h5_reader import DDD20SimpleReader
from src.renderer import EventEmulator, EventRenderer
from src.slomo import SuperSloMo
import warnings
import logging
from tqdm import tqdm
from ddd20_utils import ddd_h5_reader
import logging

from v2e_utils import OUTPUT_VIDEO_FPS, all_images, read_image

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
parser.add_argument("--output_height", type=int, default=260,
                    help="height of output DVS data in pixels. If None, same as input video.")
parser.add_argument("--output_width", type=int, default=346,
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
    rotate180 = args.rotate180

    import time
    time_run_started = time.time()

    # input file checking
    if not input_file or not Path(input_file).exists():
        logger.error('input file {} does not exist'.format(input_file))
        quit()

    # with open('a', 'w') as a, open('b', 'w') as b:
    #     do_something()
    logger.info('opening output files')
    slomo = SuperSloMo(model=args.slomo_model, slowdown_factor=args.slowdown_factor, video_path=output_folder, vid_orig=vid_orig, vid_slomo=vid_slomo, preview=preview,rotate=rotate180)
    dvsVidReal=str(dvs_vid).replace('.avi','-real.avi')
    dvsVidFake=str(dvs_vid).replace('.avi','-real.avi')
    eventRendererReal = EventRenderer(pos_thres=args.pos_thres, neg_thres=args.neg_thres, sigma_thres=args.sigma_thres, output_path=output_folder, dvs_vid=dvsVidReal, preview=preview, rotate=rotate180)
    eventRendererFake = EventRenderer(pos_thres=args.pos_thres, neg_thres=args.neg_thres, sigma_thres=args.sigma_thres, output_path=output_folder, dvs_vid=dvsVidFake, preview=preview, rotate=rotate180)
    emulator = EventEmulator(None, pos_thres=pos_thres, neg_thres=neg_thres, sigma_thres=sigma_thres, cutoff_hz=cutoff_hz,leak_rate_hz=leak_rate_hz, output_folder=output_folder, dvs_h5=dvs_h5, dvs_aedat2=dvs_aedat2, dvs_text=dvs_text)



    # generates fake DVS events from real and interpolated APS frames, renders them to frames, writes to video file, and saves events to dataset file


    davisData= DDD20SimpleReader(input_file)

    startPacket=davisData.search(timeS=start_time) if start_time else davisData.firstPacketNumber
    if not startPacket: raise ValueError('cannot find relative start time ' + str(start_time) + 's within recording')
    stopPacket=davisData.search(timeS=stop_time) if stop_time else davisData.numPackets-1
    if not stopPacket: raise ValueError('cannot find relative stop time ' + str(start_time) + 's within recording')
    if not start_time: start_time=0
    if not stop_time: stop_time=davisData.durationS

    srcDurationToBeProcessed=stop_time-start_time
    dvsFps = args.frame_rate
    dvsNumFrames = int(np.math.floor(dvsFps * srcDurationToBeProcessed))
    dvsDuration = srcDurationToBeProcessed
    dvsPlaybackDuration = dvsNumFrames / OUTPUT_VIDEO_FPS
    dvsFrameTimestamps = np.linspace(davisData.startTimeS+start_time,
                                     davisData.startTimeS+srcDurationToBeProcessed, dvsNumFrames)

    logger.info('iterating over input file contents')
    num_frames=0
    numDvsEvents=0
    # numOnEvents=0
    # numOffEvents=0
    frames=None
    frame0=None
    frame1=None
    dvsFrameTime=0
    savedEvents=np.empty([0,4],dtype=float) # to hold extra events in partial last DVS frame
    for i in tqdm(range(startPacket, stopPacket),desc='v2e-ddd20',unit='packet'):
        packet=davisData.readPacket(i)
        if not packet: continue # empty or could not parse this one
        if stop_time >0 and packet['timestamp']>davisData.startTimeS+ stop_time:
            logger.info('\n reached stop time {}'.format(stop_time))
            break
        if packet['etype']== ddd_h5_reader.DDD20SimpleReader.ETYPE_DVS:
            numDvsEvents+=packet['enumber']
            events=np.array(packet['data'],dtype=float) # get just events [:,[ts,x,y,pol]]
            events[:, 0] = events[:, 0] * 1e-6 # us timestamps
            #prepend saved events if there are some
            events=np.vstack((savedEvents,events))
            # find dvs starting frame index
            ts0=events[0,0]
            ts1=events[-1,0]
            dt=ts1-ts0
            dvsFrameStartIdx=np.searchsorted(dvsFrameTimestamps, ts0, side='left')
            dvsFrameEndIdx=np.searchsorted(dvsFrameTimestamps,ts1,side='right')
            if dvsFrameEndIdx==len(dvsFrameTimestamps):
                dvsFrameEndIdx-=1
            endEventIdx = np.searchsorted(events[:, 0], dvsFrameTimestamps[dvsFrameEndIdx], side='right')
            savedEvents=events[-endEventIdx,:]
            theseEvents=events[:endEventIdx-1,:]
            # this packet spans some time, and we need to render into DVS frames with regular spacing.
            # But render throws all leftover events into the last DVS
            eventRendererReal.renderEventsToFrames(event_arr=theseEvents, height=output_height, width=output_width, frame_ts=dvsFrameTimestamps[dvsFrameStartIdx:dvsFrameEndIdx])

        elif packet['etype']== ddd_h5_reader.DDD20SimpleReader.ETYPE_APS:
            num_frames+=1
            tmpFrame=frame0
            frame0=frame1
            frame1=packet
            if frame0 is not None and frame1 is not None:
                with TemporaryDirectory() as interpFramesFolder:
                    im0=(frame0['data'] / 256).astype(np.uint8)
                    im1=(frame1['data'] / 256).astype(np.uint8)
                    # im1=frame1['data'].astype(np.uint8)
                    twoFrames=np.stack([im0,im1],axis=0)
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
                    interpTimes = np.linspace(start=frame0['timestamp'], stop=frame1['timestamp'], num=n, endpoint=True)
                    for i in range(n - 1):  # for each interpolated frame up to last; use n-1 because we get last interpolated frame as first frame next time
                        fr = read_image(interpFramesFilenames[i])
                        newEvents = emulator.compute_events(fr, interpTimes[i],
                                                            interpTimes[i + 1])  # todo something wrong here with count
                        if not newEvents is None: events = np.append(events, newEvents, axis=0)
                    ts = np.linspace(
                        start=frame0['timestamp'],
                        stop=frame1['timestamp'],
                        num=int((frame1['timestamp']-frame0['timestamp']) * dvsFps),
                        endpoint=True
                    )  # output_ts are the timestamps of the DVS video output frames. They come from destFps
                    events = np.array(events)  # remove first None element
                    eventRendererFake.renderEventsToFrames(events, height=output_height, width=output_width, frame_ts=ts)


    logger.info("done; see output folder " + str(args.output_folder))
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
