""" Render event frames from DVS recorded file in DDD20 dataset.

This script is more for validation that runtime use.
We use it to see if the emulator does a good job in capturing reality of DVS camera,
by comparing the real DVS events with v2e events from DAVIS APS frames.

@author: Zhe He, Yuhuang Hu, Tobi Delbruck
"""

import numpy as np
import argparse
import os

from tempfile import TemporaryDirectory

from ddd20_utils.ddd_h5_reader import DDD20SimpleReader
from src.renderer import VideoSequenceFiles2EventsRenderer, Events2VideoRenderer, EventRenderer
from src.slomo import SuperSloMo
import warnings
import logging
from tqdm import tqdm
from ddd20_utils import ddd_h5_reader

logging.basicConfig()
root = logging.getLogger()
root.setLevel(logging.DEBUG)
logger=logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--pos_thres", type=float, default=0.2, help="log intensity change threshold to trigger a positive event, as positive quantity, e.g. 0.2")
parser.add_argument("--neg_thres", type=float, default=-0.2, help="log intensity change threshold to trigger a negative event, as negative quantity, e.g. -.2")
parser.add_argument("--sigma_thres", type=float, default=0.03, help="1-std deviation threshold variation in log_e intensity change")
parser.add_argument("--start_time", type=float, default=0, help="start point of video stream in seconds; 0 for start of input")
parser.add_argument("--stop_time", type=float, default=0, help="stop_time point of video stream in seconds; 0 for end of input")
parser.add_argument("--input", type=str, required=True, help="path of .h5 DVS or DAVIS input file from DDD dataset")
parser.add_argument("--slomo_model",  type=str, required=True, help="path of Super-SloMo model checkpoint file")
parser.add_argument("--slowdown_factor", type=int, required=True, help="slow motion factor, e.g. 10")
parser.add_argument("--avi_frame_rate", type=int, default=30.0, help="frame rate in Hz of output video AVI file for playback")
parser.add_argument("--output_folder", type=str, required=True, help="path to store output avi video files, e.g. data")
parser.add_argument("--rotate", type=bool, default=False, required=False, help="rotate input 90 degrees")

args = parser.parse_args()

if args.neg_thres<0:
    warnings.warn('neg_thres should be positive quantity, changing it to positive value')
    args.neg_thres=-args.neg_thres

if args.pos_thres<=0:
    warnings.warn('pos_thres should be positive quantity, changing it to positive value')
    args.pos_thres=-args.pos_thres

if args.sigma_thres<0 or args.sigma_thres>args.neg_thres or args.sigma_thres>args.pos_thres:
    warnings.warn('sigma_thres is negative or larger than the DVS events thresholds; are you sure this is what you want?')

if args.avi_frame_rate<1:
    warnings.warn('avi frame rate is less than 1Hz')

if args.slowdown_factor<1:
    warnings.warn('slowdown factor is less than 1')

for arg, value in args._get_kwargs():
    print("{}: {}".format(arg, value))

if __name__ == "__main__":

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    args_stop_time = args.stop_time
    args_start_time = args.start_time
    with open(os.path.join(args.output_folder, "info.txt"), "w") as f:
        f.write("file name: {}\n".format(args.input.split("/")[-1]))
        f.write("start point: {:.2f}\n".format(args_start_time))
        f.write("stop_time point: {:.2f}\n".format(args_stop_time))
        f.write("slow motion factor: {}\n".format(args.slowdown_factor))
        f.write("output frame rate: {}\n".format(args.avi_frame_rate))

    dvs_avi_path=os.path.join(args.output_folder, "video_dvs.avi")
    dvs_event_path = os.path.join(args.output_folder, "events_dvs.npy")
    v2e_avi_path=os.path.join(args.output_folder, "video_v2e.avi")
    v2e_event_path = os.path.join(args.output_folder, "events_v2e.npy")



    # with open('a', 'w') as a, open('b', 'w') as b:
    #     do_something()
    logger.info('opening output files')
    renderFromImages = VideoSequenceFiles2EventsRenderer(pos_thres=args.pos_thres, neg_thres=args.neg_thres,
                                                         video_path=v2e_avi_path, event_path=v2e_event_path,
                                                         rotate=args.rotate)
    # renders frames from DVS events and writes them to video file
    renderFromEvents = EventRenderer(pos_thres=args.pos_thres, neg_thres=args.neg_thres, video_path=v2e_avi_path,
                                            event_path=v2e_event_path, rotate=args.rotate)


    slomo = SuperSloMo(model=args.slomo_model, slowdown_factor=args.slowdown_factor, video_path=args.output_folder,
                       rotate=True)

    # generates fake DVS events from real and interpolated APS frames, renders them to frames, writes to video file, and saves events to dataset file

    davisData= DDD20SimpleReader(args.input);
    startPacket=davisData.search(timeS=args_start_time);
    if not startPacket: raise ValueError('cannot find start time ' + str(args_start_time) + ' within file')
    logger.info('iterating over input file contents')
    numFrames=0
    numDvsEvents=0
    # numOnEvents=0
    # numOffEvents=0
    frames=None
    frame0=None
    frame1=None
    for i in tqdm(range(startPacket, davisData.numPackets),desc='v2e',unit='packet'):
        packet=davisData.readPacket(i)
        if not packet: continue # empty or could not parse this one
        if args_stop_time >0 and packet['timestamp']>davisData.startTimeS+ args_stop_time:
            logger.info('\n reached stop time {}'.format(args_stop_time))
            break
        if packet['etype']== ddd_h5_reader.DDD20SimpleReader.ETYPE_DVS:
            numDvsEvents+=packet['enumber']
            events=packet['data']
            logger.info('rendering real DVS events AVI and numpy data file')

            renderFromEvents.renderEventsToFrames(event_arr=events, height, width, frame_ts)


        elif packet['etype']== ddd_h5_reader.DDD20SimpleReader.ETYPE_APS:
            numFrames+=1
            tmpFrame=frame0
            frame0=frame1
            frame1=packet['data']
            if frame0 and frame1:
                frames={frame0,frame1}
                frame_ts = np.arange(
                    frames['ts'][0],
                    frames['ts'][-1],
                    1 / args.avi_frame_rate
                    # 1 / (args.slowdown_factor * frames.shape[0] / (frames["ts"][-1] - frames["ts"][0]))
                )
        # logger.info('{} frames, {} dvs events'.format(numFrames,numDvsEvents))
    # quit()

                logger.info('interpolating frame pair with SuperSloMo')
                with TemporaryDirectory() as tmpdir:
                    slomo.interpolate(images=frames["frame"], output_folder=tmpdir)
                    interpolated_ts = slomo.get_ts(frames["ts"])
                    height, width = frames["frame"].shape[1:]
                    logger.info('rendering v2e synthetic DVS events from slow-motion video to AVI and numpy data file')
                    renderFromImages.render(height=height,width=width,interpolated_ts=interpolated_ts,frame_ts=interpolated_ts,)
    renderFromEvents.close()
    renderFromImages.close()
    logger.info("done; see output folder " + str(args.output_folder))
    quit()