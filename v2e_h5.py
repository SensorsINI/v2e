""" Render event frames from DVS record file in DDD17+ dataset.

@author: Zhe He
@contact: zhehe@student.ethz.ch
@latest update: 2019-08-05
"""

import numpy as np
import argparse
import os

from tempfile import TemporaryDirectory

from src.renderer import RenderFromImages, RenderFromEvents
from src.slomo import SuperSloMo
from src.reader import Reader
import warnings
import logging
#TODO rename to v2e_h5.py

# todo add live preview

logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.NOTSET)
logging.basicConfig()
logger = logging.getLogger('v2e')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

parser = argparse.ArgumentParser()
parser.add_argument("--pos_thres", type=float, default=0.2, help="log intensity change threshold to trigger a positive event, as positive quantity, e.g. 0.2")# todo change to theta_on
parser.add_argument("--neg_thres", type=float, default=-0.2, help="log intensity change threshold to trigger a negative event, as negative quantity, e.g. -.2")
parser.add_argument("--start_time", type=float, default=2.0, help="start point of video stream in seconds")
parser.add_argument("--stop_time", type=float, default=7.0, help="stop_time point of video stream in seconds")
parser.add_argument("--input", type=str, required=True, help="path of .h5 DVS or DAVIS input file from DDD dataset")# TODO fix README
parser.add_argument("--slomo_model",  type=str, required=True, help="path of Super-SloMo model checkpoint file")
parser.add_argument("--sf", type=int, required=True, help="slow motion factor, e.g. 10")
parser.add_argument("--avi_frame_rate", type=int, default=30.0, help="frame rate in Hz of output video AVI file for playback")
parser.add_argument("--output_path", type=str, required=True, help="path to store output avi video files, e.g. data")# todo fix to output_path

#TODO add usage with URL to webpage

args = parser.parse_args()

if args.neg_thres<0:
    warnings.warn('neg_thres should be positive quantity, changing it to positive value')
    args.neg_thres=-args.neg_thres

if args.avi_frame_rate<1:
    warnings.warn('avi frame rate is less than 1Hz')

if args.sf<1:
    warnings.warn('slowdown factor is less than 1')

for arg, value in args._get_kwargs():
    print("{}: {}".format(arg, value))

if __name__ == "__main__":

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    with open(os.path.join(args.output_path, "info.txt"), "w") as f:
        f.write("file name: {}\n".format(args.input.split("/")[-1]))
        f.write("start point: {:.2f}\n".format(args.start_time))
        f.write("stop_time point: {:.2f}\n".format(args.stop_time))
        f.write("slow motion factor: {}\n".format(args.sf))
        f.write("output frame rate: {}\n".format(args.avi_frame_rate))

    logging.info('reading frames to memory from h5 file '+str(args.input))

    m = Reader(args.input, start=args.start_time, stop=args.stop_time) # todo read incrementally, this reads entire file to RAM!
    frames, events = m.read()
    frame_ts = np.arange(
        frames["ts"][0],
        frames["ts"][-1],
        1 / args.avi_frame_rate
        # 1 / (args.sf * frames.shape[0] / (frames["ts"][-1] - frames["ts"][0]))
    )

    with TemporaryDirectory() as dirname:

        logging.info("Using temporary directory tmp_dir: " +str(dirname))
        logging.info('loading SuperSloMo model')

        s = SuperSloMo(
            args.slomo_model,
            args.sf,
            dirname,
            video_path=args.output_path,
            rotate=True
        )

        logging.info('interpolating frames with SuperSloMo')
        s.interpolate(frames["frame"])
        interpolated_ts = s.get_ts(frames["ts"])
        height, width = frames["frame"].shape[1:]

        logging.info('rendering real DVS events AVI and numpy data file')
        r_events = RenderFromEvents(
            frame_ts,
            events,
            os.path.join(args.output_path, "video_dvs.avi"),
            event_path=os.path.join(args.output_path, "events_dvs.npy"),
            rotate=True
        )

        _ = r_events.render(height, width)

        logging.info('rendering v2e synthetic DVS events from slow-motion video to AVI and numpy data file')
        r = RenderFromImages(
            dirname,
            frame_ts,
            interpolated_ts,
            args.pos_thres,
            args.neg_thres, # TODO rest of code treats OFF threshold as positive number
            os.path.join(args.output_path, "video_aps.avi"),
            event_path=os.path.join(args.output_path, "events_aps.npy"),
            rotate=True)

        _ = r.render(height, width)
