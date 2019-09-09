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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--pos_thres",
    type=float,
    default=0.25,
    help="threshold to trigger a positive event"
)
parser.add_argument(
    "--neg_thres",
    type=float,
    default=0.36,
    help="threshold to trigger a negative event"
)
parser.add_argument(
    "--start",
    type=float,
    default=0.0,
    help="start point of video stream"
)
parser.add_argument(
    "--stop",
    type=float,
    default=5.0,
    help="stop point of video stream"
)
parser.add_argument(
    "--fname",
    type=str,
    required=True,
    help="path of .h5 file"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="path of checkpoint"
)
parser.add_argument(
    "--sf",
    type=int,
    required=True,
    help="slow motion factor"
)
parser.add_argument(
    "--frame_rate",
    type=int,
    help="frame rate of output video"
)
parser.add_argument(
    "--path",
    type=str,
    required=True,
    help="path to store output files"
)

args = parser.parse_args()

for arg, value in args._get_kwargs():
    print("{}: {}".format(arg, value))


if __name__ == "__main__":

    if not os.path.exists(args.path):
        os.mkdir(args.path)

    with open(os.path.join(args.path, "info.txt"), "w") as f:
        f.write("file name: {}\n".format(args.fname.split("/")[-1]))
        f.write("start point: {:.2f}\n".format(args.start))
        f.write("stop point: {:.2f}\n".format(args.stop))
        f.write("slow motion factor: {}\n".format(args.sf))
        f.write("output frame rate: {}\n".format(args.frame_rate))

    m = Reader(args.fname, start=args.start, stop=args.stop)
    frames, events = m.read()
    frame_ts = np.arange(
        frames["ts"][0],
        frames["ts"][-1],
        1 / args.frame_rate
        # 1 / (args.sf * frames.shape[0] / (frames["ts"][-1] - frames["ts"][0]))
    )

    with TemporaryDirectory() as dirname:

        print("tmp_dir: ", dirname)

        s = SuperSloMo(
            args.checkpoint,
            args.sf,
            dirname,
            video_path=args.path,
            rotate=True
        )

        s.interpolate(frames["frame"])
        interpolated_ts = s.get_ts(frames["ts"])
        height, width = frames["frame"].shape[1:]

        r_events = RenderFromEvents(
            frame_ts,
            events,
            os.path.join(args.path, "video_dvs.avi"),
            event_path=os.path.join(args.path, "events_dvs.npy"),
            rotate=True
        )

        _ = r_events.render(height, width)

        r = RenderFromImages(
            dirname,
            frame_ts,
            interpolated_ts,
            args.pos_thres,
            args.neg_thres,
            os.path.join(args.path, "video_aps.avi"),
            event_path=os.path.join(args.path, "events_aps.npy"),
            rotate=True)

        _ = r.render(height, width)
