import sys
import numpy as np
import argparse

from tempfile import TemporaryDirectory


parser = argparse.ArgumentParser()
parser.add_argument(
    "--pos_thres",
    type=float,
    default=0.21,
    help="threshold to trigger a positive event"
)
parser.add_argument(
    "--neg_thres",
    type=float,
    default=0.17,
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

args = parser.parse_args()


if __name__ == "__main__":
    sys.path.append("../")
    sys.path.append("../src/")
    sys.path.append("../utils/")
    from src.renderer import RenderFromImages, RenderFromEvents
    from src.slomo import SuperSloMo
    from src.reader import Reader

    m = Reader(args.fname, start=args.start, stop=args.stop)
    frames, events = m.read()
    frame_ts = np.arange(
        frames["ts"][0],
        frames["ts"][-1],
        1 / args.frame_rate
    )

    with TemporaryDirectory() as dirname:

        print("tmp_dir: ", dirname)

        s = SuperSloMo(
            args.checkpoint,
            args.sf,
            dirname,
            video_path="../data/",
            rotate=True
        )

        s.interpolate(frames["frame"])
        # frame_ts = s.get_ts(frames["ts"])
        height, width = frames["frame"].shape[1:]

        r_events = RenderFromEvents(
            frame_ts,
            events,
            "../data/from_event.avi",
            rotate=True
        )

        _, _, _ = r_events.render(height, width)

        r = RenderFromImages(
            dirname,
            frame_ts,
            args.pos_thres,
            args.neg_thres,
            "../data/from_image.avi",
            rotate=True)

        _, _, _ = r.render(height, width)
