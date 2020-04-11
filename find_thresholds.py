import numpy as np
import argparse

from tempfile import TemporaryDirectory

from src.renderer import RenderFromImages, RenderFromEvents
from src.slomo import SuperSloMo
from src.reader import Reader

# TODO rename to find_thresholds.py

parser = argparse.ArgumentParser()

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

args = parser.parse_args()


if __name__ == "__main__":

    m = Reader(args.fname, start=args.start, stop=args.stop)
    frames, events = m.read()

    results = []

    with TemporaryDirectory() as dirname:

        print("tmp_dir: ", dirname)

        s = SuperSloMo(
            args.checkpoint,
            args.sf,
            dirname
        )

        s.interpolate(frames["frame"])
        frame_ts = s.get_ts(frames["ts"])
        height, width = frames["frame"].shape[1:]

        r_events = RenderFromEvents(
            frame_ts,
            events,
            "data/from_event.avi"
        )

        events_dvs = r_events._get_events()

        num_pos_dvs = events_dvs[events_dvs[:, 3] == 1].shape[0]
        num_neg_dvs = events_dvs.shape[0] - num_pos_dvs

        pos_thres = -1.
        neg_thres = -1.

        for threshold in np.arange(0.01, 0.91, 0.01):

            r = RenderFromImages(
                dirname,
                frame_ts,
                frame_ts,
                threshold,
                threshold,
                "data/from_image_{:.2f}.avi".format(threshold))

            events_aps = r._get_events()

            num_pos_aps = events_aps[events_aps[:, 3] == 1].shape[0]
            num_neg_aps = events_aps.shape[0] - num_pos_aps

            abs_pos_diff = np.abs(num_pos_dvs - num_pos_aps)
            abs_neg_diff = np.abs(num_neg_dvs - num_neg_aps)

            if len(results) > 0:
                if abs_pos_diff >= results[-1][1] and pos_thres < 0:
                    pos_thres = results[-1][0]
                if abs_neg_diff >= results[-1][2] and neg_thres < 0:
                    neg_thres = results[-1][0]
            if pos_thres > 0 and neg_thres > 0:
                print("Optimal Pos Threshold Found: {}".format(pos_thres))
                print("Optimal Neg Threshold Found: {}".format(neg_thres))
                break
            results.append(
                [threshold, abs_pos_diff, abs_neg_diff]
            )
            print("Threshold: {:.2f}".format(threshold))
            print("Pos Thres: {:.2f}".format(pos_thres))
            print("Neg Thres: {:.2f}".format(neg_thres))
    results = np.array(results)
    np.save('data/results.npy', results)
