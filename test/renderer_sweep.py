import sys
import numpy as np
import argparse

from tempfile import TemporaryDirectory


parser = argparse.ArgumentParser()
parser.add_argument(
    "--threshold",
    type=float,
    default=0.01,
    help="threshold to trigger event"
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

    results = []

    with TemporaryDirectory() as dirname:

        print("tmp_dir: ", dirname)

        s = SuperSloMo(
            args.checkpoint,
            10,
            dirname
        )

        s.interpolate(frames["frame"])
        frame_ts = s.get_ts(frames["ts"])
        height, width = frames["frame"].shape[1:]

        r_events = RenderFromEvents(
            frame_ts,
            events,
            "../data/from_event.avi"
        )

        (frames_events,
         num_pos_events,
         num_neg_events) = r_events.render(height, width)

        pos_flag = False
        neg_flag = False

        for threshold in np.arange(0.01, 0.91, 0.01):

            r = RenderFromImages(
                dirname,
                frame_ts,
                threshold,
                "../data/from_image_{:.2f}.avi".format(threshold))

            (frames_images,
             num_pos_images,
             num_neg_images) = r.render(height, width)

            l1_error = np.mean(
                    np.abs(frames_images - frames_events)
                )
            abs_pos_diff = np.abs(num_pos_events - num_pos_images)
            abs_neg_diff = np.abs(num_neg_events - num_neg_images)

            if len(results) > 0:
                if abs_pos_diff >= results[-1][1]:
                    print("Optimal Pos Threshold Found: {}".format(results[-1][0]))
                    pos_flag = True
                if abs_neg_diff >= results[-1][2]:
                    print("Optimal Neg Threshold Found: {}".format(results[-1][0]))
                    neg_flag = True
            if pos_flag and neg_flag:
                break
            results.append(
                [threshold, abs_pos_diff, abs_neg_diff, l1_error]
            )
            print("Threshold: {:.2f}".format(threshold))
            print("Abs Pos Diff: {}".format(abs_pos_diff))
            print("Abs Neg Diff: {}".format(abs_neg_diff))
            print("MEAN L1 ERROR: {}".format(l1_error))
    results = np.array(results)
    np.save('../data/results.npy', results)
