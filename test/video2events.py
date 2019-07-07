"""
Python code for extracting frames from video and converting them into DVS
events.

@author: Zhe He
@contact: zhehe@student.ethz.ch
@latest update: 2019-Jul-4th
"""


import argparse
import cv2
import numpy as np
import sys
import os

from tempfile import TemporaryDirectory


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="input video"
)

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
    "--sf",
    type=int,
    required=True,
    help="slow motion factor"
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="path of checkpoint"
)

parser.add_argument(
    "--frame_rate",
    type=int,
    help="frame rate of output video"
)

parser.add_argument(
    "--video_path",
    type=str,
    required=True,
    help="path to store output vidoes"
)

args = parser.parse_args()


if __name__ == "__main__":

    sys.path.append("../")
    sys.path.append("../src/")
    sys.path.append("../utils/")

    from src.renderer import RenderFromImages
    from src.slomo import SuperSloMo

    if not os.path.exists(args.video_path):
        os.mkdir(args.video_path)

    frames = []

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # convert RGB frame into luminance.
            frame = (0.02126 * frame[:, :, 0] +
                     0.7152 * frame[:, :, 1] +
                     0.0722 * frame[:, :, 2])
            frame = frame.astype(np.uint8)
            frames.append(frame)
        else:
            break
    cap.release()
    frames = np.stack(frames)
    num_frames = frames.shape[0]
    input_ts = output_ts = np.linspace(
        0,
        num_frames / fps,
        frames.shape[0],
        endpoint=False
    )

    output_ts = np.arange(
        0,
        num_frames / fps,
        num_frames * args.frame_rate if args.frame_rate else num_frames,
        endpoint=False
    )

    with TemporaryDirectory() as dirname:

        print("tmp_dir: ", dirname)

        s = SuperSloMo(
            args.checkpoint,
            args.sf,
            dirname,
            video_path=args.video_path
        )
        s.interpolate(frames)
        interpolated_ts = s.get_ts(input_ts)
        height, width = frames.shape[1:]

        r = RenderFromImages(
            dirname,
            output_ts,
            interpolated_ts,
            args.pos_thres,
            args.neg_thres,
            os.path.join(args.video_path, "from_image.avi"))

        _, _, _ = r.render(height, width)
