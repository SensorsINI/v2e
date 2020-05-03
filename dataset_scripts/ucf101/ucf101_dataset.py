"""Convert videos in UCF-101 dataset into event frames.
In each action class, one video is randomly selected.

@arthur: Zhe He
@contact: zhehe@student.ethz.ch
@latest update: 2019-Jul-7th
"""

import argparse
import cv2
import numpy as np
import os
import random
import shutil

from tempfile import TemporaryDirectory

from v2e.renderer import EventRenderer
from v2e.slomo import SuperSloMo

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # TODO add usage extra info with webpage link

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path of UCF-101 dataset"
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
        "--output",
        type=str,
        required=True,
        help="path to store output videos"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    classes = os.listdir(args.dataset)

    for i, action in enumerate(classes):

        candidates = os.listdir(os.path.join(args.dataset, action))
        video = os.path.join(args.dataset, action, random.choice(candidates))
        output_path = os.path.join(args.output, action)
        os.mkdir(output_path)
        print("{:d} - Action: {:s}\nVideo: {:s}".format(i, action, video))

        # load frames from the input video.
        frames = []

        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                # convert RGB frame into luminance frame.
                frame = (0.2126 * frame[:, :, 0] +
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
            num_frames,
            endpoint=False
        )

        with TemporaryDirectory() as dirname:

            print("tmp_dir: ", dirname)

            s = SuperSloMo(
                args.checkpoint,
                args.sf,
                dirname,
                video_path=output_path
            )
            s.interpolate(frames)
            interpolated_ts = s.get_interpolated_timestamps(input_ts)
            height, width = frames.shape[1:]

            for factor in [1, 10]:
                output_ts = np.linspace(
                    0,
                    (num_frames - 1) / fps,
                    factor * (num_frames - 1),
                    endpoint=False
                )

                r = EventRenderer(
                    dirname,
                    output_ts,
                    interpolated_ts,
                    args.pos_thres,
                    args.neg_thres,
                    os.path.join(
                        output_path,
                        "from_image_{:d}.avi".format(int(factor * fps))
                    )
                )

                _ = r.render(height, width)
            shutil.copy2(video, output_path)
