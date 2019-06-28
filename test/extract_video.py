import sys
import numpy as np
import argparse
import cv2
import tqdm

from tempfile import TemporaryDirectory


parser = argparse.ArgumentParser()

parser.add_argument(
    "--fname",
    type=str,
    required=True,
    help="path of .h5 file"
)

parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="path to store the outpur video"
)

parser.add_argument(
    action="store_true"
)

args = parser.parse_args()


if __name__ == "__main__":
    sys.path.append("../")
    sys.path.append("../src/")
    sys.path.append("../utils/")
    from src.reader import Reader
    from src.slomo import video_writer

    m = Reader(args.fname)
    frames, _ = m.read()
    writer = video_writer(args.output_path, frames.shape[2], frames.shape[1])

    for frame in tqdm(frames):
        if args.rotate:
            frame = np.rot90(frame, k=2)
            writer.write(
                cv2.cvtColor(
                    frame,
                    cv2.COLOR_GRAY2BGR
                )
            )
        if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
            break
