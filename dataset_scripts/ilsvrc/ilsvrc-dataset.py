"""Script to generate ILSVRC video object detection dataset.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from __future__ import print_function, absolute_import

import argparse
import numpy as np
import os
import glob
from skimage.io import imread

from tempfile import TemporaryDirectory

from v2e.renderer import VideoSequenceFiles2EventsRenderer, EventRenderer
from v2e.slomo import SuperSloMo

# define a parser
parser = argparse.ArgumentParser()

# root folder for either train or val partition
parser.add_argument("--dir", "-d", type=str)
parser.add_argument("--out", "-o", type=str)

parser.add_argument(
    "--pos_thres",
    type=float,
    default=0.25,
    help="threshold to trigger a positive event"
)

parser.add_argument(
    "--neg_thres",
    type=float,
    default=0.35,
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

args = parser.parse_args()

# set fps, use 30
fps = 30.

assert os.path.isdir(args.dir)

if not os.path.isdir(args.out):
    os.makedirs(args.out)

# get the list of directory
collectd_paths = []
for root, dirs, files in os.walk(args.dir):
    if len(dirs) == 0:
        collectd_paths.append(root)

for vid_path in collectd_paths:
    # set up output folder
    base_name = os.path.basename(vid_path)
    vid_out_path = os.path.join(args.out, base_name)
    if not os.path.isdir(vid_out_path):
        os.makedirs(vid_out_path)

    # get all frames
    file_list = sorted(glob.glob("{}".format(vid_path)+"/*.*"))

    frames = []

    for img_file in file_list:
        # read image
        frame = imread(img_file)

        if frame.ndim == 3:
            # convert image
            frame = (0.2126 * frame[:, :, 0] +
                     0.7152 * frame[:, :, 1] +
                     0.0722 * frame[:, :, 2])

        frame = frame.astype(np.uint8)

        frames.append(frame)
        print("Loading file {}".format(img_file))

    frames = np.stack(frames)
    num_frames = frames.shape[0]

    # this is in seconds
    input_ts = output_ts = np.linspace(
        0,
        num_frames / fps,
        num_frames,
        endpoint=False
    )

    # export frame time stamps
    np.save(os.path.join(vid_out_path, "frame_ts.npy"), input_ts)

    with TemporaryDirectory() as dirname:
        print("tmp_dir: ", dirname)

        # do not export video
        s = SuperSloMo(
            args.checkpoint,
            args.sf,
            dirname,
            video_path=None
        )
        s.interpolate(frames)
        interpolated_ts = s.get_interpolated_timestamps(input_ts)
        height, width = frames.shape[1:]

        # render events
        output_ts = np.linspace(
            0,
            (num_frames - 1) / fps,
            args.sf * (num_frames - 1),
            endpoint=False
        )

        r_slomo = EventRenderer(
            dirname,
            output_ts,
            interpolated_ts,
            args.pos_thres,
            args.neg_thres,
            os.path.join(
                vid_out_path,
                "interpolated_{:d}.avi".format(int(args.sf*fps))
            )
        )

        # generate and save events
        r_slomo.generateEventsFromFramesAndExportEventsToHDF5(
            os.path.join(vid_out_path, "events.hdf5"))
