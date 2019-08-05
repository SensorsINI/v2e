"""
Python code for extracting frames from .hdf5 file in DDD17+ dataset.

@author: Zhe He
@contact: zhehe@student.ethz.ch
@latest update: 2019-June-28 22:27
"""

import numpy as np
import argparse
import cv2
import queue as Queue
import time
# import pdb

from utils.view import HDF5Stream, MergedStream
from utils.caer import unpack_data

from src.slomo import video_writer


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
    "--start",
    type=float,
    default=None,
    help="start point of video stream"
)

parser.add_argument(
    "--stop",
    type=float,
    default=None,
    help="stop point of video stream"
)

parser.add_argument(
    "--rotate",
    action="store_true"
)

args = parser.parse_args()


def filter_frame(d):
    '''
    receives 16 bit frame,
    needs to return unsigned 8 bit img
    '''
    # add custom filters here...
    # d['data'] = my_filter(d['data'])
    frame8 = (d['data'] / 256).astype(np.uint8)
    return frame8


class Reader(object):
    """
    Read aps frames and events from hdf5 files in ddd17+.
    @author: Zhe He
    @contact: hezhehz@live.cn
    @latest update: 2019-May-31
    """

    def __init__(
        self,
        fname,
        video_writer,
        start=None,
        stop=None,
        rotate=False
    ):
        """
        Init
        @params:
            fname: str
                path of input hdf5 file.
            video_writer: cv2.VideoWriter
                video writer.
            start: float or int
                start of the video.
            stop: float or int
                end of the video.
        """

        self.m = MergedStream(HDF5Stream(fname, {'dvs'}))
        self.start = int(self.m.tmin + 1e6 * start) if start else 0
        self.stop = (self.m.tmin + 1e6 * stop) if stop else self.m.tmax
        self.m.search(self.start)
        self.writer = video_writer

    def read(self):
        """
        Read data.
        @return:
            aps_ts: np.array,
                timestamps of aps frames.
            aps_frame: np.ndarray, [n, width, height]
                aps frames
            events: numpy record array.
                events, col names: ["ts", "y", "x", "polarity"], \
                    data types: ["<f8", "<i8", "<i8", "<i8"]
        """
        sys_ts, t_offset, current = 0, 0, 0
        timestamp = 0
        while self.m.has_data and sys_ts <= self.stop * 1e-6:
            try:
                sys_ts, d = self.m.get()
            except Queue.Empty:
                # wait for queue to fill up
                time.sleep(0.01)
                continue
            if not d or sys_ts < self.start * 1e-6:
                # skip unused data
                continue
            if d['etype'] == 'special_event':
                unpack_data(d)
                # this is a timestamp reset
                if any(d['data'] == 0):
                    print('ts reset detected, setting offset', timestamp)
                    t_offset += current
                    # NOTE the timestamp of this special event is
                    # not meaningful
                    continue
            if d['etype'] == 'frame_event':
                ts = d['timestamp'] + t_offset
                frame = filter_frame(unpack_data(d))
                current = ts

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
                continue
            if d['etype'] == 'polarity_event':
                continue


if __name__ == "__main__":

    writer = video_writer(args.output_path, 260, 346)
    r = Reader(
        args.fname,
        writer,
        start=args.start,
        stop=args.stop,
        rotate=args.rotate
    )
    r.read()
