"""
Read hdf5 dvs data and return aps frames + events.
@author: Zhe He
@contact:hezhehz@live.cn
@latest update: 2019-May-31
"""

import queue as Queue
import time
# import pdb
import numpy as np

from utils.view import HDF5Stream, MergedStream
from utils.caer import unpack_data


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

    def __init__(self, fname, start=None, stop=None):
        """
        Init
        @params:
            fname: str
                path of input hdf5 file.
        """

        self.m = MergedStream(HDF5Stream(fname, {'dvs'}))
        self.start = int(self.m.tmin + 1e6 * start) if start else 0
        self.stop = (self.m.tmin + 1e6 * stop) if stop else self.m.tmax
        self.m.search(self.start)

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
        frames, events = [], []
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
                    # NOTE the timestamp of this special event is not meaningful
                    continue
            if d['etype'] == 'frame_event':
                ts = d['timestamp'] + t_offset
                frame = filter_frame(unpack_data(d))
                data = np.array(
                    [(ts, frame)],
                    dtype=np.dtype(
                        [('ts', np.float64),
                         ('frame', np.uint8, frame.shape)]
                    )
                )
                frames.append(data)
                current = ts
                continue
            if d['etype'] == 'polarity_event':
                unpack_data(d)
                data = np.core.records.fromarrays(
                    d["data"].transpose(),
                    dtype=np.dtype([("ts", np.float64),
                                    ("y", np.uint32),
                                    ("x", np.uint32),
                                    ("polarity", np.int8)])
                )
                data["ts"] = data["ts"] * 1e-6 + t_offset
                data["polarity"] = data["polarity"] * 2 - 1
                events.append(data)
                current = data["ts"][-1]
        frames = np.hstack(frames)
        events = np.hstack(events)
        frames["ts"] -= frames["ts"][0]
        events["ts"] -= events["ts"][0]

        return frames, events
