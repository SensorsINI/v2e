"""test slomo.py
    @author: Zhe He
    @contact: hezhehz@live.cn
    @latest update: 2019-May-29th
"""
import numpy as np
import sys
import pdb

if __name__ == "__main__":

    sys.path.append("../")
    sys.path.append("../src/")
    sys.path.append("../utils/")

    from slomo import SuperSloMo
    from reader import Reader

    images = np.load("../data/frames.npy")
    checkpoint = "../data/SuperSloMo38.ckpt"
    slow_factor = 5
    output_path = "../data/tmpSloMo/"
    super_slomo = SuperSloMo(
        checkpoint,
        slow_factor,
        output_path
    )
    # super_slomo.interpolate(images)

    fname = "../data/rec1487354811.hdf5"

    m = Reader(fname, start=5, stop=10)
    frames, events = m.readEntire()
    new_ts = super_slomo.get_ts(frames["ts"])
    pdb.set_trace()
