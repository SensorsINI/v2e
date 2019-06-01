import sys
import numpy as np


if __name__ == "__main__":
    sys.path.append("../")
    sys.path.append("../src/")
    sys.path.append("../utils/")
    from src.renderer import RenderFromImages
    from src.slomo import SuperSloMo
    from src.reader import Reader

    fname = "../data/rec1500394622.hdf5"

    m = Reader(fname, start=5, stop=5.1)
    frames, events = m.read()

    s = SuperSloMo(
        "../data/SuperSloMo38.ckpt",
        10,
        "../data/tmpSloMo/"
    )

    s.interpolate(frames["frame"])

    frame_ts = s.timestamps(frames["ts"])

    r = RenderFromImages(
        "../data/tmpSloMo/",
        frame_ts,
        0.02,
        "../data/")
    print(frames["frame"].shape)
    r.render(frames["frame"].shape[1], frames["frame"].shape[2])
