import sys
import numpy as np


if __name__ == "__main__":
    sys.path.append("../")
    sys.path.append("../src/")
    sys.path.append("../utils/")
    from src.renderer import RenderFromImages, RenderFromEvents
    from src.slomo import SuperSloMo
    from src.reader import Reader

    fname = "../data/rec1487354811.hdf5"

    m = Reader(fname, start=5, stop=5.5)
    frames, events = m.read()

    s = SuperSloMo(
        "../data/SuperSloMo38.ckpt",
        10,
        "../data/tmpSloMo/"
    )

    s.interpolate(frames["frame"])
    frame_ts = s.get_ts(frames["ts"])
    height, width = frames["frame"].shape[1:]

    r = RenderFromImages(
        "../data/tmpSloMo/",
        frame_ts,
        0.02,
        "../data/from_image.avi")
    r.render(height, width)

    r_events = RenderFromEvents(
        frame_ts,
        events,
        "../data/from_events.avi"
    )

    r_events.render(height, width)
