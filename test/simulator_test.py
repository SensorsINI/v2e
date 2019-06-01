import sys

sys.path.append("../")
sys.path.append("../src/")
sys.path.append("../utils/")


if __name__ == "__main__":

    from simulator import EventFrameRenderer

    e = EventFrameRenderer(
        "../data/tmpSloMo/",
        "../data/",
        17,
        17,
        0.01
    )
    e.render()
