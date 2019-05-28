"""test slomo.py
    @author: Zhe He
    @contact: hezhehz@live.cn
    @latest update: 2019-May-29th
"""
import numpy as np
import sys

if __name__ == "__main__":

    sys.path.append("../src/")

    from slomo import SuperSloMo

    images = np.load("../data/frames.npy")
    checkpoint = "../data/SuperSloMo38.ckpt"
    slow_factor = 5
    output_path = "../data/tmpSloMo/"
    super_slomo = SuperSloMo(
        checkpoint,
        slow_factor,
        output_path
    )
    super_slomo.run(images)
