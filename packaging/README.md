# v2e

v2e turns conventional video into realistic synthetic DVS event streams. It models photoreceptor bandwidth, pixel-to-pixel threshold variation, and leak noise.

![v2e tennis example](https://raw.githubusercontent.com/SensorsINI/v2e/v2e-1.8.3/media/v2e-tennis-split-screen.gif)

The full guide, quick start, and option list are in the [v2e README](https://github.com/SensorsINI/v2e#readme).

## Install

v2e requires Python 3.14.

```bash
uv tool install v2e
```

On a machine with an NVIDIA GPU, install the matching PyTorch build in that environment:

```bash
uv pip install torch torchvision --torch-backend=auto
```

## Try it

Download [tennis.mov](https://drive.google.com/file/d/1dNUXJGlpEM51UVYH4-ZInN9pf0bHGgT_/view?usp=sharing), then run v2e with no arguments and choose that file:

```bash
v2e
```

v2e opens a file chooser, prints the options it will use, and converts the first second after you answer `y`. A shorter clip is converted in full. The result is `events.aedat4` in the output folder. Open that file in [jAER](https://jaerproject.org).

## Citation

Y. Hu, S-C. Liu, and T. Delbruck. [v2e: From Video Frames to Realistic DVS Events](https://arxiv.org/abs/2006.07722). CVPRW, 2021.
