This folder has source code for _v2e_.

_emulator.py_: the DVS model that generates events from a new frame. It is stateful model that takes new frame and updates each pixel model, generating events from the update.

_renderer.py_: It renders DVS events back to frames, using a variety of methods as described in the top README.md.

_slomo.py_: Our implementation of SuperSloMo that interpolates frames.

_model.py_: The generic UNet model used by SuperSloMo.

_v2e_args.py_: All the complex arguments of v2e are collected here.

_dataloader.py_: The complex dataloader for superslomo that handles batching.

