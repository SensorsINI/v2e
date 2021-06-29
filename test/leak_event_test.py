"""Use a static frame, and only generate leak events."""

import os
import torch
import cv2

from v2ecore.emulator import EventEmulator

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_width, output_height = 346, 260

# create a emulator
emulator = EventEmulator(
    pos_thres=0.05,
    neg_thres=0.05,
    sigma_thres=0.02,
    cutoff_hz=0,
    leak_rate_hz=0.2,
    shot_noise_rate_hz=0,
    leak_jitter_fraction=0.5,
    noise_rate_cov_decades=0.3,
    device=torch_device,
    output_folder=os.path.join(os.environ["HOME"], "data"),
    dvs_aedat2="leak_event_test_leak_rate=0.2_with_jitter=0.5_noise_cov=0.3.aedat",
    output_width=output_width,
    output_height=output_height
)

# static video stats
fps = 500
delta_t = 1/fps

emulation_time_in_sec = 120
emulation_cycles = int(emulation_time_in_sec//delta_t)
current_time = 0

# load data
img = cv2.imread(os.path.join(
    os.environ["HOME"], "data", "lena.jpg"))

# resize and convert
img = cv2.resize(img, dsize=(output_width, output_height))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # much faster

# simulation start
for it in range(emulation_cycles):
    print("\rEmulating Frame: {} at {}s".format(it, current_time), end="")

    emulator.generate_events(img, current_time)

    # add time
    current_time += delta_t
