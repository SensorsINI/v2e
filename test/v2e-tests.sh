#!/bin/bash
# run various tests

# print usage
python v2e.py -h

# convert a synthetic file with clean ideal DVS model
python v2e.py  --input input/box-moving-white.mp4 --output_folder=output/v2e-test-white-dot-clean-1ms --unique_output_folder --pos_thres=.15 --neg_thres=.15 --sigma_thres=0 --cutoff=0 --leak_rate=0 --shot=0 --dvs_exposure duration 0.005 --timestamp=1e-3 --dvs_aedat2 v2e.aedat --output_width=346 --output_height=260 --batch=8 --no_preview

# convert a synthetic file with slow bandwidth and lots of noise
python v2e.py  --input input/box-moving-white.mp4 --output_folder=output/v2e-test-white-dot-noisy-1ms --unique_output_folder --pos_thres=.15 --neg_thres=.15 --sigma_thres=0.05 --cutoff=10 --leak_rate=0.1 --shot=0.1 --dvs_exposure duration 0.005 --timestamp=1e-3 --dvs_aedat2 v2e.aedat --output_width=346 --output_height=260 --batch=8 --no_preview

