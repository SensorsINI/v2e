#!/bin/bash

# long sequence of just noise
seq=prnoise
thr=.2
sig_thr=0.02
cutoff=50
refr=0
refr=0
t_total=10
dt=1e-4
shot=5

v2e --photoreceptor_noise --output_folder=output/$seq-dvs --pos_thr=$thr --neg_thr=$thr --sigma_thr=$sig_thr --cutoff_hz=$cutoff --refractory_period=$refr  --leak_rate_hz=0 --shot_noise_rate_hz=$shot --dvs_vid=$seq-dvs --unique_output --dvs_aedat2=dvs --dvs346 --dvs_exposure=source --synthetic_input=scripts.moving_dot --dt=$dt --contrast=1 --t_total=$t_total