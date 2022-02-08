#!/bin/bash

#v2e -cs_lambda_pixels=15 --cs_tau_p_ms=.1 --output_folder spots --unique_output --dvs_aedat2=spots --output_width=346 --output_height=260 --batch=64 --disable_slomo --synthetic_input=scripts.spots
#v2e --input=input/csdvs/lamp.mp4 --pos_thres=.15 --neg_thres=.15 --sigma_thres=0 --cutoff=0 --leak_rate=0 --shot=0 --dvs_exposure duration 0.005 --dvs_aedat2 v2e.aedat --dvs346 --start_time=0 --stop_time=1 --auto_timestamp=True --show_dvs new_frame diff_frame cs_surround_frame --overwrite --cs_lambda_pixels=3 --cs_tau_p_ms=10 --output_folder=output/csdvs_lamp_test_tau_10_lam_3
#v2e --input=input/csdvs/lamp.mp4 --pos_thres=.15 --neg_thres=.15 --sigma_thres=0 --cutoff=0 --leak_rate=0 --shot=0 --dvs_exposure duration 0.005 --dvs_aedat2 v2e.aedat --dvs346 --start_time=0 --stop_time=1 --auto_timestamp=True --show_dvs new_frame diff_frame cs_surround_frame --overwrite --cs_lambda_pixels=15 --cs_tau_p_ms=20 --output_folder=output/csdvs_lamp_test_tau_20_lam_15
#v2e --input=input/csdvs/lamp.mp4 --pos_thres=.15 --neg_thres=.15 --sigma_thres=0 --cutoff=0 --leak_rate=0 --shot=0 --dvs_exposure duration 0.005 --dvs_aedat2 v2e.aedat --dvs346 --start_time=0 --stop_time=1 --auto_timestamp=True --show_dvs new_frame diff_frame cs_surround_frame --overwrite --output_folder=output/csdvs_lamp_test_dvs

#v2e --cs_lambda_pixels=30 --cs_tau_p_ms=2 --pos_thr=0.2 --neg_thr=0.2 --sigma_thr=0.02 --cutoff_hz=100 --leak_rate_hz=0 --shot_noise_rate_hz=0 --output_folder=output/spots-csdvs-l30 --vid_orig=inten --dvs_vid=csdvs --unique_output --dvs_aedat2=spots --output_width=346 --output_height=260 --batch=1 --disable_slomo --synthetic_input=scripts.spots --dt=1e-3 --freq=20 --contrast=1.5 --radius=80 --dvs_exposure=source --shot_noise_rate_hz=10

# parame for CSDVS
lambda=30
tau=2
dt=.001

# spots
#seq=spots
#v2e --cs_lambda_pixels=$lambda --cs_tau_p_ms=$tau --output_folder=output/$seq-csdvs-l$lambda-t$tau --pos_thr=0.2 --neg_thr=0.2 --sigma_thr=0.02 --cutoff_hz=100 --leak_rate_hz=0 --shot_noise_rate_hz=0  --vid_orig=inten --dvs_vid=csdvs --unique_output --dvs_aedat2=csdvs --output_width=346 --output_height=260 --batch=1 --disable_slomo --synthetic_input=scripts.spots --dt=$dt --freq=20 --contrast=1.5 --radius=80 --dvs_exposure=source --shot_noise_rate_hz=10
#v2e --output_folder=output/$seq-dvs --pos_thr=0.2 --neg_thr=0.2 --sigma_thr=0.02 --cutoff_hz=100 --leak_rate_hz=0 --shot_noise_rate_hz=0  --vid_orig=inten --dvs_vid=dvs --unique_output --dvs_aedat2=dvs --output_width=346 --output_height=260 --batch=1 --disable_slomo --synthetic_input=scripts.spots --dt=$dt --freq=20 --contrast=1.5 --radius=80 --dvs_exposure=source --shot_noise_rate_hz=10

# cloudy sky
seq=cloudy-sky
stop=5
input=input/CloudySky_Slomo.mp4
#v2e --output_folder=output/$seq-dvs --stop=$stop --input=$input --input_slowmotion_factor=10 --disable_slomo  --pos_thr=0.2 --neg_thr=0.2 --sigma_thr=0.02 --cutoff_hz=100 --leak_rate_hz=0 --vid_orig=inten --dvs_vid=dvs --unique_output --dvs_aedat2=dvs --dvs346  --dvs_exposure=source
#v2e --output_folder=output/$seq-csdvs --stop=$stop --input=$input --input_slowmotion_factor=10 --disable_slomo --cs_lambda_pixels=$lambda --cs_tau_p_ms=$tau  --pos_thr=0.2 --neg_thr=0.2 --sigma_thr=0.02 --cutoff_hz=100 --leak_rate_hz=0 --vid_orig=inten --dvs_vid=csdvs --unique_output --dvs_aedat2=dvs --dvs346  --dvs_exposure=source

# flicker
seq=flicker
start=3
stop=10
input=input/flicker.mp4
lambda=10
tau=2

# --input_slowmotion_factor=10
v2e --output_folder=output/$seq-dvs --start=$start --stop=$stop --input=$input --vid_orig=$seq-input --disable_slomo  --pos_thr=0.2 --neg_thr=0.2 --sigma_thr=0.02 --cutoff_hz=100 --leak_rate_hz=0 --dvs_vid=dvs --unique_output --dvs_aedat2=dvs --dvs346  --dvs_exposure=source
v2e --output_folder=output/$seq-csdvs --start=$start --stop=$stop --input=$input  --vid_orig=$seq-input --disable_slomo --cs_lambda_pixels=$lambda --cs_tau_p_ms=$tau  --pos_thr=0.2 --neg_thr=0.2 --sigma_thr=0.02 --cutoff_hz=100 --leak_rate_hz=0 --dvs_vid=csdvs --unique_output --dvs_aedat2=dvs --dvs346  --dvs_exposure=source
