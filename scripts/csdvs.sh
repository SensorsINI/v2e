#!/bin/bash
# scripts to reproduce results in 2022 CSDVS ICIP paper Special session 20.11 ‘Neuromorphic and perception-based image acquisition and analysis”
# UTILITY AND FEASIBILITY OF A CENTER SURROUND EVENT CAMERA

# spots
# parame for CSDVS
lambda=10
tau=.5
dt=.0001
seq=spots
contrast=2
thr=.2
sig_thr=0.02
cutoff=100
refr=1e-3
exp="duration .01"

#v2e --output_folder=output/$seq-dvs --pos_thr=$thr --neg_thr=$thr --sigma_thr=$sig_thr --cutoff_hz=$cutoff --refractory_period=$refr --leak_rate_hz=0 --shot_noise_rate_hz=0  --vid_orig=inten --dvs_vid=dvs --unique_output --dvs_aedat2=dvs --output_width=346 --output_height=260 --batch=1 --disable_slomo --synthetic_input=scripts.spots --dt=$dt --freq=20 --contrast=$contrast --dvs_exposure $exp --shot_noise_rate_hz=1 --show_dvs_model_state all --save_dvs_model_state
#v2e --output_folder=output/$seq-csdvs-l$lambda-t$tau --cs_lambda_pixels=$lambda --cs_tau_p_ms=$tau  --pos_thr=$thr --neg_thr=$thr --sigma_thr=$sig_thr --cutoff_hz=$cutoff --refractory_period=$refr --leak_rate_hz=0 --shot_noise_rate_hz=0  --vid_orig=inten --dvs_vid=csdvs --unique_output --dvs_aedat2=csdvs --output_width=346 --output_height=260 --batch=1 --disable_slomo --synthetic_input=scripts.spots --dt=$dt --freq=20 --contrast=$contrast --dvs_exposure $exp --shot_noise_rate_hz=1 --show_dvs_model_state all --save_dvs_model_state

# gradients
seq=gradients

#v2e  --output_folder output/$seq-dvs --pos_thr=$thr --neg_thr=$thr --sigma_thr=$sig_thr --cutoff_hz=$cutoff --refractory_period=$refr --leak_rate_hz=0 --shot_noise_rate_hz=0  --vid_orig=inten --dvs_vid=dvs --unique_output --dvs_aedat2=dvs --output_width=346 --output_height=260 --batch=1 --disable_slomo --dt=$dt --contrast=$contrast --dvs_exposure $exp --shot_noise_rate_hz=1 --show_dvs_model_state all --save_dvs_model_state --synthetic_input=scripts.gradients
#v2e  --output_folder output/$seq-csdvs  --cs_lambda_pixels=$lambda --cs_tau_p_ms=$tau --pos_thr=$thr --neg_thr=$thr --sigma_thr=$sig_thr --cutoff_hz=$cutoff --refractory_period=$refr --leak_rate_hz=0 --shot_noise_rate_hz=0  --vid_orig=inten --dvs_vid=dvs --unique_output --dvs_aedat2=dvs --output_width=346 --output_height=260 --batch=1 --disable_slomo --dt=$dt --contrast=$contrast --dvs_exposure $exp --shot_noise_rate_hz=1 --show_dvs_model_state all --save_dvs_model_state --synthetic_input=scripts.gradients

# cloudy sky
seq=cloudy-sky
input=input/CloudySky_Slomo.mp4
lambda=10
tau=2
slomo=33

#v2e --output_folder=output/$seq-dvs --input=$input --input_slowmotion_factor=$slomo --disable_slomo  --pos_thr=$thr --neg_thr=$thr --sigma_thr=$sig_thr --cutoff_hz=$cutoff --refractory_period=$refr --leak_rate_hz=0 --vid_orig=inten --dvs_vid=dvs --unique_output --dvs_aedat2=dvs --dvs346  --dvs_exposure=source
#v2e --cs_lambda_pixels=$lambda --cs_tau_p_ms=$tau --output_folder=output/$seq-csdvs --input=$input --input_slowmotion_factor=$slomo --disable_slomo  --pos_thr=$thr --neg_thr=$thr --sigma_thr=$sig_thr --cutoff_hz=$cutoff --refractory_period=$refr --cutoff_hz=100 --leak_rate_hz=0 --vid_orig=inten --dvs_vid=csdvs --unique_output --dvs_aedat2=csdvs --dvs346  --dvs_exposure=source

# flicker
seq=flicker
start=2
stop=12
input=input/flicker.mp4
lambda=10
tau=.5
slomo=33
dt=3e-4
thr=.2
sig_thr=0.02
cutoff=100
refr=1e-3

#v2e --timestamp_resolution=$dt --auto_timestamp=False --output_folder=output/$seq-dvs --start=$start --stop=$stop --input_slowmotion_factor=$slomo --input=$input --vid_orig=$seq-input   --pos_thr=$thr --neg_thr=$thr --sigma_thr=$sig_thr --cutoff_hz=$cutoff --refractory_period=$refr --leak_rate_hz=0 --dvs_vid=dvs --unique_output --dvs_aedat2=dvs --dvs346  --dvs_exposure source --cutoff_hz=$cutoff
#v2e --cs_lambda_pixels=$lambda --cs_tau_p_ms=$tau --timestamp_resolution=$dt --auto_timestamp=False  --output_folder=output/$seq-csdvs --start=$start --stop=$stop --input_slowmotion_factor=$slomo --input=$input  --vid_orig=$seq-input ---pos_thr=$thr --neg_thr=$thr --sigma_thr=$sig_thr --cutoff_hz=$cutoff --refractory_period=$refr --leak_rate_hz=0 --dvs_vid=csdvs --unique_output --dvs_aedat2=dvs --dvs346 --dvs_exposure=source --cutoff_hz=$cutoff


# csdvs-flashing-leds
seq=shadows
start=2
stop=7
input=input/csdvs/shadows/20220507_110937.mp4
lambda=4
tau=.1
slomo=32 # 960 FPS/30FPS=32
thr=0.05
cutoff=300
refr=.1e-3
#
#v2e --timestamp_resolution=.001 --auto_timestamp=False --output_folder=output/$seq-dvs --start=$start --stop=$stop --input_slowmotion_factor=$slomo --input=$input --vid_orig=$seq-input   --pos_thr=$thr --neg_thr=$thr --sigma_thr=$sig_thr --cutoff_hz=$cutoff --refractory_period=$refr --cutoff_hz=$cutoff --refractory_period=$refr --leak_rate_hz=0 --dvs_vid=dvs --unique_output --dvs_aedat2=dvs --dvs346
#v2e --cs_lambda_pixels=$lambda --cs_tau_p_ms=$tau --disable_slomo --output_folder=output/$seq-csdvs --start=$start --stop=$stop --input_slowmotion_factor=$slomo --input=$input  --vid_orig=$seq-input ---pos_thr=$thr --neg_thr=$thr --sigma_thr=$sig_thr --cutoff_hz=$cutoff --refractory_period=$refr --cutoff_hz=$cutoff --refractory_period=$refr --leak_rate_hz=0 --dvs_vid=csdvs --unique_output --dvs_aedat2=csdvs --dvs346 --dvs_exposure=source  --show_dvs_model_state all --save_dvs_model_state


