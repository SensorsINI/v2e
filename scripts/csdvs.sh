#!bash

# spots
# parame for CSDVS
lambda=10
tau=.5
dt=.001
seq=spots
contrast=2

#v2e --output_folder=output/$seq-dvs --pos_thr=0.2 --neg_thr=0.2 --sigma_thr=0.02 --cutoff_hz=100 --leak_rate_hz=0 --shot_noise_rate_hz=0  --vid_orig=inten --dvs_vid=dvs --unique_output --dvs_aedat2=dvs --output_width=346 --output_height=260 --batch=1 --disable_slomo --synthetic_input=scripts.spots --dt=$dt --freq=20 --contrast=$contrast --dvs_exposure=source --shot_noise_rate_hz=1
#v2e --output_folder=output/$seq-csdvs-l$lambda-t$tau --cs_lambda_pixels=$lambda --cs_tau_p_ms=$tau  --pos_thr=0.2 --neg_thr=0.2 --sigma_thr=0.02 --cutoff_hz=100 --leak_rate_hz=0 --shot_noise_rate_hz=0  --vid_orig=inten --dvs_vid=csdvs --unique_output --dvs_aedat2=csdvs --output_width=346 --output_height=260 --batch=1 --disable_slomo --synthetic_input=scripts.spots --dt=$dt --freq=20 --contrast=$contrast --dvs_exposure=source --shot_noise_rate_hz=1

# cloudy sky
seq=cloudy-sky
input=input/CloudySky_Slomo.mp4
lambda=10
tau=2
slomo=33

#v2e --output_folder=output/$seq-dvs --input=$input --input_slowmotion_factor=$slomo --disable_slomo  --pos_thr=0.2 --neg_thr=0.2 --sigma_thr=0.02 --cutoff_hz=100 --leak_rate_hz=0 --vid_orig=inten --dvs_vid=dvs --unique_output --dvs_aedat2=dvs --dvs346  --dvs_exposure=source
#v2e --cs_lambda_pixels=$lambda --cs_tau_p_ms=$tau --output_folder=output/$seq-csdvs --input=$input --input_slowmotion_factor=$slomo --disable_slomo  --pos_thr=0.2 --neg_thr=0.2 --sigma_thr=0.02 --cutoff_hz=100 --leak_rate_hz=0 --vid_orig=inten --dvs_vid=csdvs --unique_output --dvs_aedat2=csdvs --dvs346  --dvs_exposure=source

# flicker
seq=flicker
start=2
stop=12
input=input/flicker.mp4
lambda=10
tau=2
slomo=33
#
#v2e --timestamp_resolution=.001 --auto_timestamp=False --output_folder=output/$seq-dvs --start=$start --stop=$stop --input_slowmotion_factor=$slomo --input=$input --vid_orig=$seq-input   --pos_thr=0.2 --neg_thr=0.2 --sigma_thr=0.02 --cutoff_hz=100 --leak_rate_hz=0 --dvs_vid=dvs --unique_output --dvs_aedat2=dvs --dvs346  --dvs_exposure source
#v2e --cs_lambda_pixels=$lambda --cs_tau_p_ms=$tau --timestamp_resolution=.001 --auto_timestamp=False  --output_folder=output/$seq-csdvs --start=$start --stop=$stop --input_slowmotion_factor=$slomo --input=$input  --vid_orig=$seq-input --pos_thr=0.2 --neg_thr=0.2 --sigma_thr=0.02 --cutoff_hz=100 --leak_rate_hz=0 --dvs_vid=csdvs --unique_output --dvs_aedat2=dvs --dvs346 --dvs_exposure=source


# csdvs-flashing-leds
seq=shadows
start=2
stop=7
input=input/csdvs/shadows/20220507_110937.mp4
lambda=8
tau=.5
slomo=32 # 960 FPS/30FPS=32
thr=0.2
cutoff=300
refr=.1e-3
#
#v2e --timestamp_resolution=.001 --auto_timestamp=False --output_folder=output/$seq-dvs --start=$start --stop=$stop --input_slowmotion_factor=$slomo --input=$input --vid_orig=$seq-input   --pos_thr=0.2 --neg_thr=0.2 --sigma_thr=0.02 --cutoff_hz=100 --leak_rate_hz=0 --dvs_vid=dvs --unique_output --dvs_aedat2=dvs --dvs346  --dvs_exposure source
v2e --cs_lambda_pixels=$lambda --cs_tau_p_ms=$tau --disable_slomo --output_folder=output/$seq-csdvs --start=$start --stop=$stop --input_slowmotion_factor=$slomo --input=$input  --vid_orig=$seq-input --pos_thr=$thr --neg_thr=$thr --sigma_thr=0.01 --cutoff_hz=$cutoff --refractory_period=$refr --leak_rate_hz=0 --dvs_vid=csdvs --unique_output --dvs_aedat2=dvs --dvs346 --dvs_exposure=source  --show_dvs_model_state diff_frame cs_surround_frame new_frame base_log_frame lp_log_frame1 --save_dvs_model_state