#!/bin/bash

v2e --input=input/csdvs/lamp.mp4 --pos_thres=.15 --neg_thres=.15 --sigma_thres=0 --cutoff=0 --leak_rate=0 --shot=0 --dvs_exposure duration 0.005 --dvs_aedat2 v2e.aedat --dvs346 --start_time=0 --stop_time=1 --auto_timestamp=True --show_dvs new_frame diff_frame cs_surround_frame --overwrite --cs_lambda_pixels=3 --cs_tau_p_ms=10 --output_folder=output/csdvs_lamp_test_tau_10_lam_3
#
#v2e --input=input/csdvs/lamp.mp4 --pos_thres=.15 --neg_thres=.15 --sigma_thres=0 --cutoff=0 --leak_rate=0 --shot=0 --dvs_exposure duration 0.005 --dvs_aedat2 v2e.aedat --dvs346 --start_time=0 --stop_time=1 --auto_timestamp=True --show_dvs new_frame diff_frame cs_surround_frame --overwrite --cs_lambda_pixels=15 --cs_tau_p_ms=20 --output_folder=output/csdvs_lamp_test_tau_20_lam_15
#
#v2e --input=input/csdvs/lamp.mp4 --pos_thres=.15 --neg_thres=.15 --sigma_thres=0 --cutoff=0 --leak_rate=0 --shot=0 --dvs_exposure duration 0.005 --dvs_aedat2 v2e.aedat --dvs346 --start_time=0 --stop_time=1 --auto_timestamp=True --show_dvs new_frame diff_frame cs_surround_frame --overwrite --output_folder=output/csdvs_lamp_test_dvs