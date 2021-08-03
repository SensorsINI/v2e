#!/bin/bash
# example of converting a folder chunk by chunk if there are a large number of frames

start=3020 # frame to start at
end=4800 # end frame
step=20 # from many source frames in each chunk
in_folder="img" # input folder
out_folder="pfb-v2e" # output folder base name, interval will be appended as -from-to
fps=12 # frame rate of frames

 echo "**************** Converting $start to $end with steps of $step *******************************"

for ((from=$start;from<=$end;from+=$step));
do
 to=$(($from + $step))
 echo "**************** Converting $from to $to *******************************"
 v2e --input_frame_rate $fps -i $in_folder -o $out_folder-$from-$to --unique  --start $from --stop  $to --dvs_aedat2 $from-$to.aedat --timestamp_res 0.001 --dvs346 || exit 1
done