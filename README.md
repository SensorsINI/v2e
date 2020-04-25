# v2e [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python torch + opencv code to go from conventional stroboscopic video frames with low frame rate into synthetic DVS event streams with much higher effective timing precision.

See the [v2e home page](https://sites.google.com/view/video2events/home) for videos and README.



## Contact
Yuhuang Hu (yuhuang.hu@ini.uzh.ch)
Zhe He (hezhehz@live.cn)
Tobi Delbruck (tobi@ini.uzh.ch)

## Environment

```bash
python==3.7.7
```

We highly recommend running the code in virtual environment. Conda is always your best friend. :)

## Install Dependencies


```bash
pip install -r requirements.txt
```

For conda users, you can first make an env with pip in it, then install with pip. The torch and opencv-python packages are not available in conda.  Make sure this pip is first in your PATH.

```bash
conda create -n pt-v2e python=3.7 pip
conda activate pt-v2e
which pip # check to make sure new pip is first in path
pip install -r requirements.txt
```

## Usage

_v2e_ serves multiple purposes. Please read to code if you would like to adapt it for your own application. Here, we only introduce the usage for generating DVS events from conventional video and from specific datasets.

**NOTE** We recommend running v2e on a CUDA GPU or it will be very slow.

## Download Checkpoint

We use the excellent [Super SloMo](https://people.cs.umass.edu/~hzjiang/projects/superslomo/) framework to interpolate the APS frames. 
However, since APS frames only record light intensity, we  retrained it on grayscale images. 
You can download our pre-trained model checkpoint from Google Drive 
[[link]](https://drive.google.com/file/d/17QSN207h05S_b2ndXjLrqPbBTnYIl0Vb/view?usp=sharing) (151 MB).


```bash
mkdir -p input
mv SuperSloMo39.ckpt ./input
```
The default value of --slomo_model argument is set to this location.

## Render emulated DVS events from conventional video.

_v2e.py_ reads a standard video (e.g. in .avi, .mp4, .mov, or .wmv) and generates emulated DVS events at upsampled timestamp resolution.

```bash
(base)$ conda activate pt-v2e # activate your workspace 
(pt-v2e)$ python v2e.py -h
usage: v2e.py [-h] [-i INPUT] [--start_time START_TIME]
              [--stop_time STOP_TIME] [--pos_thres POS_THRES]
              [--neg_thres NEG_THRES] [--sigma_thres SIGMA_THRES]
              [--cutoff_hz CUTOFF_HZ] [--leak_rate_hz LEAK_RATE_HZ]
              [--slowdown_factor SLOWDOWN_FACTOR]
              [--output_height OUTPUT_HEIGHT] [--output_width OUTPUT_WIDTH]
              [--rotate180] [--slomo_model SLOMO_MODEL] -o OUTPUT_FOLDER
              [--frame_rate FRAME_RATE] [--dvs_vid DVS_VID]
              [--dvs_vid_full_scale DVS_VID_FULL_SCALE] [--dvs_h5 DVS_H5]
              [--dvs_aedat2 DVS_AEDAT2] [--dvs_text DVS_TEXT]
              [--vid_orig VID_ORIG] [--vid_slomo VID_SLOMO] [-p] [--overwrite]

v2e: generate simulated DVS events from video.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input video file; leave empty for file chooser dialog
                        (default: None)
  --start_time START_TIME
                        start at this time in seconds in video (default: None)
  --stop_time STOP_TIME
                        stop at this time in seconds in video (default: None)
  --pos_thres POS_THRES
                        threshold in log_e intensity change to trigger a
                        positive event (default: 0.21)
  --neg_thres NEG_THRES
                        threshold in log_e intensity change to trigger a
                        negative event (default: 0.17)
  --sigma_thres SIGMA_THRES
                        1-std deviation threshold variation in log_e intensity
                        change (default: 0.03)
  --cutoff_hz CUTOFF_HZ
                        photoreceptor first order IIR lowpass cutoff-off 3dB
                        frequency in Hz - see
                        https://ieeexplore.ieee.org/document/4444573 (default:
                        300)
  --leak_rate_hz LEAK_RATE_HZ
                        leak event rate per pixel in Hz - see
                        https://ieeexplore.ieee.org/abstract/document/7962235
                        (default: 0.05)
  --slowdown_factor SLOWDOWN_FACTOR
                        slow motion factor; if the input video has frame rate
                        fps, then the DVS events will have time resolution of
                        1/(fps*slowdown_factor) (default: 10)
  --output_height OUTPUT_HEIGHT
                        height of output DVS data in pixels. If None, same as
                        input video. (default: None)
  --output_width OUTPUT_WIDTH
                        width of output DVS data in pixels. If None, same as
                        input video. (default: None)
  --rotate180           rotate all output 180 deg (default: False)
  --slomo_model SLOMO_MODEL
                        path of slomo_model checkpoint (default:
                        input/SuperSloMo39.ckpt)
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        folder to store outputs (default: None)
  --frame_rate FRAME_RATE
                        equivalent frame rate of --dvs_vid output video; the
                        events will be accummulated as this sample rate; DVS
                        frames will be accumulated for duration 1/frame_rate
                        (default: 300)
  --dvs_vid DVS_VID     output DVS events as AVI video at frame_rate (default:
                        dvs-video.avi)
  --dvs_vid_full_scale DVS_VID_FULL_SCALE
                        set full scale count for DVS videos to be this many ON
                        or OFF events (default: 3)
  --dvs_h5 DVS_H5       output DVS events as hdf5 event database (default:
                        None)
  --dvs_aedat2 DVS_AEDAT2
                        output DVS events as AEDAT-2.0 event file for jAER
                        (default: None)
  --dvs_text DVS_TEXT   output DVS events as text file with one event per line
                        [timestamp (float s), x, y, polarity (0,1)] (default:
                        None)
  --vid_orig VID_ORIG   output src video at same rate as slomo video (with
                        duplicated frames) (default: video_orig.avi)
  --vid_slomo VID_SLOMO
                        output slomo of src video slowed down by
                        slowdown_factor (default: video_slomo.avi)
  --no_preview         disable preview in cv2 windows for faster processing (default: False)
  --overwrite           overwrites files in existing folder (checks existence
                        of non-empty output_folder) (default: False)

Run with no --input to open file dialog
```
You can put [tennis.mov](https://drive.google.com/file/d/1dNUXJGlpEM51UVYH4-ZInN9pf0bHGgT_/view?usp=sharing) in the __input__ folder to try it out with the command line below.

```bash
python v2e.py --i input/tennis.mov --slowdown_factor=10 --o=output --pos_thres=.15 --neg_thres=.15 --sigma_thres=0.01 --frame_rate=300 --dvs_aedat2 tennis.aedat --output_width=346 --output_height=260
```
Run the command above, and the following files will be created in a folder called _output_.

```bash
original.avi  slomo.avi  dvs-video.avi  tennis.aedat 
```

* _original.avi_: input video, but converted to luma and resized to output (width,height) and with repeated frames to allow comparison to _slomo.avi_.
* _slomo.avi_: slow motion video (with playback rate 30Hz) but slowed down by slowdown_factor.
* _dvs-video.avi_: DVS video (with playback rate 30Hz) but with frame rate (DVS timestamp resolution) set by source video frame rate times slowdown_factor.
* _tennis.aedat_: AEDAT-2.0 file for playback and algorithm experiments in [jAER](https://jaerproject.net) (use the AEChip _Davis346Blue_ to play this file.)

The [v2e site](https://sites.google.com/view/video2events/home) shows these videos.
 
## DAVIS camera conversion Dataset

v2e can convert recordings from [DDD17](https://docs.google.com/document/d/1HM0CSmjO8nOpUeTvmPjopcBcVCk7KXvLUuiZFS6TWSg/pub) which is the first public end-to-end training dataset 
of automotive driving using a DAVIS event + frame camera. It lets you compare the real DVS data with the conversion. 
This dataset is maintained by the Sensors Research Group of Institute of Neuroinformatics. 
Please go to the datasets website [[link]](http://sensors.ini.uzh.ch/databases.html) of Sensors Group  for details about downloading _DDD17_.

For your convenience, we put one recording from _DDD20_ (our newer DDD dataset) of 800s of
Los Angeles street driving. 
The file is _aug04/rec1501902136.hdf5_ [[link]](https://drive.google.com/open?id=1KIaHsn72ZpVBZR6SGeFcd2lILyZoD2-5)
  in Google Drive for you to try it with v2e (***Warning:*** 2GB 7z compressed, 5.4 GB uncompressed).

```bash
mkdir -p input
mv rec1501902136.hdf5 ./input
```

### Extract data from DDD recording

_ddd_h5_extract_data.py_ extracts the DDD recording DVS events to jAER _.aedat_ and video _.avi_ files.

```bash
(pt-v2e) $ python ddd_h5_extract_data.py -h
usage: ddd_h5_extract_data.py [-h] [-i INPUT] -o OUTPUT_FOLDER
                              [--start_time START_TIME]
                              [--stop_time STOP_TIME] [--rotate180]
                              [--overwrite]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input video file; leave empty for file chooser dialog
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        folder to store outputs
  --start_time START_TIME
                        start at this time in seconds in video
  --stop_time STOP_TIME
                        stop point of video stream
  --rotate180           rotate output 180 deg
  --overwrite           overwrites files in existing folder (checks existance
                        of non-empty output_folder)

```

### Synthesize events from DDD recording

_ddd20-v2e.py_ is like v2e.py but it reads DDD .hdf5 recordings and extracts the real DVS events from the same part of the recording used for the synthesis of DVS events.

You can try it like this:
```bash
/home/tobi/anaconda3/envs/pt-v2e/bin/python /home/tobi/Dropbox/GitHub/SensorsINI/v2e/ddd20-v2e.py --input input/rec1501350986.hdf5 --slomo_model input/SuperSloMo39.ckpt --slowdown_factor 20 --start 70 --stop 73 --output_folder output/ddd20-v2e-short --dvs_aedat dvs --pos_thres=.2 --neg_thres=.2 --overwrite --dvs_vid_full_scale=2 --frame_rate=100
INFO:__main__:arguments:
cutoff_hz:      300
dvs_aedat2:     dvs
dvs_h5: None
dvs_text:       None
dvs_vid:        dvs-video.avi
dvs_vid_full_scale:     2
frame_rate:     100
input:  input/rec1501350986.hdf5
leak_rate_hz:   0.05
neg_thres:      0.2
no_preview:     False
output_folder:  output/ddd20-v2e-short
output_height:  260
output_width:   346
overwrite:      True
pos_thres:      0.2
rotate180:      True
sigma_thres:    0.03
slomo_model:    input/SuperSloMo39.ckpt
slowdown_factor:        20
start_time:     70.0
stop_time:      73.0
vid_orig:       video_orig.avi
vid_slomo:      video_slomo.avi

INFO:__main__:opening output files
INFO:src.slomo:CUDA available, running on GPU :-)
INFO:src.emulator:ON/OFF log_e temporal contrast thresholds: 0.2 / 0.2 +/- 0.03
INFO:src.emulator:opening AEDAT-2.0 output file output/ddd20-v2e-short/dvs.aedat
INFO:root:opening AEDAT-2.0 output file output/ddd20-v2e-short/dvs.aedat in binary mode
INFO:src.output.aedat2_output:opened output/ddd20-v2e-short/dvs.aedat for DVS output data for jAER
INFO:src.ddd20_utils.ddd_h5_reader:making reader for DDD recording input/rec1501350986.hdf5
INFO:src.ddd20_utils.ddd_h5_reader:input/rec1501350986.hdf5 contains following keys
accelerator_pedal_position
brake_pedal_status
dvs
engine_speed
fine_odometer_since_restart
fuel_consumed_since_restart
fuel_level
gear_lever_position
headlamp_status
high_beam_status
ignition_status
lateral_acceleration
latitude
longitude
longitudinal_acceleration
odometer
parking_brake_status
steering_wheel_angle
torque_at_transmission
transmission_gear_position
vehicle_speed
windshield_wiper_status
INFO:src.ddd20_utils.ddd_h5_reader:group dvs contains following keys
data
timestamp
INFO:src.ddd20_utils.ddd_h5_reader:group dvs contains following items
('data', <HDF5 dataset "data": shape (38912, 3), type "|O">)
('timestamp', <HDF5 dataset "timestamp": shape (38912,), type "<i8">)
INFO:src.ddd20_utils.ddd_h5_reader:The DAVIS data has the shape (38912, 3)
INFO:src.ddd20_utils.ddd_h5_reader:input/rec1501350986.hdf5 has 38271 packets with start time 1123.34s and end time 1246.31s (duration    123.0s)
INFO:src.ddd20_utils.ddd_h5_reader:searching for time 70.0
ddd-h5-search:  56%|███████████████████████████████████████▉                                | 21244/38271 [00:08<00:05, 3002.45packet/s]INFO:src.ddd20_utils.ddd_h5_reader:
found start time 70.0 at packet 21369

INFO:src.ddd20_utils.ddd_h5_reader:searching for time 73.0
ddd-h5-search:  58%|█████████████████████████████████████████▊                              | 22197/38271 [00:08<00:05, 2692.97packet/s]INFO:src.ddd20_utils.ddd_h5_reader:
found start time 73.0 at packet 22469

INFO:__main__:iterating over input file contents
v2e-ddd20:   0%|                                                                                           | 0/1100 [00:00<?, ?packet/s]INFO:root:opening AEDAT-2.0 output file output/ddd20-v2e-short/dvs-real.aedat in binary mode
INFO:src.output.aedat2_output:opened output/ddd20-v2e-short/dvs-real.aedat for DVS output data for jAER
INFO:src.renderer:opening DVS video output file dvs-video-real.avi
DEBUG:src.v2e_utils:opened output/ddd20-v2e-short/dvs-video-real.avi with  XVID https://www.fourcc.org/ codec, 30.0fps, and (346x260) size
v2e-ddd20:   5%|███▊                                                                              | 51/1100 [00:00<00:15, 67.70packet/s]INFO:src.slomo:loading SuperSloMo model from input/SuperSloMo39.ckpt

```

The generated outputs will be
```angular2
dvs-v2e.aedat
dvs-v2e-real.aedat
dvs-video-fake.avi
dvs-video-real.avi
info.txt
original.avi
slomo.avi
```


### Plot the Events

```bash
python plot.py \
--path [path of input files] \
--bin_size [size of the time bin] \
--start [start timestamp] \
--stop [stop timestamp] \
--x [range of x coordinate] \
--y [range of y coordinate] \
--rotate [if the video needs to be rotated]
```

'--path' is the folder which contains the output files generated by executing 'v2e_h5.py'.

'--rotate' is **IMPORTANT**, because some files in the DDD20 dataset are recorded upside down. More information regarding this can be found in the documentation of DDD20 dataset.

One example is shown below, the left side is the ground-truth DVS frames, and the figure on the right side shows the histogram plot of the generated self within the region denoted by the black box. Histograms of the ground-truth self and our generated self are plotted in the same figure. It can be seen that the distribution of generated self is quite similar to the distribution of the real self.

<p float="left">
  <img src="media/counting.gif" width="320" class="center" />
  <img src="media/plot.png" width="350"  class="center"/> 
</p>

## Calibrate the Thresholds

To get the threshold of triggering an event, you need to run the commands below.

```bash
python renderer_sweep.py \
--start [start] \
--stop [end] \
--fname [path to the .hdf5 DVS recording file]] \
--checkpoint [the .ckpt checkpoint of the slow motion network] \
--sf [slow motion factor]
```

The program will take the DVS recording data, which starts at time 'start' and ends at time 'end', to calculate the best threshold values for positive and negative self separately.

For the best frame interpolation by SuperSloMo, the input video needs to satisfy the requirements below,

- Daytime
- Cloudy
- No rain
- High frame rate

If the video is underexposed, overexposed, has motion blur or aliasing, then the emulated DVS events will have poor realism.

### Default Thresholds ####
_pos_thres_: 0.25
_neg_thres_: 0.35
Both of them are approximated based on the file rec1500403661.hdf5.

**NOTE** 

The thresholds vary slightly depending on the time interval of the input APS frames.

|  Time Interval   |  _pos_thres_ | _neg_thres_ |
|  ----  | ----  | ----|
| 5s - 15s  | 0.25 | 0.36|
| 15s - 25s  | 0.24 | 0.33|
| 25s - 35s  | 0.21 | 0.31|
| 35s - 45s  | 0.22 | 0.33|

All the thresholds above are estimated based on the file rec1500403661.hdf5. The estimated thresholds also slightly vary depending on the input file. For example, based on the APS frames in the time interval 35s - 45s from the file rec1499025222.hdf5, the estimated positive threshold is 0.28, and the estimated negative threshold is 0.42.

## Generating Synthetic DVS Dataset from UCF-101 action recognition dataset ##

To generate synthetic data from a single input video UCF-101 [[link] (https://www.crcv.ucf.edu/data/UCF101.php)] .
video

```bash
python ucf101_single.py \
--input [path to the input video] \
--pos_thres [positive threshold] \
--neg_thres [negative threshold] \
--sf [slow motion factor] \
--checkpoint [the .ckpt checkpoint of the slow motion network] \
--output_dir [path to store the output videos]
```

The code needs to be modified accordingly if the input video is from a different dataset.

## Technical Details ##

Click [PDF](docs/technical_report.pdf) to download.
