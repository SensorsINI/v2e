# v2e [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1czx-GJnx-UkhFVBbfoACLVZs8cYlcr_M?usp=sharing)

Python torch + opencv code to go from conventional stroboscopic video frames with low frame rate into realistic synthetic DVS event streams with much higher effective timing precision. v2e includes finite intensity-depenedent photoreceptor bandwidth, Gaussian pixel to pixel event threshold variation, and noise 'leak' events.

See the [v2e home page](https://sites.google.com/view/video2events/home) for videos and further information. 

Our paper (below) about _v2e_ **debunks incorrect assertions about event cameras that pervade the current computer vision literature.**

See [v2e-sample-input-data](https://drive.google.com/drive/folders/1oWxTB9sMPp6UylAdxg5O1ko_GaIXz7wo?usp=sharing) for sample input files.

![v2e-tennis-example](media/v2e-tennis-split-screen.gif)

Vote for [new v2e features](https://docs.google.com/forms/d/e/1FAIpQLSdJoIH3wBkPANWTng56VeXiItkh_fl5Lz3QwZIpQ6ut1AMFCw/viewform?usp=sf_link).

## News

See [changelog](https://github.com/SensorsINI/v2e/blob/master/CHANGELOG.md) for latest news

## Contact
Yuhuang Hu (yuhuang.hu@ini.uzh.ch)
Tobi Delbruck (tobi@ini.uzh.ch)

### Citation
If you use v2e, we appreciate a citation to the paper below. See the [v2e home page](https://sites.google.com/view/video2events/home) for futher background papers.

+ Y. Hu, S-C. Liu, and T. Delbruck. v2e: From Video Frames to Realistic DVS Events. In _2021 IEEE/CVF  Conference  on  Computer  Vision  and  Pattern Recognition Workshops (CVPRW)_, URL: https://arxiv.org/abs/2006.07722, 2021

To reproduce the experiments of the paper, please find [this repository](https://github.com/SensorsINI/v2e_exps_public).

## Installation

If you don't want to install, try [opening v2e in google colab](https://colab.research.google.com/drive/1czx-GJnx-UkhFVBbfoACLVZs8cYlcr_M?usp=sharing).

### Advice about conversion time
We recommend running _v2e_ on a CUDA GPU or it will be very slow. 
With a low-end GTX-1050, _v2e_ runs about 50-200X slower than real time 
using 10X slowdown factor and 346x260 video.

Conversion speed depends linearly on the reciprocal of the desired DVS timestamp resolution.
If you demand fine resolution of e.g. 100us, 
then expect many minutes of computing per second of source video. Running on Google colab
with GPU, it took 500s per second of 12FPS source video, because of the very high upsampling ratio
of over 800X and the 220k frames that needed to be produced for DVS modeling.

We advise using the _--stop_ option for trial run before starting a long conversion.


### Make conda environment
You are encouraged to install v2e on a separate Python environment
such as `conda` environment:

```bash
conda create -n v2e python=3.9  # create a new environment
conda activate v2e  # activate the environment
```

### Install v2e
v2e works with Python 3.6 and above. To install v2e in developer mode (so your edits to source take effect immediately), run the following command in terminal:
```bash
git clone https://github.com/SensorsINI/v2e
cd v2e
python setup.py develop
```

+ For additional Windows GUI interface, you will need to install [Gooey](https://github.com/chriskiehl/Gooey) package. This package works the best on Windows:
    ```bash
    pip install Gooey
    ```
    On Linux, `Gooey` can be hard to install.

    For a sample of conversion using the gooey GUI, see https://youtu.be/THJqRC_q2kY


### Download SuperSloMo model

We use the excellent [Super SloMo](https://people.cs.umass.edu/~hzjiang/projects/superslomo/) framework to interpolate the APS frames.
However, since APS frames only record light intensity, we  retrained it on grayscale images.

Download our pre-trained model checkpoint from Google Drive
[[SuperSloMo39.ckpt]](https://drive.google.com/u/0/uc?id=17QSN207h05S_b2ndXjLrqPbBTnYIl0Vb&export=download) (151 MB) and save it to the _input_ folder. 
The default value of --slomo_model argument is set to this location.

### Download sample input data
The sample input videos to try _v2e_ with are in [v2e-sample-input-data](https://drive.google.com/drive/folders/1oWxTB9sMPp6UylAdxg5O1ko_GaIXz7wo?usp=sharing) on google drive.

Download the [tennis.mov](https://drive.google.com/file/d/1dNUXJGlpEM51UVYH4-ZInN9pf0bHGgT_/view?usp=sharing)
video and put in the _input_ folder
to run the example below.

## Usage

_v2e_ serves multiple purposes. Please read to code if you would like to adapt it for your own application. Here, we only introduce the usage for generating DVS events from conventional video and from specific datasets.

## Render emulated DVS events from conventional video.

_v2e.py_ reads a standard video (e.g. in .avi, .mp4, .mov, or .wmv), or a folder of images, and generates emulated DVS events at upsampled timestamp resolution.

Don't be intimidated by the huge number of options. Running _v2e.py_ with no arguments sets reasonable values and opens a file browser to let you select an input video. Inspect the logging output for hints.

**Hint:** Note the options _[--dvs128 | --dvs240 | --dvs346 | --dvs640 | --dvs1024]_; they set output size and width to popular DVS cameras.

**On headless platforms**, with no graphics output, use --no_preview option to suppress the OpenCV windows.

```
(base)$ conda activate v2e # activate your workspace
(v2e)$ v2e -h
usage: v2e.py [-h] [-o OUTPUT_FOLDER] [--avi_frame_rate AVI_FRAME_RATE]
              [--output_in_place [OUTPUT_IN_PLACE]] [--overwrite]
              [--unique_output_folder [UNIQUE_OUTPUT_FOLDER]]
              [--auto_timestamp_resolution [AUTO_TIMESTAMP_RESOLUTION]]
              [--timestamp_resolution TIMESTAMP_RESOLUTION]
              [--output_height OUTPUT_HEIGHT] [--output_width OUTPUT_WIDTH]
              [--dvs_params DVS_PARAMS] [--pos_thres POS_THRES]
              [--neg_thres NEG_THRES] [--sigma_thres SIGMA_THRES]
              [--cutoff_hz CUTOFF_HZ] [--leak_rate_hz LEAK_RATE_HZ]
              [--shot_noise_rate_hz SHOT_NOISE_RATE_HZ]
              [--leak_jitter_fraction LEAK_JITTER_FRACTION]
              [--noise_rate_cov_decades NOISE_RATE_COV_DECADES]
              [--refractory_period REFRACTORY_PERIOD]
              [--dvs_emulator_seed DVS_EMULATOR_SEED]
              [--show_dvs_model_state SHOW_DVS_MODEL_STATE]
              [--dvs128 | --dvs240 | --dvs346 | --dvs640 | --dvs1024]
              [--disable_slomo] [--slomo_model SLOMO_MODEL]
              [--batch_size BATCH_SIZE] [--vid_orig VID_ORIG]
              [--vid_slomo VID_SLOMO] [--slomo_stats_plot] [-i INPUT]
              [--input_frame_rate INPUT_FRAME_RATE]
              [--input_slowmotion_factor INPUT_SLOWMOTION_FACTOR]
              [--start_time START_TIME] [--stop_time STOP_TIME] [--crop CROP]
              [--synthetic_input SYNTHETIC_INPUT]
              [--dvs_exposure DVS_EXPOSURE [DVS_EXPOSURE ...]]
              [--dvs_vid DVS_VID] [--dvs_vid_full_scale DVS_VID_FULL_SCALE]
              [--skip_video_output] [--no_preview] [--davis_output]
              [--dvs_h5 DVS_H5] [--dvs_aedat2 DVS_AEDAT2]
              [--dvs_text DVS_TEXT]
v2e: generate simulated DVS events from video.
optional arguments:
  -h, --help            show this help message and exit
Output: General:
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        folder to store outputs.
  --avi_frame_rate AVI_FRAME_RATE
                        frame rate of output AVI video files; only affects
                        playback rate.
  --output_in_place [OUTPUT_IN_PLACE]
                        store output files in same folder as source video.
  --overwrite           overwrites files in existing folder (checks existence
                        of non-empty output_folder).
  --unique_output_folder [UNIQUE_OUTPUT_FOLDER]
                        If specifying --output_folder, makes unique output
                        folder based on output_folder, e.g. output1 (if non-
                        empty output_folder already exists)
DVS timestamp resolution:
  --auto_timestamp_resolution [AUTO_TIMESTAMP_RESOLUTION]
                        (Ignored by --disable_slomo.) If True (default),
                        upsampling_factor is automatically determined to limit
                        maximum movement between frames to 1 pixel. If False,
                        --timestamp_resolution sets the upsampling factor for
                        input video. Can be combined with
                        --timestamp_resolution to ensure DVS events have at
                        most some resolution.
  --timestamp_resolution TIMESTAMP_RESOLUTION
                        (Ignored by --disable_slomo.) Desired DVS timestamp
                        resolution in seconds; determines slow motion
                        upsampling factor; the video will be upsampled from
                        source fps to achieve the at least this timestamp
                        resolution.I.e. slowdown_factor =
                        (1/fps)/timestamp_resolution; using a high resolution
                        e.g. of 1ms will result in slow rendering since it
                        will force high upsampling ratio. Can be combind with
                        --auto_timestamp_resolution to limit upsampling to a
                        maximum limit value.
DVS model:
  --output_height OUTPUT_HEIGHT
                        Height of output DVS data in pixels. If None, same as
                        input video. Use --output_height=260 for Davis346.
  --output_width OUTPUT_WIDTH
                        Width of output DVS data in pixels. If None, same as
                        input video. Use --output_width=346 for Davis346.
  --dvs_params DVS_PARAMS
                        Easy optional setting of parameters for DVS
                        model:None, 'clean', 'noisy'; 'clean' turns off noise
                        and makes threshold variation zero. 'noisy' sets
                        limited bandwidth and adds leak events and shot
                        noise.This option by default will disable user set DVS
                        parameters. To use custom DVS paramters, use None
                        here.
  --pos_thres POS_THRES
                        threshold in log_e intensity change to trigger a
                        positive event.
  --neg_thres NEG_THRES
                        threshold in log_e intensity change to trigger a
                        negative event.
  --sigma_thres SIGMA_THRES
                        1-std deviation threshold variation in log_e intensity
                        change.
  --cutoff_hz CUTOFF_HZ
                        photoreceptor IIR lowpass filter cutoff-off 3dB
                        frequency in Hz - see
                        https://ieeexplore.ieee.org/document/4444573.CAUTION:
                        See interaction with timestamp_resolution and
                        auto_timestamp_resolution; check output logger
                        warnings.
  --leak_rate_hz LEAK_RATE_HZ
                        leak event rate per pixel in Hz - see
                        https://ieeexplore.ieee.org/abstract/document/7962235
  --shot_noise_rate_hz SHOT_NOISE_RATE_HZ
                        Temporal noise rate of ON+OFF events in darkest parts
                        of scene; reduced in brightest parts.
  --leak_jitter_fraction LEAK_JITTER_FRACTION
                        Jitter of leak noise events relative to the (FPN)
                        interval, drawn from normal distribution
  --noise_rate_cov_decades NOISE_RATE_COV_DECADES
                        Coefficient of Variation of noise rates (shot and
                        leak) in log normal distribution decades across pixel
                        arrayWARNING: currently only in leak events
  --refractory_period REFRACTORY_PERIOD
                        Refractory period in seconds, default is 0.5ms.The new
                        event will be ignore if the previous event is
                        triggered less than refractory_period ago.Set to 0 to
                        disable this feature.
  --dvs_emulator_seed DVS_EMULATOR_SEED
                        Set to a integer >0 to use a fixed random seed.default
                        is 0 which means the random seed is not fixed.
  --show_dvs_model_state SHOW_DVS_MODEL_STATE
                        one of new_frame baseLogFrame lpLogFrame0 lpLogFrame1
                        diff_frame (without quotes)
SloMo upsampling (see also "DVS timestamp resolution" group):
  --disable_slomo       Disables slomo interpolation; the output DVS events
                        will have exactly the timestamp resolution of the
                        source video (which is perhaps modified by
                        --input_slowmotion_factor).
  --slomo_model SLOMO_MODEL
                        path of slomo_model checkpoint.
  --batch_size BATCH_SIZE
                        Batch size in frames for SuperSloMo. Batch size 8-16
                        is recommended if your GPU has sufficient memory.
  --vid_orig VID_ORIG   Output src video at same rate as slomo video (with
                        duplicated frames).
  --vid_slomo VID_SLOMO
                        Output slomo of src video slowed down by
                        slowdown_factor.
  --slomo_stats_plot    show a plot of slomo statistics
Input file handling:
  -i INPUT, --input INPUT
                        Input video file or a image folder; leave empty for
                        file chooser dialog.If the input is a folder, the
                        folder should contain a ordered list of image files.In
                        addition, the user has to set the frame rate manually.
  --input_frame_rate INPUT_FRAME_RATE
                        Manually define the video frame rate when the video is
                        presented as a list of image files.When the input
                        video is a video file, this option will be ignored.
  --input_slowmotion_factor INPUT_SLOWMOTION_FACTOR
                        Sets the known slow-motion factor of the input video, 
                        i.e. how much the video is slowed down, i.e., 
                        the ratio of shooting frame rate to playback frame rate. 
                        input_slowmotion_factor<1 for sped-up video and 
                        input_slowmotion_factor>1 for slowmotion video.
                        If an input video is shot at 120fps yet is presented as a 30fps video 
                        (has specified playback frame rate of 30Hz, 
                        according to file's FPS setting), 
                        then set --input_slowdown_factor=4.
                        It means that each input frame represents (1/30)/4 s=(1/120)s.
                        If input is video with intended frame intervals of 
                        1ms that is in AVI file 
                        with default 30 FPS playback spec, 
                        then use ((1/30)s)*(1000Hz)=33.33333.
  --start_time START_TIME
                        Start at this time in seconds in video. Use None to
                        start at beginning of source video.
  --stop_time STOP_TIME
                        Stop at this time in seconds in video. Use None to end
                        at end of source video.
  --crop CROP           Crop input video by (left, right, top, bottom) pixels.
                        E.g. CROP=(100,100,0,0) crops 100 pixels from left and
                        right of input frames. CROP can also be specified as
                        L,R,T,B without ()
Synthetic input:
  --synthetic_input SYNTHETIC_INPUT
                        Input from class SYNTHETIC_INPUT that has methods
                        next_frame() and total_frames(). Disables file input
                        and SuperSloMo frame interpolation.
                        SYNTHETIC_INPUT.next_frame() should return a frame of
                        the correct resolution (see DVS model arguments) which
                        is array[y][x] with pixel [0][0] at upper left corner
                        and pixel values 0-255. SYNTHETIC_INPUT must be
                        resolvable from the classpath. SYNTHETIC_INPUT is the
                        module name without .py suffix. See example
                        moving_dot.py.
Output: DVS video:
  --dvs_exposure DVS_EXPOSURE [DVS_EXPOSURE ...]
                        Mode to finish DVS frame event integration:
                        	duration time: Use fixed accumulation time in seconds, e.g. 
                        		--dvs_exposure duration .005; 
                        	count n: Count n events per frame,e.g.
                        		-dvs_exposure count 5000;
                        	area_count M N: frame ends when any area of N x N pixels fills with M events, e.g.
                        		-dvs_exposure area_count 500 64
  --dvs_vid DVS_VID     Output DVS events as AVI video at frame_rate. To
                        suppress, supply argument None.
  --dvs_vid_full_scale DVS_VID_FULL_SCALE
                        Set full scale event count histogram count for DVS
                        videos to be this many ON or OFF events for full white
                        or black.
  --skip_video_output   Skip producing video outputs, including the original
                        video, SloMo video, and DVS video. This mode also
                        prevents showing preview of output (cf --no_preview).
  --no_preview          disable preview in cv2 windows for faster processing.
Output: DVS events:
  --davis_output        Save frames, frame timestamp and corresponding event
                        indexin HDF5. Default is False.
  --dvs_h5 DVS_H5       Output DVS events as hdf5 event database.
  --dvs_aedat2 DVS_AEDAT2
                        Output DVS events as DAVIS346 camera AEDAT-2.0 event
                        file for jAER; one file for real and one file for v2e
                        events. To suppress, supply argument None.
  --dvs_text DVS_TEXT   Output DVS events as text file with one event per line
                        [timestamp (float s), x, y, polarity (0,1)].
Run with no --input to open file dialog
```
You can put [tennis.mov](https://drive.google.com/file/d/1dNUXJGlpEM51UVYH4-ZInN9pf0bHGgT_/view?usp=sharing) in the _input_ folder to try it out with the command line below.  Or leave out all options and just use the file chooser to select the movie.

From root of v2e, run the following
```
python v2e.py -i input/tennis.mov --overwrite --timestamp_resolution=.003 --auto_timestamp_resolution=False --dvs_exposure duration 0.005 --output_folder=output/tennis --overwrite --pos_thres=.15 --neg_thres=.15 --sigma_thres=0.03 --dvs_aedat2 tennis.aedat --output_width=346 --output_height=260 --stop_time=3 --cutoff_hz=15 
```
Run the command above, and the following files will be created in a folder called _output/tennis_.

```
dvs-video.avi
dvs-video-frame_times.txt
tennis.aedat
v2e-args.txt
video_orig.avi
video_slomo.avi
```

* _dvs-video.avi_: DVS video (with playback rate 30Hz) but with frame rate (DVS timestamp resolution) set by source video frame rate times slowdown_factor.
* _dvs-video-frame_times.txt_: The times of the DVS frames. Useful when _--dvs_exposure count_ or _--dvs_exposure area-count_ methods are used.
* _tennis.aedat_: AEDAT-2.0 file for playback and algorithm experiments in [jAER](https://jaerproject.net) (use the AEChip _Davis346Blue_ to play this file.)
* _v2e-args.txt_: All the parameters and logging output from the run.
* _video_orig.avi_: input video, but converted to luma and resized to output (width,height) and with repeated frames to allow comparison to _slomo.avi_.
* _video_slomo.avi_: slow motion video (with playback rate 30Hz) but slowed down by slowdown_factor.

The [v2e site](https://sites.google.com/view/video2events/home) shows these videos.
## Scripting v2e
See the [scripts folder](https://github.com/SensorsINI/v2e/blob/master/scripts)  for various samples of shell and cmd scripts to run v2e from terminal.

### Synthetic input

There are also samples in the [scripts folder](https://github.com/SensorsINI/v2e/blob/master/scripts) of 
python modules to generate synthetic input to v2e, e.g. [particles.py](https://github.com/SensorsINI/v2e/blob/master/scripts/particles.py)

You can specify particles as the class that generates input frames to generate DVS events from using the command line option
````shell
v2e --synthetic_input scripts.particles ...
````
You synthetic input class should subclass _base_synthetic_class.py_. You should override the constructor and the _next_frame()_ method.

  * See [base_synthetic_input.py](https://github.com/SensorsINI/v2e/blob/master/scripts/base_synthetic_input.py) for more information.
  * You can pass command line arguments into your class; see [particles.py](https://github.com/SensorsINI/v2e/blob/master/scripts/particles.py) for example.

## Model parameters

The DVS ON and OFF threshold nominal values are set by _pos_thres_ and _neg_thres_. The pixel to pixel variation is set by _sigma_thres_. The pixel cutoff frequency in Hz is set by _cutoff_hz_. The leak event rate is set by _leak_rate_hz_. 

The _-dvs_params_ argument sets reasonable DVS model parameters for high and low light conditions.

See our technical paper for futher information about these parameters.
 
 ### Automatic vs manual DVS timestamp resolution
 The output DVS timestamps will be quantized to some value depending on options chosen.
 
  *  _--disable_slomo_ will disable slomo interpolation and the DVS events will have exactly the times of the input video, perhaps modified by --input_slowmotion_factor
  *  _--timestamp_resolution=X_ will upsample as needed to obtain this desired timestamp resolution _X_ in seconds. If auto_timestamp_resolution is set, then timestamp_resolution will still set the minimum timestamp resolution, i.e. if automatic timestamping would result in 5ms timestamps but timestamp_resolution is 1ms, then 1ms will still be the timestamp resolution.
  *  _--auto_timestamp_resolution_ will upsample in each _--batch_size_ frames using the computed optical flow to limit motion per frame to at most 1 pixel. In this case, turning on _--slomo_stats_plot_ will generate a plot like the following, which came from a driving video where the car sped up during part of the video:
 
 ![auto_slomo_stats](media/slomo_stats.png)
 
 This plot shows the actual timestamps of the interpolated frames (in orange) and the frame intervals for each batch of frames (in blue).
 
### Photoreceptor lowpass filtering
_v2e_ includes an intensity-dependent 1st-order lowpass filtering of light intensity; see the paper for details. 
If you set a nonzero --cutofffreq_hz, then it is important that the sample rate be high enough to allow the IIR lowpass filters to update properly, i.e.
the time constant tau of the lowpass filters must be at least 3 times larger than the frame interval.
Check the console output for warnings about undersampling for lowpass filtering.


 
 ### Frame rate and DVS timestamp resolution in v2e
There are several different 'frame rates' in v2e. On opening the input video, v2e reads the frame rate of the video and assumes the video is shot in real time, except that you can specify a _--input_slowmotion_factor_ slowdown_factor if the video is already a slow-motion video. The desired DVS timestamp resolution is combined with the source frame rate to compute the slow-motion upsampling factor. The output DVS AVI video is then generated using a _--dvs-exposure_ method.

 * _--avi_frame_rate_: Just sets the frame rate for playback of output AVI files
 * _--dvs-exposure_: See next section
 * _--input_slowmotion_factor_: Specifies by what factor the input video is slowed down.
 

## DVS frame exposure modes

The DVS allows arbritrary frame rates. _v2e_ provides 3 methods to 'expose' DVS video frames, which are selected by the
--dvs_exposure argument:
 1. **Constant-Duration**: _--dvs_exposure _duration_ _T__:  - Each frame has constant duration _T_.
 2. **Constant-Count**: _--dvs_exposure_count_ _N_:  - each frame has the same number _N_ of DVS events, as first described in Delbruck, Tobi. 2008. “Frame-Free Dynamic Digital Vision.” In Proceedings of Intl. Symp. on Secure-Life Electronics, Advanced Electronics for Quality Life and Society, 1:21–26. Tokyo, Japan: Tokyo. https://drive.google.com/open?id=0BzvXOhBHjRheTS1rSVlZN0l2MDg..
 3. **Area-Event**: _--dvs_exposure_ _area_event_ _N_ _M_:  - frames are accumulated until any block of *M*x*M* pixels fills up with _N_ events, as first described in Liu, Min, and T. Delbruck. 2018. “Adaptive Time-Slice Block-Matching Optical Flow Algorithm for Dynamic Vision Sensors.” In Proceedings of British Machine Vision Conference (BMVC 2018). Newcastle upon Tyne, UK: Proceedings of BMVC 2018. https://doi.org/10.5167/uzh-168589.

 - _Constant-Duration_ is like normal video, i.e. sampled at regular, ideally Nyquist rate. 
 - _Constant-Count_ frames have the same number of pixel brightness change events per frame. But if the scene is very textured (i.e. busy) then frame can get very brief, while parts of the input with only a small object moving can have very long frames.
 - _Area-Event_ compensates for this effect to some extent by concluding exposure when any block of pixels fills with a constant count.

## DAVIS camera conversion Dataset

v2e can convert recordings from
 [DDD20](https://sites.google.com/view/davis-driving-dataset-2020/home) and the original [DDD17](https://docs.google.com/document/d/1HM0CSmjO8nOpUeTvmPjopcBcVCk7KXvLUuiZFS6TWSg/pub) 
which are the first public end-to-end training datasets 
of automotive driving using a DAVIS event + frame camera. It lets you compare the real DVS data with the conversion. 
This dataset is maintained by the Sensors Research Group of Institute of Neuroinformatics. 

For your convenience, we offer via google drive one recording from _DDD20_ (our newer DDD dataset) of 800s of
Los Angeles street driving. 
The file is _aug04/rec1501902136.hdf5_ [[link]](https://drive.google.com/open?id=1KIaHsn72ZpVBZR6SGeFcd2lILyZoD2-5)
  in Google Drive for you to try it with v2e (***Warning:*** 2GB 7z compressed, 5.4 GB uncompressed).

```bash
mkdir -p input
mv rec1501902136.hdf5 ./input
```

**NOTE** you must run these scripts with the _-m package.script.py_ notation, not by directly pointing to the .py file.

### Extract data from DDD recording

_ddd_h5_extract_data.py_ extracts the DDD recording DVS events to jAER _.aedat_ and video _.avi_ files.

```
(pt-v2e) $ python -m dataset_scripts.ddd.ddd_extract_data.py -h
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
Running it from python console with
```python
runfile('E:\\Dropbox\\GitHub\\SensorsINI\\v2e\\ddd_extract_data.py', args=['--overwrite', '--output_folder=output/output-ddd-h5-data', '--overwrite', '--rotate180'], wdir='E:/Dropbox/GitHub/SensorsINI/v2e')
```
produces
```
output/output-ddd-h5-data/rec1501350986.aedat
output/output-ddd-h5-data/rec1501350986.avi
```
### Synthesize events from DDD recording

_ddd-v2e.py_ is like _v2e.py_ but it reads DDD .hdf5 recordings and extracts the real DVS events from the same part of the recording used for the synthesis of DVS events.

You can try it like this: 
```
$ python -m dataset_scripts.ddd.ddd-v2e.py --input input/rec1501350986.hdf5 --slomo_model input/SuperSloMo39.ckpt --slowdown_factor 20 --start 70 --stop 73 --output_folder output/ddd20-v2e-short --dvs_aedat dvs --pos_thres=.2 --neg_thres=.2 --overwrite --dvs_vid_full_scale=2 --frame_rate=100
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

...

INFO:src.ddd20_utils.ddd_h5_reader:input/rec1501350986.hdf5 has 38271 packets with start time 1123.34s and end time 1246.31s (duration    123.0s)
INFO:src.ddd20_utils.ddd_h5_reader:searching for time 70.0
ddd-h5-search:  56%|███████████████████████████████████████▉                                | 21244/38271 [00:08<00:05, 3002.45packet/s]INFO:src.ddd20_utils.ddd_h5_reader:
found start time 70.0 at packet 21369

...

:src.v2e_utils:opened output/ddd20-v2e-short/dvs-video-real.avi with  XVID https://www.fourcc.org/ codec, 30.0fps, and (346x260) size
v2e-ddd20:   5%|███▊                                                                              | 51/1100 [00:00<00:15, 67.70packet/s]INFO:src.slomo:loading SuperSloMo model from input/SuperSloMo39.ckpt

```

The generated outputs in folder _output/ddd20-v2e-short_ will be
```
dvs-v2e.aedat
dvs-v2e-real.aedat
dvs-video-fake.avi
dvs-video-real.avi
info.txt
original.avi
slomo.avi
```

## Working with jAER DAVIS recordings

DAVIS cameras like the one that recorded DDD17 and DDD20 are often used with [jAER](https://jaerproject.net) (although DDD recordings were made with custom python wrapper around caer). _v2e_ will output a jAER-compatible .aedat file in [AEDAT-2.0 format](https://inivation.com/support/software/fileformat/#aedat-20), which jAER uses.

To work with existing jAER DAVIS .aedat, you can export the DAVIS APS frames using the jAER EventFilter [DavisFrameAVIWriter](https://github.com/SensorsINI/jaer/blob/master/src/ch/unizh/ini/jaer/projects/davis/frames/DavisFrameAviWriter.java); see the [jAER user guide](https://docs.google.com/document/d/1fb7VA8tdoxuYqZfrPfT46_wiT1isQZwTHgX8O22dJ0Q/edit?usp=sharing), in particular, the [section about using DavisFrameAVIWriter](https://docs.google.com/document/d/1fb7VA8tdoxuYqZfrPfT46_wiT1isQZwTHgX8O22dJ0Q/edit#heading=h.g4cschniofmo). In DavisFrameAVIWriter, **don't forget to set the frameRate to the actual frame rate of the DAVIS frames** (which you can see at the top of the jAER display). This will make the conversion have approximately the correct DVS event timing. (jAER can drop APS frames if there are too many DVS events, so don't count on this.) Once you have the AVI from jAER, you can generate v2e events from it with _v2e.py_ and see how they compare with the original DVS events in jAER, by playing the exported v2e .aedat file in jAER.

An example of this conversion and comparison is on the [v2e home page](https://sites.google.com/view/video2events/home).

## Technical Details ##

See the [v2e home page](https://sites.google.com/view/video2events/home).
