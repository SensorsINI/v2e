# frame2dvs [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python code for converting APS frames with low frame rate into DVS frames with high frame rate.

## Contact
Zhe He (zhhe@ini.uzh.ch)
Yuhuang Hu (yuhuang.hu@ini.uzh.ch)
Tobi Delbruck (tobi@ini.uzh.ch)

## Environment

```bash
python==3.7.3
```

We highly recommend running the code in virtual environment. Miniconda is always your best friend. :)

## Install Dependencies

```bash
pip install -r requirements.txt
```

The packages listed below will be installed.
```bash
h5py==2.9.0
numpy==1.16.2
opencv-python==4.1.0.25
Pillow==5.4.1
torch==1.1.0
torchvision==0.2.1
tqdm==4.31.1
```

## Usage

The program is designed to serve multiple purposes. Please read to code if you would like to adapt it for your own application. Here, we only introduce the usage for extracting DVS events from APS frames.

## Dataset

DDD17 is the first public end-to-end training dataset of automotive driving using a DAVIS event + frame camera. This dataset is maintained by the Sensors Research Group of Institute of Neuroinformatics. Please read the document [[link](https://docs.google.com/document/d/1HM0CSmjO8nOpUeTvmPjopcBcVCk7KXvLUuiZFS6TWSg/pub)] for details about downloading the dataset.

## Download Checkpoint

We used the [Super SloMo](https://people.cs.umass.edu/~hzjiang/projects/superslomo/) framework to interpolate the APS frames. However, since APS frames only record illumination indensity, the model needs to be trained on grayscale images. You can download our pre-trained model from Google Drive [[link](https://drive.google.com/file/d/17QSN207h05S_b2ndXjLrqPbBTnYIl0Vb/view?usp=sharing)].

```bash
mkdir data
mv SuperSloMo39.ckpt ./data
```

## Render DVS Frames

```bash
cd test
python renderer_test.py \
--start [start] \
--stop [end] \
--fname [path to the .hdf5 DVS recording file]] \
--checkpoint [the .ckpt checkpoint of the slow motion network]] \
--pos_thres [threshold of triggering a positive event] \
--nega_thres [threshold of triggering a negative event] \
--sf [slow motion factor]
```

Run the commands above, then two videos 'from_images.avi' and 'from_events.avi' will be created under './data/'

If you don't like the file names, you could change them in 'renderer_test.py'.

## Calibrate the Threshold

To get the threshold of triggering an event, you need to run the commands below,

```bash
cd test
python renderer_sweep.py \
--start [start] \
--stop [end] \
--fname [path to the .hdf5 DVS recording file]] \
--checkpoint [the .ckpt checkpoint of the slow motion network]] \
--sf [slow motion factor]
```

The program will take the DVS recording data, which starts at time 'start' and ends at time 'end', to calculate the best threshold values for positive events and negative events separately.

In order to get the best approximations, the input video needs to satisfy the requirements below,

- Daytime
- Cloudy
- No rain
- High frame rate
