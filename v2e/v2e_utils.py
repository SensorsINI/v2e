import logging
import os
import sys

import numpy as np
import cv2
import glob
import tkinter as tk
from tkinter import filedialog

DVS_WIDTH, DVS_HEIGHT = 346,260  # adjust for different sensor than DAVIS346

OUTPUT_VIDEO_FPS = 30.0 # playback frame rate specified for output video AVI file
# VIDEO_CODEC_FOURCC='RGBA' # uncompressed, >10MB for a few seconds of video
OUTPUT_VIDEO_CODEC_FOURCC= 'XVID' # good codec, basically mp4 with simplest compression, packed in AVI, only 15kB for a few seconds

logger=logging.getLogger(__name__)



def v2e_quit():
    try:
        quit() # not defined in pydev console, e.g. running in pycharm
    finally:
        sys.exit()


def check_lowpass(cutoffhz, fs, logger):
    import numpy as np
    from engineering_notation import EngNumber as eng
    if cutoffhz==0 or fs==0: return
    tau = 1 / (2 * np.pi * cutoffhz)
    dt = 1 / fs
    eps = dt / tau
    if eps>0.3:
        logger.warning(' Lowpass cutoff is {}Hz with sample rate {}Hz (sample interval {}ms),\nbut this results in tau={}ms and mixing factor eps={:5.3f},\n which means your lowpass will filter few or even 1 samples'.format(eng(cutoffhz), eng(fs), eng(dt*1000), eng(tau*1000), eps))
    else:
        logger.info(' Lowpass cutoff is {}Hz with sample rate {}Hz (sample interval {}ms),\nIt has tau={}ms and mixing factor eps={:5.3f}'.format(eng(cutoffhz), eng(fs), eng(dt*1000), eng(tau*1000), eps))


def inputVideoFileDialog():
    return _inputFileDialog([("Video/Data files", ".avi .mp4 .wmv"),('Any type','*')])

def inputDDDFileDialog():
    return _inputFileDialog([("DDD recordings", ".hdf5"),('Any type','*')])

def _inputFileDialog(types):
    from pathlib import Path
    root = tk.Tk()
    root.tk.call('tk', 'scaling', 4.0)  # doesn't help on hdpi screen
    root.withdraw()
    indir='./input'
    if Path(indir).is_dir():
        os.chdir(indir)
    filetypes=types
    filepath = filedialog.askopenfilename(filetypes=filetypes)
    os.chdir('..')
    return filepath

def checkAddSuffix(path:str,suffix:str):
    if path.endswith(suffix):
        return path
    else:
        return os.path.splitext(path)[0]+suffix

def video_writer(output_path, height, width):
    """ Return a video writer.

    Parameters
    ----------
    output_path: str,
        path to store output video.
    height: int,
        height of a frame.
    width: int,
        width of a frame.

    Returns
    -------
    an instance of cv2.VideoWriter.
    """

    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC_FOURCC)
    out = cv2.VideoWriter(
                output_path,
                fourcc,
                OUTPUT_VIDEO_FPS,
                (width, height))
    logger.debug('opened {} with  {} https://www.fourcc.org/ codec, {}fps, and ({}x{}) size'.format(output_path, OUTPUT_VIDEO_CODEC_FOURCC, OUTPUT_VIDEO_FPS, width, height))
    return out


def all_images(data_path:str):
    """Return path of all input images. Assume that the ascending order of
    file names is the same as the order of time sequence.

    Parameters
    ----------
    data_path: str
        path of the folder which contains input images.

    Returns
    -------
    List[str]
        sorted in numerical order.
    """
    images = glob.glob(os.path.join(data_path, '*.png'))
    if len(images) == 0:
        raise ValueError(("Input folder is empty or images are not in"
                          " 'png' format."))
    images_sorted = sorted(
        images,
        key=lambda line: int(line.split(os.sep)[-1].split('.')[0]))
    return images_sorted


def read_image(path: str) -> np.ndarray:
    """Read image and returns it as grayscale np.ndarray float scaled 0-255.

    Parameters
    ----------
    path: str
        path of image.

    Returns
    -------
    img: np.ndarray scaled 0-255
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float)
    return img


def read_aedat_txt_events(fname: str):
    """
    reads txt data DVS events
    Parameters
    ----------
    fname:str
        filename
    Returns
    -------
        np.ndarray with each row having ts,x,y,pol
        ts is in seconds
        pol is 0,1
    """
    import pandas as pd
    import numpy as np
    dat = pd.read_table(fname,
                        sep=' ',  # field separator
                        comment='#',  # comment
                        skipinitialspace=False,
                        skip_blank_lines=True,
                        error_bad_lines=False,
                        warn_bad_lines=True,
                        encoding='utf-8',
                        names=['t', 'x', 'y', 'p'],
                        dtype={'a': np.float64, 'b': np.int32, 'c': np.int32, 'd': np.int32}
                        )

    return np.array(dat.values) # array[N,4] with each row having ts, x, y, pol. ts is in float seconds. pol is 0,1


def select_events_in_roi(events, x, y):
    """ Select the events inside the region specified by x and y, including the x and y values.
    Parameters
    ----------
    events: np.ndarray, [timestamp, x, y, polarity]
    x: int or tuple, x coordinate.
    y: int or tuple, y coordinate.

    Returns
    -------
    np.ndarray, event just in ROI with the same shape as events.
    """
    x_lim = DVS_WIDTH-1 # events[:, 1].max()
    y_lim = DVS_HEIGHT-1 # events[:, 2].max()

    if isinstance(x, int):
        if x < 0 or x > x_lim:
            raise ValueError("x is not in the valid range.")
        x_region = (events[:, 1] == x)

    elif isinstance(x, tuple):
        if x[0] < 0 or x[1] < 0 or \
           x[0] > x_lim or x[1] > x_lim or \
           x[0] > x[1]:
            raise ValueError("x is not in the valid range.")
        x_region = np.logical_and(events[:, 1] >= x[0], events[:, 1] <= x[1])
    else:
        raise TypeError("x must be int or tuple.")

    if isinstance(y, int):
        if y < 0 or y > y_lim:
            raise ValueError("y is not in the valid range.")
        y_region = (events[:, 2] == y)

    elif isinstance(y, tuple):
        if y[0] < 0 or y[1] < 0 or \
           y[0] > y_lim or y[1] > y_lim or \
           y[0] > y[1]:
            raise ValueError("y is not in the valid range.")
        y_region = np.logical_and(events[:, 2] >= y[0], events[:, 2] <= y[1])
        raise TypeError("y must be int or tuple.")

    region = np.logical_and(x_region, y_region)

    return events[region]


def histogram_events_in_time_bins(events, start=0, stop=3.5, time_bin_ms=50, polarity=None):
    """ Count the amount of events in each bin.
    Parameters
    ----------
    events: np.ndarray, [timestamp, x, y, polarity].
    start: float, start time in s
    stop: float, end time in s
    polarity: int or None. If int, it must be 1 or -1.

    Returns
    -------
    histogram of counts

    """
    time_bin_s=time_bin_ms*0.001

    if start < 0 or stop < 0:
        raise ValueError("start and stop must be int.")
    if start + time_bin_s > stop:
        raise ValueError("start must be less than (stop - time_bin_s).")
    if polarity and polarity not in [1, -1]:
        raise ValueError("polarity must be 1 or -1.")

    ticks = np.arange(start, stop, time_bin_s)
    bin_num = ticks.shape[0]
    ts_cnt = np.zeros([bin_num - 1, 2])
    for i in range(bin_num - 1):
        condition = np.logical_and(events[:, 0] >= ticks[i],
                                   events[:, 0] < ticks[i + 1])
        if polarity:
            condition = np.logical_and(condition, events[:, 3] == polarity)
        cnt = events[condition].shape[0]
        ts_cnt[i][0] = (ticks[i] + ticks[i + 1]) / 2
        ts_cnt[i][1] = cnt

    return ts_cnt
