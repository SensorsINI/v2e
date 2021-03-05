import logging
import os
import sys

import numpy as np
import cv2
import glob
import tkinter as tk
from tkinter import filedialog
from numba import njit
from engineering_notation import EngNumber as eng

# adjust for different sensor than DAVIS346
DVS_WIDTH, DVS_HEIGHT = 346, 260

# VIDEO_CODEC_FOURCC='RGBA' # uncompressed, >10MB for a few seconds of video

# good codec, basically mp4 with simplest compression, packed in AVI,
# only 15kB for a few seconds
OUTPUT_VIDEO_CODEC_FOURCC = 'XVID'
logger = logging.getLogger(__name__)


class ImageFolderReader(object):
    def __init__(self, image_folder_path, frame_rate):
        """ImageFolderReader.

        This class implements functions that are similar to
        VideoCapture in OpenCV.

        This class is used when the frames are available as
        a sequence of image files in a folder.

        NOTE: the folder should contain only image files and
        the files have to be ordered!!!

        All images are assumed to have the same dimension.
        """
        self.image_folder_path = image_folder_path

        self.image_file_list = sorted(
            glob.glob("{}".format(self.image_folder_path) + "/*.*"))

        self.frame_rate = frame_rate

        self.current_frame_idx = 0

        self.num_frames = len(self.image_file_list)

        frame = cv2.imread(self.image_file_list[0])
        self.frame_height, self.frame_width = frame.shape[0], frame.shape[1]
        self.frame_channels = 1 if frame.ndim < 3 else frame.shape[2]

    def read(self):
        frame = cv2.imread(self.image_file_list[self.current_frame_idx])
        self.current_frame_idx += 1

        # To match with OpenCV API
        return True, frame

    def release(self):
        """Just to match with OpenCV API."""
        pass


def v2e_quit(code=None):
    try:
        quit(code)  # not defined in pydev console, e.g. running in pycharm
    finally:
        sys.exit()


def make_output_folder(output_folder_base, suffix_counter,
                       overwrite, unique_output_folder):
    if overwrite and unique_output_folder:
        logger.error(
            "specify one or the other of "
            "--overwrite and --unique_output_folder")
        v2e_quit()

    output_folder = output_folder_base+"-{}".format(suffix_counter) \
        if suffix_counter > 0 else output_folder_base

    non_empty_folder_exists = not overwrite and \
        os.path.exists(output_folder) and os.listdir(output_folder)

    if non_empty_folder_exists and not overwrite and not unique_output_folder:
        logger.error(
            'non-empty output folder {} already exists \n '
            '- use --overwrite or --unique_output_folder'.format(
                os.path.abspath(output_folder)))
        v2e_quit()

    if non_empty_folder_exists and unique_output_folder:
        return make_output_folder(
            output_folder_base, suffix_counter+1,
            overwrite, unique_output_folder)
    else:
        logger.info('using output folder {}'.format(output_folder))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        return output_folder


def set_output_folder(output_folder,
                      input_file,
                      unique_output_folder,
                      overwrite,
                      output_in_place,
                      logger):
    """Set output folder in a single function."""

    if output_in_place:
        parts = os.path.split(input_file)
        output_folder = parts[0]
        logger.info(f'output_in_place==True so output_folder={output_folder}')
    else:
        output_folder = make_output_folder(
            output_folder, 0, overwrite, unique_output_folder)
        logger.info(
            f'output_in_place==False so made output_folder={output_folder}')

    return output_folder


def set_output_dimension(output_width, output_height,
                         dvs128, dvs240, dvs346, dvs640, dvs1024,
                         logger):
    """Return output_height and output_width based on arguments."""

    if dvs128:
        output_width, output_height = 128, 128
    elif dvs240:
        output_width, output_height = 240, 180
    elif dvs346:
        output_width, output_height = 346, 260
    elif dvs640:
        output_width, output_height = 640, 480
    elif dvs1024:
        output_width, output_height = 1024, 768

    if (output_width is None) ^ (output_height is None):
        logger.error('set neither or both of output_width and output_height')
        v2e_quit()

    return output_width, output_height


def check_lowpass(cutoffhz, fs, logger):
    """ checks if cutoffhz is ok given sample rate fs

    """
    if cutoffhz == 0 or fs == 0:
        logger.info('lowpass filter is disabled, no need for check')
        return
    maxeps = 0.3
    tau = 1/(2*np.pi*cutoffhz)
    dt = 1/fs
    eps = dt/tau
    maxdt = tau*maxeps
    maxcutoff = maxeps/(2*np.pi*dt)
    if eps > maxeps:
        logger.warning(
            'Lowpass 3dB cutoff is f_3dB={}Hz (time constant tau={}s) with '
            'sample rate fs={}Hz (sample interval dt={}s) '
            ',\n  but this results in large IIR mixing factor '
            'eps = dt/tau = {:5.3f} > {:4.1f} (maxeps),'
            '\n which means the lowpass will filter  few or even just '
            'last sample, i.e. you will not be lowpassing as expected.'
            '\nWe recommend either'
            '\n -decreasing --timestamp_resolution of DVS events below {}s'
            '\n -decreasing --cutoff_frequency_hz below {}Hz'.format(
                eng(cutoffhz), eng(tau), eng(fs), eng(dt), eps,
                maxeps, eng(maxdt), eng(maxcutoff)))
    else:
        logger.info(
            'Lowpass cutoff is f_3dB={}Hz with tau={}s and '
            'with sample rate fs={}Hz (sample interval dt={}s)'
            ',\nIt has IIR mixing factor eps={:5.3f} which is OK '
            'because it is less than recommended maxeps={:4.1f}'.format(
                eng(cutoffhz), eng(tau), eng(fs), eng(dt), eps, maxeps))


def inputVideoFileDialog():
    return _inputFileDialog(
        [("Video/Data files", ".avi .mp4 .wmv"), ('Any type', '*')])


def inputDDDFileDialog():
    return _inputFileDialog([("DDD recordings", ".hdf5"), ('Any type', '*')])


def _inputFileDialog(types):
    from pathlib import Path
    root = tk.Tk()
    root.tk.call('tk', 'scaling', 4.0)  # doesn't help on hdpi screen
    root.withdraw()
    indir = './input'
    if Path(indir).is_dir():
        os.chdir(indir)
    filetypes = types
    filepath = filedialog.askopenfilename(filetypes=filetypes)
    os.chdir('..')
    return filepath


def checkAddSuffix(path: str, suffix: str):
    if path.endswith(suffix):
        return path
    else:
        return os.path.splitext(path)[0]+suffix


def video_writer(output_path, height, width,
                 frame_rate=30, fourcc=OUTPUT_VIDEO_CODEC_FOURCC):
    """ Return a video writer.

    Parameters
    ----------
    output_path: str,
        path to store output video.
    height: int,
        height of a frame.
    width: int,
        width of a frame.
    frame_rate: int
        playback frame rate in Hz
    fourcc: cv2.VideoWriter_fourcc
        codec, None results in default XVID
    Returns
    -------
    an instance of cv2.VideoWriter.
    """
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    out = cv2.VideoWriter(
                output_path,
                fourcc,
                frame_rate,
                (width, height))
    logger.debug(
        'opened {} with {} https://www.fourcc.org/ codec, {}fps, '
        'and ({}x{}) size'.format(
            output_path, OUTPUT_VIDEO_CODEC_FOURCC, frame_rate,
            width, height))
    return out


def all_images(data_path):
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
    img = img.astype(np.float32)
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
    dat = pd.read_table(
        fname,
        sep=' ',  # field separator
        comment='#',  # comment
        skipinitialspace=False,
        skip_blank_lines=True,
        error_bad_lines=False,
        warn_bad_lines=True,
        encoding='utf-8',
        names=['t', 'x', 'y', 'p'],
        dtype={'a': np.float64, 'b': np.int32, 'c': np.int32, 'd': np.int32})

    # array[N,4] with each row having ts, x, y, pol.
    # ts is in float seconds. pol is 0,1
    return np.array(dat.values)


def select_events_in_roi(events, x, y):
    """ Select the events inside the region specified by x and y.
    including the x and y values.

    Parameters
    ----------
    events: np.ndarray, [timestamp, x, y, polarity]
    x: int or tuple, x coordinate.
    y: int or tuple, y coordinate.

    Returns
    -------
    np.ndarray, event just in ROI with the same shape as events.
    """
    x_lim = DVS_WIDTH-1  # events[:, 1].max()
    y_lim = DVS_HEIGHT-1  # events[:, 2].max()

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
    else:
        raise TypeError("y must be int or tuple.")

    region = np.logical_and(x_region, y_region)

    return events[region]


def histogram_events_in_time_bins(
        events, start=0, stop=3.5,
        time_bin_ms=50, polarity=None):
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
    time_bin_s = time_bin_ms*0.001

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


@njit("float64[:, :](float64[:, :], int64[:], int64[:, :])",
      nogil=True, parallel=False)
def hist2d_numba_seq(tracks, bins, ranges):
    H = np.zeros((bins[0], bins[1]), dtype=np.float64)
    delta = 1/((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(tracks.shape[1]):
        i = (tracks[0, t] - ranges[0, 0]) * delta[0]
        j = (tracks[1, t] - ranges[1, 0]) * delta[1]
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            H[int(i), int(j)] += 1

    return H
