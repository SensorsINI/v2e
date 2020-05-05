import numpy as np
import cv2
import glob
import os
import logging
import tkinter as tk
from tkinter import filedialog

DVS_WIDTH, DVS_HEIGHT = 346,260  # adjust for different sensor than DAVIS346

OUTPUT_VIDEO_FPS = 30.0 # playback frame rate specified for output video AVI file
# VIDEO_CODEC_FOURCC='RGBA' # uncompressed, >10MB for a few seconds of video
OUTPUT_VIDEO_CODEC_FOURCC= 'XVID' # good codec, basically mp4 with simplest compression, packed in AVI, only 15kB for a few seconds

logger=logging.getLogger(__name__)

def v2e_args(parser):
    dir_path = os.getcwd()  # check and add prefix if running script in subfolder
    if dir_path.endswith('ddd'):
        prepend='../../'
    else:
        prepend=''
    parser.add_argument("-i", "--input", type=str, help="input video file; leave empty for file chooser dialog.")
    parser.add_argument("--start_time", type=float, default=None, help="start at this time in seconds in video.")
    parser.add_argument("--stop_time", type=float, default=None, help="stop at this time in seconds in video.")
    parser.add_argument("--pos_thres", type=float, default=0.21,
                        help="threshold in log_e intensity change to trigger a positive event.")
    parser.add_argument("--neg_thres", type=float, default=0.17,
                        help="threshold in log_e intensity change to trigger a negative event.")
    parser.add_argument("--sigma_thres", type=float, default=0.03,
                        help="1-std deviation threshold variation in log_e intensity change.")
    parser.add_argument("--cutoff_hz", type=float, default=300,
                        help="photoreceptor second-order IIR lowpass filter cutoff-off 3dB frequency in Hz - see https://ieeexplore.ieee.org/document/4444573")
    parser.add_argument("--leak_rate_hz", type=float, default=0.05,
                        help="leak event rate per pixel in Hz - see https://ieeexplore.ieee.org/abstract/document/7962235")
    parser.add_argument("--shot_noise_rate_hz", type=float, default=0,
                        help="Temporal noise rate of ON+OFF events in darkest parts of scene; reduced in brightest parts. ")
    parser.add_argument("--slowdown_factor", type=int, default=10,
                        help="slow motion factor; if the input video has frame rate fps, then the DVS events will have time resolution of 1/(fps*slowdown_factor).")
    parser.add_argument("--output_height", type=int, default=260,
                        help="height of output DVS data in pixels. If None, same as input video.")
    parser.add_argument("--output_width", type=int, default=346,
                        help="width of output DVS data in pixels. If None, same as input video.")
    parser.add_argument("--slomo_model", type=str, default=prepend+"input/SuperSloMo39.ckpt", help="path of slomo_model checkpoint.")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="folder to store outputs.")
    parser.add_argument("--frame_rate", type=int, default=300,
                        help="equivalent frame rate of --dvs_vid output video; the events will be accummulated as this sample rate; DVS frames will be accumulated for duration 1/frame_rate")
    parser.add_argument("--dvs_vid", type=str, default="dvs-video.avi", help="output DVS events as AVI video at frame_rate.")
    parser.add_argument("--dvs_vid_full_scale", type=int, default=3, help="set full scale count for DVS videos to be this many ON or OFF events.")
    parser.add_argument("--dvs_h5", type=str, default=None, help="output DVS events as hdf5 event database.")
    parser.add_argument("--dvs_aedat2", type=str, default=None, help="output DVS events as DAVIS346 camera AEDAT-2.0 event file for jAER; one file for real and one file for v2e events.")
    parser.add_argument("--dvs_text", type=str, default=None, help="output DVS events as text file with one event per line [timestamp (float s), x, y, polarity (0,1)].")
    parser.add_argument("--dvs_numpy", type=str, default=None, help="accumulates DVS events to memory and writes final numpy data file with this name holding vector of events. WARNING: memory use is unbounded.")
    parser.add_argument("--vid_orig", type=str, default="video_orig.avi", help="output src video at same rate as slomo video (with duplicated frames).")
    parser.add_argument("--vid_slomo", type=str, default="video_slomo.avi", help="output slomo of src video slowed down by slowdown_factor.")
    parser.add_argument("--no_preview", action="store_true", help="disable preview in cv2 windows for faster processing.")
    parser.add_argument("--overwrite", action="store_true", help="overwrites files in existing folder (checks existence of non-empty output_folder).")

    # # perform basic checks, however this fails if script adds more arguments later
    # args = parser.parse_args()
    # if args.input and not os.path.isfile(args.input):
    #     logger.error('input file {} not found'.format(args.input))
    #     quit(1)
    # if args.slomo_model and not os.path.isfile(args.slomo_model):
    #     logger.error('slomo model checkpoint {} not found'.format(args.slomo_model))
    #     quit(1)

    return parser

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
    root = tk.Tk()
    root.tk.call('tk', 'scaling', 4.0)  # doesn't help on hdpi screen
    root.withdraw()
    os.chdir('./input')
    filetypes=types
    filepath = filedialog.askopenfilename(filetypes=filetypes)
    os.chdir('..')
    return filepath

def checkAddSuffix(path:str,suffix:str):
    if path.endswith(suffix):
        return path
    else:
        return path+suffix

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
    else:
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