import numpy as np
import cv2
import glob
import os
import logging
import tkinter as tk
from tkinter import filedialog

OUTPUT_VIDEO_FPS = 30.0 # playback frame rate specified for output video AVI file
# VIDEO_CODEC_FOURCC='RGBA' # uncompressed, >10MB for a few seconds of video
OUTPUT_VIDEO_CODEC_FOURCC= 'XVID' # good codec, basically mp4 with simplest compression, packed in AVI, only 15kB for a few seconds

logger=logging.getLogger(__name__)

def v2e_args(parser):
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
                        help="photoreceptor first order IIR lowpass cutoff-off 3dB frequency in Hz - see https://ieeexplore.ieee.org/document/4444573")
    parser.add_argument("--leak_rate_hz", type=float, default=0.05,
                        help="leak event rate per pixel in Hz - see https://ieeexplore.ieee.org/abstract/document/7962235")
    parser.add_argument("--slowdown_factor", type=int, default=10,
                        help="slow motion factor; if the input video has frame rate fps, then the DVS events will have time resolution of 1/(fps*slowdown_factor).")
    parser.add_argument("--output_height", type=int, default=260,
                        help="height of output DVS data in pixels. If None, same as input video.")
    parser.add_argument("--output_width", type=int, default=346,
                        help="width of output DVS data in pixels. If None, same as input video.")
    parser.add_argument("--slomo_model", type=str, default="input/SuperSloMo39.ckpt", help="path of slomo_model checkpoint.")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="folder to store outputs.")
    parser.add_argument("--frame_rate", type=int, default=300,
                        help="equivalent frame rate of --dvs_vid output video; the events will be accummulated as this sample rate; DVS frames will be accumulated for duration 1/frame_rate")
    parser.add_argument("--dvs_vid", type=str, default="dvs-video.avi", help="output DVS events as AVI video at frame_rate.")
    parser.add_argument("--dvs_vid_full_scale", type=int, default=3, help="set full scale count for DVS videos to be this many ON or OFF events.")
    parser.add_argument("--dvs_h5", type=str, default=None, help="output DVS events as hdf5 event database.")
    parser.add_argument("--dvs_aedat2", type=str, default=None, help="output DVS events as DAVIS346 camera AEDAT-2.0 event file for jAER; one file for real and one file for v2e events.")
    parser.add_argument("--dvs_text", type=str, default=None, help="output DVS events as text file with one event per line [timestamp (float s), x, y, polarity (0,1)].")
    parser.add_argument("--vid_orig", type=str, default="video_orig.avi", help="output src video at same rate as slomo video (with duplicated frames).")
    parser.add_argument("--vid_slomo", type=str, default="video_slomo.avi", help="output slomo of src video slowed down by slowdown_factor.")
    parser.add_argument("--no_preview", action="store_false", help="disable preview in cv2 windows for faster processing.")
    parser.add_argument("--overwrite", action="store_true", help="overwrites files in existing folder (checks existence of non-empty output_folder).")
    return parser


def inputFileDialog():
    root = tk.Tk()
    root.tk.call('tk', 'scaling', 4.0)  # doesn't help on hdpi screen
    root.withdraw()
    os.chdir('./input')
    filetypes=[("Video/Data files", ".avi .mp4 .wmv .hdf5"),('Any type','*')]
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
