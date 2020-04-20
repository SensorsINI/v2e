import numpy as np
import cv2
import glob
import os
import logging

OUTPUT_VIDEO_FPS = 30.0 # playback frame rate specified for output video AVI file
# VIDEO_CODEC_FOURCC='RGBA' # uncompressed, >10MB for a few seconds of video
OUTPUT_VIDEO_CODEC_FOURCC= 'XVID' # good codec, basically mp4 with simplest compression, packed in AVI, only 15kB for a few seconds

logger=logging.getLogger(__name__)

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
