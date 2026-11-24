import logging
import os
import sys
import tempfile
import urllib.error
import urllib.request

import numpy as np
import cv2
import glob
import tkinter as tk
from tkinter import filedialog
from numba import njit
from engineering_notation import EngNumber as eng
from pathlib import Path

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
        if frame is None:
            logger.error(f'could not read a frame from file "{self.image_file_list[0]}" in folder "{self.image_folder_path}"')
            raise FileNotFoundError(f'could not read a frame named {self.image_file_list[0]} from folder {self.image_folder_path}')
        self.frame_height, self.frame_width = frame.shape[0], frame.shape[1]
        self.frame_channels = 1 if frame.ndim < 3 else frame.shape[2]

    def read(self, skip=False):
        """
        Reads the next frame

        :param skip: skip the frame

        :returns: always True
        """
        if not skip:
            frame = cv2.imread(self.image_file_list[self.current_frame_idx])
        else:
            frame=None
        self.current_frame_idx += 1

        # To match with OpenCV API
        return True, frame

    def release(self):
        """Just to match with OpenCV API."""
        pass

    def __str__(self):
        s=f'ImageFolderReader reading folder {self.image_folder_path} frame number {self.current_frame_idx}'
        try:
            s=s+f' named {self.image_file_list[self.current_frame_idx-1]}'
        except:
            pass
        return s


def normalize_dvs_params(value):
    """Return 'clean' or 'noisy', or None when the preset is off.

    Windows passes --dvs_params='noisy' with the quote marks still in the
    value, because single quotes are not argument quotes there. Strip one
    matching pair of quotes so that form selects the preset.
    """
    if value is None:
        return None
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1].strip()
    if text.lower() in ("", "none"):
        return None
    return text.lower()


def v2e_quit(code=0):
    try:
        quit(code)  # not defined in pydev console, e.g. running in pycharm
    finally:
        sys.exit(code)


def make_output_folder(output_folder_base, suffix_counter,overwrite, unique_output_folder) -> str:
    """Makes the output folder if it does not exist yet, or makes unique new numbered folder
    :param output_folder_base: the base name of folder. If it is absolute path, then make folder at absolute location, otherwise relative to startup folder
    :param suffix_counter: a counter value to append
    :param overwrite: to overwrite existing folder
    :param unique_output_folder: set True to make a new uniquely named numbered folder

    :returns: output folder path
    """
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
                      logger) -> str:
    """Set output folder in a single function.

    :param output_folder: path to folder, if supplied, otherwise None
    :param input_file: the input file to v2e, used for output_in_place. If folder, this folder is used.
    :param overwrite: set true to overwrite existing files in the folder
    :param output_in_place: set True to output in input_file or input folder folder
    :param logger: logger to report errors and warnings to

    :returns: the output folder path
    """

    if (not output_folder is None) and output_in_place:
        raise ValueError(f'both output_folder={output_folder} and output_in_place={output_in_place} cannot be set true at same time')

    if output_in_place:
        ip=Path(input_file)
        if ip.is_file():
            output_folder=ip.parent.absolute()
        elif ip.is_dir():
            output_folder=ip.absolute()
        logger.info(f'output_in_place==True so output_folder={output_folder}')
    else:
        output_folder = make_output_folder(
            output_folder, 0, overwrite, unique_output_folder)
        p=Path(output_folder)
        logger.info(
            f'output_in_place==False so made output_folder={p.absolute()}')

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

    if (output_width is None) or (output_height is None):
        logger.warning(
            "Either output_width is None or output_height is None,"
            "or both. Setting both of them to None. \n"
            "Dimension will be set automatically from video input if available. \n"
            "Check DVS camera size arguments.")
        output_width, output_height = None, None

    return output_width, output_height


# IIR mixing factor dt/tau above this updates the lowpass from too few samples.
LOWPASS_MAX_EPS = 0.3


def min_lowpass_sample_rate_hz(cutoff_hz: float) -> float:
    """Lowest sample rate that keeps dt/tau at LOWPASS_MAX_EPS.

    tau = 1/(2*pi*cutoff_hz), so fs >= (2*pi*cutoff_hz)/LOWPASS_MAX_EPS.
    Returns 0 when the lowpass is off.
    """
    if cutoff_hz <= 0:
        return 0.0
    return (2 * np.pi * cutoff_hz) / LOWPASS_MAX_EPS


def check_lowpass(cutoffhz, fs, logger):
    """ checks if cutoffhz is ok given sample rate fs

    """
    if cutoffhz == 0 or fs == 0:
        logger.info('lowpass filter is disabled, no need for check')
        return
    maxeps = LOWPASS_MAX_EPS
    tau = 1/(2*np.pi*cutoffhz)
    dt = 1/fs
    eps = dt/tau
    maxdt = tau*maxeps
    maxcutoff = maxeps/(2*np.pi*dt)
    min_fs = min_lowpass_sample_rate_hz(cutoffhz)
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
            '\n -decreasing --cutoff_hz below {}Hz'
            '\n -raising the source frame rate above {}Hz'.format(
                eng(cutoffhz), eng(tau), eng(fs), eng(dt), eps,
                maxeps, eng(maxdt), eng(maxcutoff), eng(min_fs)))
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


def _checkout_root():
    """Repository root when this v2e is the git checkout, not an installed copy.

    A checkout has v2ecore next to v2e.py and .git. An installed copy lives
    under site-packages and has no .git directory there.
    """
    root = Path(__file__).resolve().parent.parent
    if (root / '.git').exists() and (root / 'v2e.py').is_file():
        return str(root)
    return None


SLOMO_CHECKPOINT_NAME = "SuperSloMo39.ckpt"
SLOMO_DRIVE_ID = "1ETID_4xqLpRBrRo1aOT7Yphs3QqWR_fx"
SLOMO_DRIVE_PAGE = (
    "https://drive.google.com/file/d/1ETID_4xqLpRBrRo1aOT7Yphs3QqWR_fx/view")
# confirm=t skips the large-file interstitial. The second URL is the older
# export endpoint, used when the first one returns a warning page.
_SLOMO_DOWNLOAD_URLS = (
    "https://drive.usercontent.google.com/download"
    f"?id={SLOMO_DRIVE_ID}&export=download&confirm=t",
    "https://docs.google.com/uc"
    f"?export=download&id={SLOMO_DRIVE_ID}&confirm=t",
)
# The published checkpoint is 151 MB. A smaller body is the HTML warning page.
_SLOMO_MIN_BYTES = 50 * 1024 * 1024


def default_slomo_directory():
    """Folder that holds SuperSloMo39.ckpt, independent of the process cwd.

    A git checkout, or a shell whose working directory is that checkout, uses
    ``input``. An installed copy uses a user data folder so the same file is
    found when v2e is started from any directory.
    """
    checkout = _checkout_root()
    if checkout:
        return Path(checkout) / "input"
    cwd = Path.cwd()
    if (cwd / ".git").exists() and (cwd / "v2e.py").is_file():
        return cwd / "input"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            base = str(Path.home() / "AppData" / "Local")
        return Path(base) / "v2e" / "input"
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        return Path(data_home) / "v2e" / "input"
    return Path.home() / ".v2e" / "input"


def default_slomo_checkpoint():
    """Absolute path of the SuperSloMo checkpoint v2e loads by default."""
    return str((default_slomo_directory() / SLOMO_CHECKPOINT_NAME).resolve())


def ensure_slomo_checkpoint(path=None):
    """Return a readable checkpoint path, downloading the default if needed.

    A path the user chose is returned as-is when the file exists, and raises
    ``FileNotFoundError`` when it does not. The default path is downloaded
    into :func:`default_slomo_directory` when that file is not there yet.
    """
    if path is None or path == "":
        path = default_slomo_checkpoint()
    path = Path(os.path.abspath(os.path.expanduser(str(path))))
    if path.is_file():
        return str(path)
    default = Path(default_slomo_checkpoint())
    if os.path.normcase(path) != os.path.normcase(default):
        raise FileNotFoundError(
            "SuperSloMo model checkpoint " + str(path) +
            " does not exist or is not readable")
    _download_slomo_checkpoint(default)
    return str(default)


def _slomo_download_failed(dest, reason):
    raise FileNotFoundError(
        f"Could not download {SLOMO_CHECKPOINT_NAME} to {dest} ({reason}). "
        f"Download {SLOMO_DRIVE_PAGE} and save it as {dest}")


def _download_slomo_checkpoint(dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    logger.info(
        "SuperSloMo checkpoint not found. Downloading %s to %s",
        SLOMO_CHECKPOINT_NAME, dest)
    last_reason = "no download URL could be reached"
    for url in _SLOMO_DOWNLOAD_URLS:
        try:
            _download_slomo_once(url, partial)
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            last_reason = str(exc)
            partial.unlink(missing_ok=True)
            logger.warning("SuperSloMo download failed: %s", last_reason)
            continue
        if partial.stat().st_size < _SLOMO_MIN_BYTES:
            last_reason = (
                f"file is only {partial.stat().st_size} bytes; "
                "expected the 151 MB checkpoint")
            partial.unlink(missing_ok=True)
            continue
        partial.replace(dest)
        logger.info(
            "saved SuperSloMo checkpoint %s (%d bytes)",
            dest, dest.stat().st_size)
        return
    _slomo_download_failed(dest, last_reason)


def _download_slomo_once(url, partial: Path):
    from tqdm import tqdm

    request = urllib.request.Request(url, headers={"User-Agent": "v2e"})
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "text/html" in content_type:
            raise RuntimeError("server returned an HTML page")
        with open(partial, "wb") as out, tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=SLOMO_CHECKPOINT_NAME) as bar:
            first = response.read(1024 * 1024)
            if not _slomo_bytes_look_like_checkpoint(first):
                raise RuntimeError("server returned a web page, not the checkpoint")
            out.write(first)
            bar.update(len(first))
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))


def _slomo_bytes_look_like_checkpoint(blob: bytes) -> bool:
    """PyTorch checkpoints start with a pickle protocol byte or a zip header."""
    if not blob:
        return False
    if blob.lstrip()[:1] == b"<":
        return False
    return blob[:1] == b"\x80" or blob[:2] == b"PK"


def _foreground_file_dialog(title, filetypes, initialdir, initialfile):
    """Open a file chooser above the terminal.

    easygui's chooser uses a withdrawn Tk root, and Windows then leaves the
    dialog behind the console that launched v2e. A topmost parent keeps the
    native dialog in front for as long as it is open.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.lift()
    root.focus_force()
    root.update()
    if sys.platform == 'win32':
        import ctypes
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        user32.AllowSetForegroundWindow(-1)
        foreground = user32.GetForegroundWindow()
        fg_thread = user32.GetWindowThreadProcessId(foreground, None)
        this_thread = kernel32.GetCurrentThreadId()
        attached = False
        if fg_thread and fg_thread != this_thread:
            attached = bool(user32.AttachThreadInput(fg_thread, this_thread, True))
        user32.SetForegroundWindow(int(root.winfo_id()))
        if attached:
            user32.AttachThreadInput(fg_thread, this_thread, False)
    options = dict(parent=root, title=title, filetypes=filetypes)
    if initialdir:
        options['initialdir'] = os.path.normpath(initialdir)
    if initialfile:
        options['initialfile'] = initialfile
    # Windows remembers the last folder for this dialog and otherwise opens
    # Documents. Starting in initialdir makes that folder win.
    previous = os.getcwd()
    try:
        if initialdir and os.path.isdir(initialdir):
            os.chdir(initialdir)
        return filedialog.askopenfilename(**options)
    finally:
        os.chdir(previous)
        root.destroy()


def _inputFileDialog(types):
    LAST_FILE_NAME_FILE = 'v2e_last_file_chosen.txt'
    fn = os.path.join(tempfile.gettempdir(), LAST_FILE_NAME_FILE)
    default = None
    try:
        with open(fn, 'r') as f:
            default = f.read()
    except FileNotFoundError:
        pass
    initialdir = None
    initialfile = None
    if default:
        folder = os.path.dirname(default)
        if folder and os.path.isdir(folder):
            initialdir = folder
        if os.path.isfile(default):
            initialfile = os.path.basename(default)
    if initialdir is None:
        initialdir = _checkout_root()
    filetypes = []
    for description, pattern in types:
        masks = []
        for token in pattern.split():
            if token in ('*', '*.*'):
                masks.append('*.*')
            elif token.startswith('*'):
                masks.append(token)
            elif token.startswith('.'):
                masks.append('*' + token)
            else:
                masks.append(token)
        filetypes.append((description, ' '.join(masks)))
    if initialdir:
        logger.info(f'Select the input file in the dialog ({initialdir})')
    else:
        logger.info('Select the input file in the dialog')
    filename = _foreground_file_dialog(
        'v2e: Select file to convert', filetypes, initialdir, initialfile)
    if not filename:
        logger.info('no file selected, quitting')
        quit(0)
    logger.info(f'selected {filename} with file dialog')
    try:
        with open(fn, 'w') as f:
            f.write(filename)
    except:
        pass
    return filename


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
    logger.info(
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
