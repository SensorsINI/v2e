import logging
import os
import sys

import numpy as np
import argparse
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("TkAgg") # use in pycharm to avoid scientific mode plot?
from tqdm import tqdm

from v2e import desktop
from v2e.emulator import EventEmulator
from v2e.slomo import SuperSloMo
from v2e.ddd20_utils.ddd_h5_reader import DDD20ReaderMultiProcessing, DDD20SimpleReader
from v2e.v2e_utils import inputVideoFileDialog, inputDDDFileDialog, select_events_in_roi, DVS_WIDTH, DVS_HEIGHT
from v2e.v2e_args import write_args_info

logging.basicConfig()
root = logging.getLogger()
root.setLevel(logging.INFO)
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='ddd_find_thresholds.py: generate simulated DVS events from video with sweep of thresholds to compare with real DVS to find optimal thresholds.',
                                 epilog='Run with no --input to open file dialog.\nIf slomo.avi aleady exists in output_folder, script will load frames from there rather than regenerating them with SuperSloMo.', allow_abbrev=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--start", type=float, default=0.0, help="start point of video stream")
parser.add_argument("--stop", type=float, default=5.0, help="stop point of video stream")
parser.add_argument("-i", "--input", type=str, help="path of DDD .hdf5 file")
parser.add_argument("-o", "--output_folder", type=str,required=True, help="path to where output is stored (usually previous output folder from ddd_v2e.py)")
parser.add_argument("--slowdown_factor", type=int, default=10, help="slow motion factor")
parser.add_argument("--slomo_model", type=str, default="input/SuperSloMo39.ckpt", help="path of slomo_model checkpoint.")
parser.add_argument("--no_preview", action="store_true", help="disable preview in cv2 windows for faster processing.")
parser.add_argument("--x", type=int, nargs=2, default=None, help="x ROI, two integers, e.g. 10 20; if None, entire address space")
parser.add_argument("--y", type=int, nargs=2, default=None, help="y ROI, two integers, e.g. 40 50; if None, entire address space")
parser.add_argument("--rotate180", type=bool, default=True, help="whether the video needs to be rotated 180 degrees")
args = parser.parse_args()

if __name__ == "__main__":

    # if os.name=='nt':
    #     logger.warning('A Windows python multiprocessing threading problem means that the HDF5 reader will probably not work '
    #                    '\n you may get the infamous "TypeError: h5py objects cannot be pickled" bug')

    if not args.input:
        input_file = inputDDDFileDialog()
        if not input_file:
            logger.info('no file selected, quitting')
            quit()
    else:
        input_file = args.input

    rotate180 = args.rotate180
    preview=not args.no_preview
    assert os.path.exists(input_file)
    assert args.start == None or args.stop == None or args.start < args.stop
    assert os.path.exists(args.slomo_model)

    if args.x==None: args.x=tuple(0,DVS_WIDTH)
    if args.y==None: args.y=tuple(0,DVS_HEIGHT)

    # DDD recordings are mostly upside down. So if x and y refer to rotated input, then we should select real DVS events using rotated x and y coordinates, but when we generate frames with slomo using rotate180, these frames are rotated already, so we should select events from them using original x and y
    x = tuple(args.x)
    y = tuple(args.y)

    write_args_info(args,args.output_folder)

    from pathlib import Path

    output_folder = args.output_folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    slomoVideoFile=os.path.join(output_folder,'slomo.avi')
    if Path(slomoVideoFile).exists():
        logger.info('{} already exists, will use frames from it'.format(slomoVideoFile))
    else:
        slomoVideoFile=None

    frames, dvsEvents = [], []
    dddReader = DDD20SimpleReader(input_file,rotate180=rotate180)
    frames, dvsEvents = dddReader.readEntire(startTimeS=args.start, stopTimeS=args.stop)
    if frames is None or dvsEvents is None: raise Exception('no frames or no events')

    # debug
    ff=frames[0]['frame'].flatten()
    b=np.arange(0,256,1)
    plt.hist(ff,b)
    plt.show()

    dvsEvents=select_events_in_roi(dvsEvents,x,y)
    dvsOnCount = np.count_nonzero((dvsEvents[:, 3] == 1))
    dvsOffCount = dvsEvents.shape[0] - dvsOnCount

    with TemporaryDirectory() as interp_frames_dir:
        logger.info("intepolated frames folder: {}".format(interp_frames_dir))
        slomo = SuperSloMo(model=args.slomo_model, upsampling_factor=args.slowdown_factor, preview=preview)
        slomo.interpolate(images=frames['frame'], output_folder=interp_frames_dir)  # writes all frames to interp_frames_folder
        frame_ts = slomo.get_interpolated_timestamps(frames["ts"])
        height, width = frames['frame'].shape[1:]
        nFrames = frames.shape[0]

        pos_thres = -1.
        neg_thres = -1.

        results = np.empty((0,3),float)
        thresholds = np.arange(1, 0.05, -0.01)
        on_diffs = np.zeros_like(thresholds)
        on_diffs[:]=np.nan
        off_diffs = np.zeros_like(thresholds)
        off_diffs[:]=np.nan

        emulator = EventEmulator(output_folder=None, show_dvs_model_state=None) # 'baseLogFrame'
        k=0
        min_pos_diff=np.inf
        min_neg_diff=np.inf

        fig,ax=plt.subplots()
        plt.rcParams.update({'font.size': 18})
        online, offline=ax.plot(thresholds, on_diffs, 'g-', thresholds, off_diffs, 'r-')
        online.set_label('On')
        offline.set_label('Off')
        ax.set_ylabel('absolute event count difference')
        ax.set_xlabel('threshold (log_e)')
        plt.ion()
        plt.show()
        # plt.legend()

        for threshold in tqdm(thresholds, desc='thr sweep'):
            apsOnEvents = 0
            apsOffEvents = 0
            emulator.pos_thres = threshold
            emulator.neg_thres = threshold
            emulator.reset()
            for i in range(nFrames):
                events_v2e = emulator.generate_events(frames['frame'][i], frame_ts[i])
                if not events_v2e is None:
                    events_v2e=select_events_in_roi(events_v2e,x,y)
                    onCount=np.count_nonzero(events_v2e[:,3]==1)
                    offCount=events_v2e.shape[0]-onCount
                    apsOnEvents += onCount
                    apsOffEvents += offCount

            abs_pos_diff = np.abs(dvsOnCount - apsOnEvents)
            abs_neg_diff = np.abs(dvsOffCount - apsOffEvents)

            if abs_pos_diff < min_pos_diff:
                min_pos_diff=abs_pos_diff
                pos_thres = threshold
            if abs_neg_diff < min_neg_diff:
                min_neg_diff=abs_neg_diff
                neg_thres = threshold

            on_diffs[k]=abs_pos_diff
            off_diffs[k]=abs_neg_diff
            online.set_ydata(on_diffs)
            offline.set_ydata(off_diffs)
            ax.relim()  # Recalculate limits
            ax.autoscale_view(True, True, True)  # Autoscale
            plt.draw()
            plt.pause(.2)
            k=k+1

    if pos_thres > 0 and neg_thres > 0:
        logger.info("Optimal Pos Threshold Found: {}".format(pos_thres))
        logger.info("Optimal Neg Threshold Found: {}".format(neg_thres))

    print("Optimal thresholds for smallest difference in event counts")
    print("thres_on={:.2f} thres_off={:.2f}".format(pos_thres, neg_thres))

    results=np.stack((thresholds,on_diffs,off_diffs),axis=0)
    path = os.path.join(output_folder, 'find_thresholds.npy')
    np.save(path, results)

    path = os.path.join(output_folder, 'find_thresholds.pdf')
    fig.savefig(path)
    path = os.path.join(output_folder, 'find_thresholds.png')
    fig.savefig(path)
    logger.info('saved results to {}'.format(output_folder))

    plt.show()
    try:
        desktop.open(os.path.abspath(output_folder))
    except Exception as e:
        logger.warning('{}: could not open {} in desktop'.format(e, output_folder))
    slomo.cleanup()
    try:
        quit()
    finally:
        sys.exit()

