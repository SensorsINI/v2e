import logging
import os
import numpy as np
import argparse
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.emulator import EventEmulator
from src.slomo import SuperSloMo
from src.ddd20_utils.ddd_h5_reader import DDD20ReaderMultiProcessing, DDD20SimpleReader
from src.v2e_utils import inputVideoFileDialog, inputDDDFileDialog

logging.basicConfig()
root = logging.getLogger()
root.setLevel(logging.INFO)
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='ddd_find_thresholds.py: generate simulated DVS events from video with sweep of thresholds to compare with real DVS to find optimal thresholds.',
                                 epilog='Run with no --input to open file dialog', allow_abbrev=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--start", type=float, default=0.0, help="start point of video stream")
parser.add_argument("--stop", type=float, default=5.0, help="stop point of video stream")
parser.add_argument("-i", type=str, help="path of DDD .hdf5 file")
parser.add_argument("-o", type=str, default='output/find_thresholds', help="path to where output is stored")
parser.add_argument("--slowdown_factor", type=int, default=10, help="slow motion factor")
parser.add_argument("--slomo_model", type=str, default="input/SuperSloMo39.ckpt", help="path of slomo_model checkpoint.")
parser.add_argument("--no_preview", action="store_true", help="disable preview in cv2 windows for faster processing.")
args = parser.parse_args()

if __name__ == "__main__":

    # if os.name=='nt':
    #     logger.warning('A Windows python multiprocessing threading problem means that the HDF5 reader will probably not work '
    #                    '\n you may get the infamous "TypeError: h5py objects cannot be pickled" bug')

    if not args.i:
        input_file = inputDDDFileDialog()
        if not input_file:
            logger.info('no file selected, quitting')
            quit()
    else:
        input_file = args.i

    preview=not args.no_preview
    assert os.path.exists(input_file)
    assert args.start == None or args.stop == None or args.start < args.stop
    assert os.path.exists(args.slomo_model)

    from pathlib import Path

    outdir = args.o
    Path(outdir).mkdir(parents=True, exist_ok=True)

    frames, dvsEvents = [], []
    dddReader = DDD20SimpleReader(input_file)
    frames, dvsEvents = dddReader.readEntire(startTimeS=args.start, stopTimeS=args.stop)
    if frames is None or dvsEvents is None: raise Exception('no frames or no events')

    dvsOnEvents = dvsEvents[dvsEvents[:, 3] == 1].shape[0]
    dvsOffEvents = dvsEvents.shape[0] - dvsOnEvents

    with TemporaryDirectory() as interp_frames_dir:
        logger.info("intepolated frames folder: {}".format(interp_frames_dir))
        slomo = SuperSloMo(model=args.slomo_model, slowdown_factor=args.slowdown_factor,preview=preview)
        slomo.interpolate(images=frames['frame'], output_folder=interp_frames_dir)  # writes all frames to interp_frames_folder
        frame_ts = slomo.get_interpolated_timestamps(frames["ts"])
        height, width = frames['frame'].shape[1:]
        nFrames = frames.shape[0]

        pos_thres = -1.
        neg_thres = -1.

        results = np.empty((0,3),float)
        thresholds = np.arange(1, 0.2, -0.01)
        on_diffs = np.zeros_like(thresholds)
        on_diffs[:]=np.nan
        off_diffs = np.zeros_like(thresholds)
        off_diffs[:]=np.nan

        emulator = EventEmulator(output_folder=None, show_input=False)
        k=0
        min_pos_diff=np.inf
        min_neg_diff=np.inf

        fig,ax=plt.subplots()
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
                e = emulator.generate_events(frames['frame'][i], frame_ts[i], frame_ts[i + 1])
                apsOnEvents += emulator.num_events_on
                apsOffEvents += emulator.num_events_off

            abs_pos_diff = np.abs(dvsOnEvents - apsOnEvents)
            abs_neg_diff = np.abs(dvsOffEvents - apsOffEvents)

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
    path = os.path.join(outdir, 'find_thresholds.npy')
    np.save(path, results)

    path = os.path.join(outdir, 'find_thresholds.pdf')
    fig.savefig(path)
    path = os.path.join(outdir, 'find_thresholds.png')
    fig.savefig(path)
    logger.info('saved results to {}'.format(outdir))


