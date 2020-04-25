import logging
import os

import numpy as np
import argparse

from tempfile import TemporaryDirectory

from emulator import EventEmulator
from src.slomo import SuperSloMo
from src.ddd20_utils.ddd_h5_reader import DDD20ReaderMultiProcessing, DDD20SimpleReader

# TODO rename to find_thresholds.py
from v2e_utils import inputVideoFileDialog, inputDDDFileDialog

logging.basicConfig()
root = logging.getLogger()
root.setLevel(logging.DEBUG)
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
logging.addLevelName( logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName( logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
logger=logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=float, default=0.0, help="start point of video stream")
parser.add_argument("--stop", type=float, default=5.0, help="stop point of video stream")
parser.add_argument("-i", type=str, help="path of DDD .hdf5 file")
parser.add_argument("--model", type=str, default='input/SuperSlo39.ckpt', help="path of checkpoint")
parser.add_argument("--slowdown_factor", type=int, default=10, help="slow motion factor")
args = parser.parse_args()

if __name__ == "__main__":

    if os.name=='nt':
        logger.warning('A Windows python multiprocessing threading problem means that the HDF5 reader will probably not work '
                       '\n you may get the infamous "TypeError: h5py objects cannot be pickled" bug')


    if not args.i:
        input_file = inputDDDFileDialog()
        if not input_file:
            logger.info('no file selected, quitting')
            quit()
    else: input_file=args.i

    dddReader = DDD20ReaderMultiProcessing(input_file, startTimeS=args.start, stopTimeS=args.stop)
    frames, events = dddReader.readEntire()

    results = []

    with TemporaryDirectory() as dirname:
        logger.info("tmp_dir: ", dirname)
        s = SuperSloMo(args.checkpoint, args.sf, dirname)
        s.interpolate(frames["frame"])
        frame_ts = s.get_ts(frames["ts"])
        height, width = frames["frame"].shape[1:]
        nFrames=frames.shape[0]

        emulator= EventEmulator(frames[0],output_folder=None,rotate180=True,show_input=True)

        events_dvs=[]
        for i in range(nFrames):
            e=emulator.compute_events(frames[i],frame_ts[i],frame_ts[i+1])
            events.append(e)

        num_pos_dvs = events_dvs[events_dvs[:, 3] == 1].shape[0]
        num_neg_dvs = events_dvs.shape[0] - num_pos_dvs

        pos_thres = -1.
        neg_thres = -1.

        for threshold in np.arange(0.01, 0.91, 0.01):

            r = VideoSequenceFiles2EventsRenderer(
                dirname,
                frame_ts,
                frame_ts,
                threshold,
                threshold,
                "data/from_image_{:.2f}.avi".format(threshold))

            events_aps = r._get_events()

            num_pos_aps = events_aps[events_aps[:, 3] == 1].shape[0]
            num_neg_aps = events_aps.shape[0] - num_pos_aps

            abs_pos_diff = np.abs(num_pos_dvs - num_pos_aps)
            abs_neg_diff = np.abs(num_neg_dvs - num_neg_aps)

            if len(results) > 0:
                if abs_pos_diff >= results[-1][1] and pos_thres < 0:
                    pos_thres = results[-1][0]
                if abs_neg_diff >= results[-1][2] and neg_thres < 0:
                    neg_thres = results[-1][0]
            if pos_thres > 0 and neg_thres > 0:
                print("Optimal Pos Threshold Found: {}".format(pos_thres))
                print("Optimal Neg Threshold Found: {}".format(neg_thres))
                break
            results.append(
                [threshold, abs_pos_diff, abs_neg_diff]
            )
            print("Threshold: {:.2f}".format(threshold))
            print("Pos Thres: {:.2f}".format(pos_thres))
            print("Neg Thres: {:.2f}".format(neg_thres))
    results = np.array(results)
    np.save('data/results.npy', results)
