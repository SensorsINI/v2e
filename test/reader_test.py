import sys
import argparse

sys.path.append("../")
sys.path.append("../utils")

from src.dddh5reader import DDD20ReaderMultiProcessing


parser = argparse.ArgumentParser()
parser.add_argument(
    "--fname",
    type=str,
    help="path of the input .hdf5 file")

args = parser.parse_args()

fname = args.fname

m = DDD20ReaderMultiProcessing(fname, startTimeS=5)
frames, events = m.readEntire()
