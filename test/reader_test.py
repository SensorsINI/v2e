import sys
import argparse

sys.path.append("../")
sys.path.append("../utils")

from src.reader import Reader


parser = argparse.ArgumentParser()
parser.add_argument(
    "--fname",
    type=str,
    help="path of the input .hdf5 file")

args = parser.parse_args()

fname = args.fname

m = Reader(fname, start=5)
frames, events = m.read()
