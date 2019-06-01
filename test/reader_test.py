import sys
import queue as Queue
import pdb
import time

sys.path.append("../")
sys.path.append("../utils")

from src.reader import Reader

fname = "../data/rec1487354811.hdf5"

m = Reader(fname, start=5)
frames, events = m.read()
pdb.set_trace()
