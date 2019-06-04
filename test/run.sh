#!/bin/bash
for i in {0..0.2..0.01}
  do 
    python renderer_test.py --fname ../data/rec1500394622.hdf5 --checkpoint ../data/SuperSloMo38.ckpt --start 5.0 --stop 10.0 --threshold i
 done