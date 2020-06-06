import sys
from pathlib import Path

import numpy as np
import argparse
import os
import cv2
import logging

from v2e import desktop

logging.basicConfig()
root = logging.getLogger()
root.setLevel(logging.INFO)
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
logging.addLevelName( logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName( logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
logger=logging.getLogger(__name__)


from v2e.v2e_utils import video_writer, select_events_in_roi, histogram_events_in_time_bins, DVS_WIDTH, DVS_HEIGHT
from v2e.v2e_args import write_args_info

# matplotlib.use('PS')

from matplotlib import pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plots time trace of event activity from real DVS and v2e emualated events',
                                 epilog='Run with no --input to open file dialog.\nAssumes that numpy data files have been generated using the --numpy_output option to ddd-v2e.py', allow_abbrev=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--path", type=str, required=True, help="path to numpy input files and for storing output")
    parser.add_argument("--time_bin_ms", type=float, default=50, help="the duration of time bins in ms")
    parser.add_argument("--start", required=True,type=float, help="start time in seconds relative to start of numpy data file (might not be same as recording)")
    parser.add_argument("--stop", required=True,type=float, help="stop time in seconds relative to start of numpy data file (might not be same as recording)")
    parser.add_argument("--x", type=int, nargs=2, required=True, help="x ROI, two integers, e.g. 10 20")
    parser.add_argument("--y", type=int, nargs=2, required=True, help="y ROI, two integers, e.g. 40 50")
    parser.add_argument("--rotate180", type=bool, default=True, help="whether the video needs to be rotated 180 degrees")

    args = parser.parse_args()


    path=args.path
    time_bin_ms=args.time_bin_ms
    time_bin_s=time_bin_ms*.001
    rotate180=args.rotate180

    if not Path(path).exists():
        logger.error('input folder {} not accessible'.format(Path(path)))
        sys.exit(1)

    dvs_video_real_avi = os.path.join(path, "dvs-video-real.avi")
    if not Path(dvs_video_real_avi).exists():
        logger.error('video {} not accessible'.format(dvs_video_real_avi))
        sys.exit(1)

    dvs_v2e_npy = os.path.join(path, "dvs_v2e.npy")
    dvs_real_npy = os.path.join(path, "dvs_real.npy")
    if not Path(dvs_v2e_npy).exists() or not Path(dvs_real_npy).exists():
        logger.error('numpy event files {} or {} not accessible\nDid you run ddd-v2e.py first?'.format(dvs_v2e_npy, dvs_real_npy))
        sys.exit(1)

    events_aps = np.load(dvs_v2e_npy)
    events_dvs = np.load(dvs_real_npy)

    cap = cv2.VideoCapture(dvs_video_real_avi)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outputVideoPath = os.path.join(path, "ddd_plot_event_counts.avi")
    out=video_writer(outputVideoPath, width=width, height=height)

    logger.info("Input video {}: width: {} \t height: {}".format(dvs_video_real_avi,width, height))

    x = tuple(args.x)
    y = tuple(args.y)

    write_args_info(args,path)

    ts0=np.min((events_dvs[0][0],events_aps[0][0]))

    start = (ts0+args.start)
    stop = (args.stop+ts0)
    i = 0
    logger.info('writing video output file {} with ROI labeled'.format(outputVideoPath))
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret: # and i >= start and i <= stop:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.rectangle(
                img, (args.x[0], args.y[0]), (args.x[1], args.y[1]), 255, 2)
            out.write(cv2.cvtColor((img * 255).astype(np.uint8),
                                   cv2.COLOR_GRAY2BGR))
            if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
                break
        else:
            break
        i += 1
    out.release()
    logger.info("finished writing {}.".format(outputVideoPath))

    aps = select_events_in_roi(events_aps, x, y)
    dvs_v2e_npy = select_events_in_roi(events_dvs, x, y)

    logger.info('histogram_events_in_time_bins events...')
    aps_pos = histogram_events_in_time_bins(aps, time_bin_ms=time_bin_ms, polarity=1, start=start, stop=stop)
    dvs_pos = histogram_events_in_time_bins(dvs_v2e_npy, time_bin_ms=time_bin_ms, polarity=1, start=start, stop=stop) # APS off events from v2e have polarity -1
    aps_neg = histogram_events_in_time_bins(aps, time_bin_ms=time_bin_ms, polarity=-1, start=start, stop=stop)
    dvs_neg = histogram_events_in_time_bins(dvs_v2e_npy, time_bin_ms=time_bin_ms, polarity=0, start=start, stop=stop) # note DVS off events have polarity 0

    fig = plt.figure(figsize=(8, 6))
    width = time_bin_s / 2

    y_max = max(aps_pos[:, 1].max(), dvs_pos[:, 1].max()) + 1
    y_min = -max(aps_neg[:, 1].max(), dvs_neg[:, 1].max()) - 1

    plt.bar(dvs_pos[:, 0] + 0.5 * width-ts0, dvs_pos[:, 1],
            width=width, color='red',label='DVS ON')
    plt.bar(aps_pos[:, 0] - 0.5 * width-ts0, aps_pos[:, 1],
            width=width, color='blue', label="v2e ON")
    plt.bar(dvs_neg[:, 0] + 0.5 * width-ts0, -dvs_neg[:, 1],
            width=width, color='orange', label='DVS OFF')
    plt.bar(aps_neg[:, 0] - 0.5 * width-ts0, -aps_neg[:, 1],
            width=width, color='green', label="v2e OFF")

    plt.xlabel("t [s]", fontsize=16)
    plt.ylabel("#(events)", fontsize=16)
    plt.ylim([y_min, y_max])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("#(Events) - Time", fontsize=18)

    plt.legend()

    # save the figure
    plt.savefig(os.path.join(path, "ddd-plot-event-counts.pdf"))
    plt.savefig(os.path.join(path, "ddd-plot-event-counts.png"))
    logger.info('plots written to ddd-plot-event-counts.* in folder {}'.format(path))
    plt.show()
    try:
        desktop.open(os.path.abspath(path))
    except Exception as e:
        logger.warning('{}: could not open {} in desktop'.format(e, path))
    try:
        quit()
    finally:
        sys.exit()

