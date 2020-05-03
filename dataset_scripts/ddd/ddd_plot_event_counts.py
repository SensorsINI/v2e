from pathlib import Path

import numpy as np
import argparse
import os
import matplotlib
import cv2

from src.v2e_utils import video_writer

# matplotlib.use('PS')

from matplotlib import pyplot as plt

DVS_WIDTH, DVS_HEIGHT = 346,260  # adjust for different recording


def select(events, x, y):
    """ Select the events at the region specified by x and y.
    Parameters
    ----------
    events: np.ndarray, [timestamp, x, y, polarity]
    x: int or tuple, x coordinate.
    y: int or tuple, y coordinate.

    Returns
    -------
    np.ndarray, with the same shape as events.
    """
    x_lim = DVS_WIDTH-1 # events[:, 1].max()
    y_lim = DVS_HEIGHT-1 # events[:, 2].max()

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


def counting(events, start=0, stop=3.5, time_bin_ms=50, polarity=None):
    """ Count the amount of events in each bin.
    Parameters
    ----------
    events: np.ndarray, [timestamp, x, y, polarity].
    bin_num: int, default value is 20, the number of bins.
    polarity: int or None. If int, it must be 1 or -1.

    Returns
    -------

    """
    time_bin_s=time_bin_ms*0.001

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plots time trace of event activity from real DVS and v2e emualated events',
                                 epilog='Run with no --input to open file dialog', allow_abbrev=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--path", type=str, required=True, help="path to numpy input files and for storing output")
    parser.add_argument("--time_bin_ms", type=float, default=50, help="the duration of time bins in ms")
    parser.add_argument("--start", required=True,type=float, help="start time in seconds")
    parser.add_argument("--stop", required=True,type=float, help="stop time in seconds")
    parser.add_argument("--x", type=int, nargs=2, required=True, help="x, two integers, e.g. 10 20")
    parser.add_argument("--y", type=int, nargs=2, required=True, help="y, two integers, e.g. 40 50")
    parser.add_argument("--rotate180", type=bool, default=True, help="whether the video needs to be rotated")
    args = parser.parse_args()


    path=args.path
    time_bin_ms=args.time_bin_ms
    time_bin_s=time_bin_ms*.001
    rotate180=args.rotate180

    assert Path(path).exists()
    assert Path(os.path.join(path, "dvs-video-real.avi")).exists()

    events_aps = np.load(os.path.join(path, "dvs_v2e.npy"))
    events_dvs = np.load(os.path.join(path, "dvs_real.npy"))

    # cap = cv2.VideoCapture(os.path.join(path, "dvs-video-real.avi"))
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # out=video_writer(os.path.join(path, "counting.avi"),width=width,height=height)
    #
    # print("width: {} \t height: {}".format(width, height))

    if not rotate180:
        x = tuple(args.x)
        y = tuple(args.y)
    else:
        x = tuple([DVS_WIDTH - 1 - args.x[1], DVS_WIDTH - 1 - args.x[0]])
        y = tuple([DVS_HEIGHT - 1 - args.y[1], DVS_HEIGHT - 1 - args.y[0]])

    ts0=np.min((events_dvs[0][0],events_aps[0][0]))

    start = (ts0+args.start)
    stop = (args.stop+ts0)
    # i = 0
    # while(cap.isOpened()):
    #     ret, img = cap.read()
    #     if ret and i >= start and i <= stop:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         cv2.rectangle(
    #             img, (args.x[0], args.y[0]), (args.x[1], args.y[1]), 255, 2)
    #         out.write(cv2.cvtColor((img * 255).astype(np.uint8),
    #                                cv2.COLOR_GRAY2BGR))
    #         if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break
    #     i += 1
    # out.release()
    # print("finished writing video.")

    aps = select(events_aps, x, y)
    dvs = select(events_dvs, x, y)

    aps_pos = counting(aps, time_bin_ms=time_bin_ms, polarity=1, start=start, stop=stop)
    dvs_pos = counting(dvs, time_bin_ms=time_bin_ms, polarity=1, start=start, stop=stop) # APS off events from v2e have polarity -1
    aps_neg = counting(aps, time_bin_ms=time_bin_ms, polarity=-1, start=start, stop=stop)
    dvs_neg = counting(dvs, time_bin_ms=time_bin_ms, polarity=0, start=start, stop=stop) # note DVS off events have polarity 0

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
    plt.show()
