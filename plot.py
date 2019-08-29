import numpy as np
import argparse
import os
import matplotlib
matplotlib.use('PS')

from matplotlib import pyplot as plt


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
    x_lim = events[:, 1].max()
    y_lim = events[:, 2].max()

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
           y[0] > y_lim or x[1] > y_lim or \
           y[0] > y[1]:
            raise ValueError("y is not in the valid range.")
        y_region = np.logical_and(events[:, 2] >= y[0], events[:, 2] <= y[1])
    else:
        raise TypeError("y must be int or tuple.")

    region = np.logical_and(x_region, y_region)

    return events[region]


def counting(events, start=0, stop=3.5, bin_size=0.5, polarity=None):
    """ Count the amount of events in each bin.
    Parameters
    ----------
    events: np.ndarray, [timestamp, x, y, polarity].
    bin_num: int, default value is 20, the number of bins.
    polarity: int or None. If int, it must be 1 or -1.

    Returns
    -------

    """

    if start < 0 or stop < 0:
        raise ValueError("start and stop must be int.")
    if start + bin_size > stop:
        raise ValueError("start must be less than (stop - bin_size).")
    if polarity and polarity not in [1, -1]:
        raise ValueError("polarity must be 1 or -1.")

    ticks = np.arange(start, stop, bin_size)
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path of input files")
    parser.add_argument("--bin_size", type=float, help="the size of bin")
    parser.add_argument("--start", type=float, help="start time")
    parser.add_argument("--stop", type=float, help="stop time")
    parser.add_argument("--x", type=int, nargs=2, help="x, two integers")
    parser.add_argument("--y", type=int, nargs=2, help="y, two integers")
    # parser.add_argument(
    #     "--type",
    #     type=str,
    #     choices=["positive", "negative", "all"],
    #     help="polarity")
    args = parser.parse_args()

    # if args.type == "positive":
    #     polarity = 1
    # elif args.type == "negative":
    #     polarity = -1
    # else:
    #     polarity = None

    events_aps = np.load(os.path.join(args.path, "events_aps.npy"))
    events_dvs = np.load(os.path.join(args.path, "events_dvs.npy"))

    x = tuple(args.x)
    y = tuple(args.y)

    aps = select(events_aps, x, y)
    dvs = select(events_dvs, x, y)

    aps_all = counting(aps, bin_size=args.bin_size, polarity=None,
                       start=args.start, stop=args.stop)
    dvs_all = counting(dvs, bin_size=args.bin_size, polarity=None,
                       start=args.start, stop=args.stop)
    aps_pos = counting(aps, bin_size=args.bin_size, polarity=1,
                       start=args.start, stop=args.stop)
    dvs_pos = counting(dvs, bin_size=args.bin_size, polarity=1,
                       start=args.start, stop=args.stop)
    aps_neg = counting(aps, bin_size=args.bin_size, polarity=-1,
                       start=args.start, stop=args.stop)
    dvs_neg = counting(dvs, bin_size=args.bin_size, polarity=-1,
                       start=args.start, stop=args.stop)

    fig = plt.figure(figsize=(20, 20))

    y_max = max(aps_all[:, 1].max(), dvs_all[:, 1].max()) + 1
    plt.subplot(3, 2, 1)
    plt.bar(aps_all[:, 0], aps_all[:, 1],
            width=args.bin_size / 2, color='blue')
    plt.xlabel("t [s]", fontsize=16)
    plt.ylabel("#(events)", fontsize=16)
    plt.ylim([0, y_max])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("From APS (all)", fontsize=18)

    plt.subplot(3, 2, 2)
    plt.bar(dvs_all[:, 0], dvs_all[:, 1],
            width=args.bin_size / 1.6, color='orange')
    plt.xlabel("t [s]", fontsize=16)
    plt.ylabel("#(events)", fontsize=16)
    plt.ylim([0, y_max])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("From DVS (all)", fontsize=18)

    y_max = max(aps_pos[:, 1].max(), dvs_pos[:, 1].max()) + 1
    plt.subplot(3, 2, 3)
    plt.bar(aps_pos[:, 0], aps_pos[:, 1],
            width=args.bin_size / 2, color='blue')
    plt.xlabel("t [s]", fontsize=16)
    plt.ylabel("#(events)", fontsize=16)
    plt.ylim([0, y_max])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("From APS (positive)", fontsize=18)

    plt.subplot(3, 2, 4)
    plt.bar(dvs_pos[:, 0], dvs_pos[:, 1],
            width=args.bin_size / 1.6, color='orange')
    plt.xlabel("t [s]", fontsize=16)
    plt.ylabel("#(events)", fontsize=16)
    plt.ylim([0, y_max])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("From DVS (positive)", fontsize=18)

    y_max = max(aps_neg[:, 1].max(), dvs_neg[:, 1].max()) + 1

    plt.subplot(3, 2, 5)
    plt.bar(aps_neg[:, 0], aps_neg[:, 1],
            width=args.bin_size / 2, color='blue')
    plt.xlabel("t [s]", fontsize=16)
    plt.ylabel("#(events)", fontsize=16)
    plt.ylim([0, y_max])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("From APS (negative)", fontsize=18)

    plt.subplot(3, 2, 6)
    plt.bar(dvs_neg[:, 0], dvs_neg[:, 1],
            width=args.bin_size / 1.6, color='orange')
    plt.xlabel("t [s]", fontsize=16)
    plt.ylabel("#(events)", fontsize=16)
    plt.ylim([0, y_max])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("From DVS (negative)", fontsize=18)

    # save the figure
    plt.savefig(os.path.join(args.path, "plot.pdf"))
