import numpy as np


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
    ts_cnt = np.zeros([bin_num, 2])
    for i in range(bin_num - 1):
        condition = np.logical_and(events[:, 0] >= ticks[i],
                                   events[:, 0] < ticks[i + 1])
        if polarity:
            condition = np.logical_and(condition, events[:, 3] == polarity)
        cnt = events[condition].shape[0]
        ts_cnt[i][0] = (ticks[i] + ticks[i + 1]) / 2
        ts_cnt[i][1] = cnt
    
    return ts_cnt
