"""
DVS simulator.
Compute events from input frames.

@author: Zhe He
@contact: zhehe@student.ethz.ch
@credits: Yuhuang Hu
@latest updaste: 2019-Jun-13
"""

import glob
import os
import cv2
import numpy as np

# random seed. Why 42? it is the answer to everything :)
np.random.seed(42)


def lin_log(x, threshold=20):
    """
    linear mapping + logrithmic mapping.
    @author: Zhe He
    @contact: hezhehz@live.cn
    """

    # converting x into np.float32.
    if x.dtype is not np.float32:
        x = x.astype(np.float32)

    y = np.piecewise(
        x,
        [x < threshold, x >= threshold],
        [lambda x: x / threshold * np.log(threshold),
         lambda x: np.log(x)]
    )

    return y


class EventEmulator(object):
    """compute events based on the input frame.
    - author: Zhe He
    - contact: zhehe@student.ethz.ch
    """
    # todo add event count statistics for ON and OFF events

    def __init__(
        self,
        base_frame,
        pos_thres=0.21,
        neg_thres=0.17,
            sigma=0.03,
            seed=0
    ):
        """
        Parameters
        ----------
        base_frame: np.ndarray
            [height, width].
        pos_thres: float, default 0.21
            threshold of triggering positive event in log intensity.
        neg_thres: float, default 0.17
            threshold of triggering negative event in log intensity.
        sigma: float, default 0.03
            std deviation of threshold in log intensity.
        seed: int, default=0
            seed for random threshold variations, fix it to nonzero value to get same mismatch every time
        """

        print("positive threshold: {}".format(pos_thres))
        print("negative threshold: {}".format(neg_thres))

        self.base_frame = lin_log(base_frame) # base_frame are memorized loglin pixel values
        # take the variance of threshold into account.
        if seed !=0: np.random.seed(seed)
        pos_thres = np.random.normal(pos_thres, sigma, base_frame.shape) # todo put sigma to args
        # to avoid the situation where the threshold is too small.
        pos_thres[pos_thres < 0.01] = 0.01
        neg_thres = np.random.normal(neg_thres, sigma, base_frame.shape)# todo put sigma to args
        neg_thres[neg_thres < 0.01] = 0.01
        self.pos_thres = pos_thres
        self.neg_thres = neg_thres

    def compute_events(self, new_frame, t_start, t_end):
        """Compute events in new frame.

        Parameters
        ----------
        new_frame: np.ndarray
            [height, width]
        ts: float
            timestamp of new frame.
        verbose: bool
            verbose.

        Returns
        -------
        events: np.ndarray if any event else None
            [N, 4], each row contains [timestamp, y cordinate,
            x cordinate, sign of event].
        """

        if t_start > t_end:
            raise ValueError("t_start must be smaller than t_end")

        log_frame = lin_log(new_frame)
        diff_frame = log_frame - self.base_frame

        pos_frame = np.zeros_like(diff_frame)
        neg_frame = np.zeros_like(diff_frame)
        pos_frame[diff_frame > 0] = diff_frame[diff_frame > 0]
        neg_frame[diff_frame < 0] = np.abs(diff_frame[diff_frame < 0])

        pos_evts_frame = pos_frame // self.pos_thres
        pos_iters = int((pos_frame // self.pos_thres).max())
        neg_evts_frame = neg_frame // self.neg_thres
        neg_iters = int((neg_frame // self.neg_thres).max())

        num_iters = max(pos_iters, neg_iters)

        events = []

        for i in range(num_iters):

            ts = t_start + (t_end - t_start) * (i + 1) / (num_iters + 1)

            pos_cord = (pos_frame > self.pos_thres * (i + 1))
            neg_cord = (neg_frame > self.neg_thres * (i + 1))

            # generate events
            pos_event_xy = np.where(pos_cord)
            num_pos_events = pos_event_xy[0].shape[0]
            neg_event_xy = np.where(neg_cord)
            num_neg_events = neg_event_xy[0].shape[0]
            num_events = num_pos_events + num_neg_events

            # sort out the positive event and negative event
            if num_pos_events > 0:
                pos_events = np.hstack(
                    (np.ones((num_pos_events, 1), dtype=np.float32) * ts,
                     pos_event_xy[1][..., np.newaxis],
                     pos_event_xy[0][..., np.newaxis],
                     np.ones((num_pos_events, 1), dtype=np.float32) * 1))

            else:
                pos_events = None

            if num_neg_events > 0:
                neg_events = np.hstack(
                    (np.ones((num_neg_events, 1), dtype=np.float32) * ts,
                     neg_event_xy[1][..., np.newaxis],
                     neg_event_xy[0][..., np.newaxis],
                     np.ones((num_neg_events, 1), dtype=np.float32) * -1))

            else:
                neg_events = None

            if pos_events is not None and neg_events is not None:
                events_tmp = np.vstack((pos_events, neg_events))
                events_tmp = events_tmp.take(
                    np.random.permutation(
                        events_tmp.shape[0]), axis=0)
            else:
                if pos_events is not None:
                    events_tmp = pos_events
                else:
                    events_tmp = neg_events

            if i == 0:
                # update base frame todo check math correct here with yuhu
                if num_pos_events > 0:
                    #  self.base_frame[pos_cord] = log_frame[pos_cord]
                    self.base_frame[pos_cord] += \
                        pos_evts_frame[pos_cord]*self.pos_thres[pos_cord] # add to memorized brightness values just the events we emitted. don't add the remainder. the next aps frame might have sufficient value to trigger another event or it might not, but we are correct in not storing the current frame brightness
                if num_neg_events > 0:
                    #  self.base_frame[neg_cord] = log_frame[neg_cord]
                    self.base_frame[neg_cord] -= \
                        neg_evts_frame[neg_cord]*self.neg_thres[neg_cord]

            if num_events > 0:
                events.append(events_tmp)

        if len(events) > 0:
            events = np.vstack(events)
            return events
        else:
            return None


class EventFrameRenderer(object):
    """ Deprecated
    class for rendering event frames.
    - author: Zhe He
    - contact: zhehe@student.ethz.ch
    """

    def __init__(self,
                 data_path,
                 output_path,
                 input_fps,
                 output_fps,
                 pos_thres,
                 neg_thres):
        """
        Parameters
        ----------
        data_path: str
            path of frames.
        output_path: str
            path of output video.
        input_fps: int
            frame rate of input video.
        output_fps: int
            frame rate of output video.
        """

        self.data_path = data_path
        self.output_path = output_path
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.pos_thres = pos_thres
        self.neg_thres = neg_thres

    def __all_images(self, data_path):
        """Return path of all input images. Assume that the ascending order of
        file names is the same as the order of time sequence.

        Parameters
        ----------
        data_path: str
            path of the folder which contains input images.

        Returns
        -------
        List[str]
            sorted in numerical order.
        """
        images = glob.glob(os.path.join(data_path, '*.png'))
        if len(images) == 0:
            raise ValueError(("Input folder is empty or images are not in"
                              " 'png' format."))
        images_sorted = sorted(
                images,
                key=lambda line: int(line.split('/')[-1].split('.')[0]))
        return images_sorted

    @staticmethod
    def __read_image(path):
        """Read image.

        Parameters
        ----------
        path: str
            path of image.

        Returns
        -------
        img: np.ndarray
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float) / 255.
        return img

    def _get_events(self):
        """Get all events.
        """
        images = self.__all_images(self.data_path)
        num_frames = len(images)
        input_ts = np.linspace(
                0,
                num_frames / self.input_fps,
                num_frames,
                dtype=np.float)
        base_frame = self.__read_image(images[0])
        print(base_frame.shape)
        height = base_frame.shape[0]
        width = base_frame.shape[1]
        emulator = EventEmulator(
            base_frame,
            pos_thres=self.pos_thres,
            neg_thres=self.neg_thres
        )

        event_list = list()
        time_list = list()
        pos_list = list()

        # index of the first element at timestamp t.
        pos = 0

        for idx in range(1, num_frames):
            new_frame = self.__read_image(images[idx])
            t_start = input_ts[idx - 1]
            t_end = input_ts[idx]
            tmp_events = emulator.compute_events(
                new_frame,
                t_start,
                t_end
            )

            if tmp_events is not None:
                event_list.append(tmp_events)
                pos_list.append(pos)
                time_list.append(t_end)

                # update pos
                pos += tmp_events.shape[0]

            if (idx + 1) % 20 == 0:
                print("Image2Events processed {} frames".format(idx + 1))

        event_arr = np.vstack(event_list)
        print("Number of events: {}".format(event_arr.shape[0])) # TODO engineering format, events/sec

        return event_arr, time_list, pos_list, num_frames, height, width

    def render(self):
        """Render event frames."""
        (event_arr, time_list, pos_list,
         num_frames, height, width) = self._get_events()

        output_ts = np.linspace(
                0,
                num_frames / self.input_fps,
                int(num_frames / self.input_fps * self.output_fps),
                dtype=np.float)
        clip_value = 2
        histrange = [(0, v) for v in (height, width)]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
                  os.path.join(self.output_path, 'output.avi'),
                  fourcc,
                  30.0,
                  (width, height))

        for ts_idx in range(output_ts.shape[0] - 1):
            # assume time_list is sorted.
            start = np.searchsorted(time_list,
                                    output_ts[ts_idx],
                                    side='right')
            end = np.searchsorted(time_list,
                                  output_ts[ts_idx + 1],
                                  side='right')
            # select events, assume that pos_list is sorted
            if end < len(pos_list):
                events = event_arr[pos_list[start]: pos_list[end], :]
            else:
                events = event_arr[pos_list[start]:, :]

            pol_on = (events[:, 3] == 1)
            pol_off = np.logical_not(pol_on)
            img_on, _, _ = np.histogram2d(
                    events[pol_on, 2], events[pol_on, 1],
                    bins=(height, width), range=histrange)
            img_off, _, _ = np.histogram2d(
                    events[pol_off, 2], events[pol_off, 1],
                    bins=(height, width), range=histrange)
            if clip_value is not None:
                integrated_img = np.clip(
                    (img_on-img_off), -clip_value, clip_value)
            else:
                integrated_img = (img_on-img_off)
            img = (integrated_img + clip_value) / float(clip_value * 2)
            out.write(
                cv2.cvtColor(
                    (img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
            if ts_idx % 20 == 0:
                print('Rendered {} frames'.format(ts_idx))
            if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
                break
        out.release()
