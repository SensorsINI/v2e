import numpy as np
import cv2
import os
import glob

from tqdm import tqdm

from src.simulator import EventEmulator


class Base(object):
    """
    Base class for DVS rendering.
    @author: Zhe He
    @contact: hezhehz@live.cn
    """

    def __init__(
        self,
        frame_ts,
        video_path,
        event_path=None,
        rotate=False
    ):
        """ Init.

        Parameters
        ----------
        frame_ts: np.array, timestamps of output frames.
        video_path: str, path of output video. Example: ../../XX.avi.
        rotate: bool, to ratate the output frames or not.
        event_path: str or None, str if the events need to be saved \
            else None.
        """
        self.frame_ts = frame_ts
        self.video_path = video_path
        self.rotate = rotate
        self.event_path = event_path

    def _get_events(self):
        """ return all events.

        Returns
        -------
        np.ndarray, [timestamp, x, y, polarity]

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError(
            "method self._get_events() needs to be defined."
        )

    def render(self, height, width):
        """ Render event frames.

        Parameters
        ----------
        height: int, height of the frame.
        width: int, width of the frame.

        Returns
        -------
        rendered_frames: np.ndarray, rendered event frames.
        """

        event_arr = self._get_events()

        if self.event_path:
            np.save(self.event_path, event_arr)
            print("events saved!")

        clip_value = 2
        histrange = [(0, v) for v in (height, width)]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
                  self.video_path,
                  fourcc,
                  30.0,
                  (width, height))

        rendered_frames = list()

        for ts_idx in tqdm(range(self.frame_ts.shape[0] - 1),
                           desc="rendering: "):
            # assume time_list is sorted.

            start = np.searchsorted(event_arr[:, 0],
                                    self.frame_ts[ts_idx],
                                    side='left')
            end = np.searchsorted(event_arr[:, 0],
                                  self.frame_ts[ts_idx + 1],
                                  side='right')
            # select events, assume that pos_list is sorted
            if ts_idx < len(self.frame_ts) - 1:
                events = event_arr[start: end]
            else:
                events = event_arr[start:]

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
                    (img_on - img_off), -clip_value, clip_value)
            else:
                integrated_img = (img_on - img_off)
            rendered_frames.append(integrated_img)
            img = (integrated_img + clip_value) / float(clip_value * 2)

            if self.rotate:
                img = np.rot90(img, k=2)

            out.write(
                cv2.cvtColor(
                    (img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
            if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
                break
        out.release()

        rendered_frames = np.vstack(rendered_frames)
        return rendered_frames


class RenderFromImages(Base):
    """
    Subclass of Base, to render events frames from input images.
    @authur: Zhe He
    @contact: hezhehz@live.cn
    @latest update: 2019-May-30
    """

    def __init__(
        self,
        images_path,
        frame_ts,
        interpolated_ts,
        pos_thres,
        neg_thres,
        video_path,
        event_path=None,
        rotate=False
    ):
        """ Init.

        Parameters
        ----------
        images_path: str, path of all images.
        frame_ts: np.array, ts of output frames.
        interpolated_ts: np.array, ts of interpolated frames.
        pos_thres: float, threshold of triggering a positive event.
        neg_thres: float, threshold of triggering a negative event.
        video_path: str, path to store output video.
        event_path: str if the events need to be saved, else None.
        ratate: bool, True if the output frames need to be rotated, else False.
        """
        super().__init__(
            frame_ts, video_path, event_path=event_path, rotate=rotate)
        self.all_images = self.__all_images(images_path)
        self.frame_ts = frame_ts
        self.interpolated_ts = interpolated_ts
        base_frame = self.__read_image(self.all_images[0])
        self.emulator = EventEmulator(
            base_frame,
            pos_thres=pos_thres,
            neg_thres=neg_thres
        )

    def __all_images(self, data_path):
        """Return path of all input images. Assume that the ascending order of
        file names is the same as the order of time sequence.

        Parameters
        ----------
        data_path: str, path of the folder which contains input images.

        Return
        ------
        List[str], sorted in numerical order.
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
        path: str, path of image.

        Returns
        -------
        img: np.ndarray, pixel value in the range of [0., 255.].
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float)
        return img

    def _get_events(self):
        """Get all events."""

        event_list = list()

        for i in tqdm(range(self.interpolated_ts.shape[0] - 1),
                      desc="image2events: "):
            new_frame = self.__read_image(self.all_images[i + 1])
            tmp_events = self.emulator.compute_events(
                new_frame,
                self.interpolated_ts[i],
                self.interpolated_ts[i + 1]
            )
            if tmp_events is not None:
                event_list.append(tmp_events)
        event_arr = np.vstack(event_list)
        print("Amount of events: {}".format(event_arr.shape[0]))

        return event_arr


class RenderFromArray(Base):
    """
    Subclass of Base, to render events frames from image array.
    @authur: Zhe He
    @contact: hezhehz@live.cn
    @latest update: 2019-July-13
    """

    def __init__(
        self,
        image_arr,
        frame_ts,
        input_ts,
        pos_thres,
        neg_thres,
        video_path,
        event_path=None,
        rotate=False
    ):
        """ Init.

        Parameters
        ----------
        images_arr: np.ndarray, input frame array.
        frame_ts: np.array, ts of output frames.
        input_ts: np.array, ts of input frames.
        pos_thres: float, threshold of triggering a positive event.
        neg_thres: float, threshold of triggering a negative event.
        video_path: str, path to store output video.
        event_path: str if the events need to be saved, else None.
        ratate: bool, True if the output frames need to be rotated, else False.
        """
        super().__init__(
            frame_ts, video_path, event_path=event_path, rotate=rotate)
        if not image_arr.shape[0] == input_ts.shape[0]:
            raise ValueError(
                "first dim of image_arr does not match first dim of input_ts")
        self.all_images = image_arr
        self.frame_ts = frame_ts
        self.input_ts = input_ts
        self.emulator = EventEmulator(
            image_arr[0],
            pos_thres=pos_thres,
            neg_thres=neg_thres
        )

    def _get_events(self):
        """Get all events."""

        event_list = list()

        for i in tqdm(range(self.input_ts.shape[0] - 1),
                      desc="image2events: "):
            new_frame = self.all_images[i + 1]
            tmp_events = self.emulator.compute_events(
                new_frame,
                self.input_ts[i],
                self.input_ts[i + 1]
            )
            if tmp_events is not None:
                event_list.append(tmp_events)
        event_arr = np.vstack(event_list)
        print("Amount of events: {}".format(event_arr.shape[0]))

        return event_arr


class RenderFromEvents(Base):
    """
    @author: Zhe He
    @contact: hezhehz@live.cn
    @latest update: 2019-June-2
    """

    def __init__(
        self,
        frame_ts,
        events,
        video_path,
        event_path=None,
        rotate=False,
    ):
        """ Init.

        Parameters
        ----------
        frame_ts: np.array, timestamps of interpolated frames.
        events: numpy structured array, keys: {"ts", "events"}
        video_path: str, path to store output video.
        event_path: str if the events need to be saved, else None.
        ratate: bool, True if the output frames need to be rotated, else False.
        """
        super().__init__(
            frame_ts, video_path, event_path=event_path, rotate=rotate)
        self.events = events

    def _get_events(self):
        """ Return events.

        Returns
        -------
        self.events: numpy structured array, keys: {"ts", "events"}
        """
        return self.events
