


class Base(object):
    """
    Base class for DVS rendering.
    @author: Zhe He
    @contact: hezhehz@live.cn
    """

    def __init__(self):
        """
        """

    def _get_events(self):
        """
        return all events.
        @Return: np.ndarray.
            [timestamp, x, y, polarity]
        """
        raise NotImplementedError(
            "method self._get_events() needs to be defined."
        )

    def render(self, height, width):
        """Render event frames."""
        event_arr = self._get_events()
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