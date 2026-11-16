import logging

import dv_processing as dv
import numpy as np
from engineering_notation import EngNumber

logger = logging.getLogger(__name__)

# DV camera family names with a trailing serial. jAER's RecordingChipDetector
# uses FAMILY_SERIAL to select the chip and treats those coordinates as
# OpenCV (origin at the top left), which is v2e's convention.
_JAER_SOURCES = {
    (346, 260): "DAVIS346_v2e",
    (240, 180): "DAVIS240C_v2e",
    (128, 128): "DVS128_v2e",
    (640, 480): "DVS640_v2e",
}


def jaer_aedat4_source(output_width: int, output_height: int) -> str:
    """Camera name stored in the AEDAT-4 infoNode for jAER chip detection."""
    source = _JAER_SOURCES.get((output_width, output_height))
    if source is not None:
        return source
    return f"V2E_{output_width}x{output_height}"


class AEDat4Output:
    """Write DVS events as an iniVation AEDAT-4.0 file that jAER can play."""

    def __init__(self, filepath: str, output_width: int, output_height: int):
        if output_width is None or output_height is None:
            raise ValueError(
                "AEDAT-4.0 output needs output_width and output_height. "
                "Set them or pass a camera size such as --dvs346.")
        self.filepath = filepath
        self.sizex = int(output_width)
        self.sizey = int(output_height)
        self.numEventsWritten = 0
        self.numOnEvents = 0
        self.numOffEvents = 0
        self.last_us = None
        self._warned_noise_label = False
        self.source = jaer_aedat4_source(self.sizex, self.sizey)
        logger.info(
            "opening AEDAT-4.0 output file {} ({}x{}, source {})".format(
                filepath, self.sizex, self.sizey, self.source))
        if self.source.startswith("V2E_"):
            logger.warning(
                "size {}x{} has no standard jAER chip. "
                "Open the file in jAER and select the chip class yourself.".format(
                    self.sizex, self.sizey))

        config = dv.io.MonoCameraWriter.EventOnlyConfig(
            self.source,
            (self.sizex, self.sizey),
            dv.CompressionType.LZ4)
        self.writer = dv.io.MonoCameraWriter(filepath, config)

    def cleanup(self):
        self.close()

    def close(self):
        if self.writer is None:
            return
        logger.info(
            "Closing {} after writing {} events ({} on, {} off)".format(
                self.filepath,
                EngNumber(self.numEventsWritten),
                EngNumber(self.numOnEvents),
                EngNumber(self.numOffEvents)))
        # Dropping the writer flushes the background IO thread.
        self.writer = None

    def appendEvents(self, events: np.ndarray, signnoise_label: np.ndarray = None):
        """Append events to the AEDAT-4.0 file.

        Parameters
        ----------
        events: np.ndarray
            [N, 4] rows of [timestamp (s), x, y, sign (+1 ON, -1 OFF)].
        signnoise_label: np.ndarray
            Unused. AEDAT-4 polarity events have no noise bit.
        """
        if self.writer is None or events is None or len(events) == 0:
            return
        if signnoise_label is not None and not self._warned_noise_label:
            logger.warning(
                "AEDAT-4.0 does not store signal/noise labels. "
                "Use --dvs_text or --dvs_aedat2 to keep them.")
            self._warned_noise_label = True

        timestamp_us = np.rint(events[:, 0].astype(np.float64) * 1e6).astype(np.int64)
        x = events[:, 1].astype(np.int32)
        y = events[:, 2].astype(np.int32)
        on = events[:, 3] > 0
        valid = (x >= 0) & (x < self.sizex) & (y >= 0) & (y < self.sizey)
        if not np.any(valid):
            return
        timestamp_us = timestamp_us[valid]
        x = x[valid]
        y = y[valid]
        on = on[valid]
        if self.last_us is not None and timestamp_us[0] < self.last_us:
            timestamp_us[0] = self.last_us
        np.maximum.accumulate(timestamp_us, out=timestamp_us)
        self.last_us = int(timestamp_us[-1])

        store = dv.EventStore()
        push = store.push_back
        for timestamp, xe, ye, polarity in zip(
                timestamp_us.tolist(), x.tolist(), y.tolist(), on.tolist()):
            push(int(timestamp), int(xe), int(ye), bool(polarity))
        self.writer.writeEvents(store)

        n_on = int(np.count_nonzero(on))
        self.numOnEvents += n_on
        self.numOffEvents += int(on.size - n_on)
        self.numEventsWritten += int(on.size)
