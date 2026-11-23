# Frames from a 3D engine, or any other process, on stdin.
#
# uv run v2e --pipeframes --dvs346 --overwrite --no_preview --output_folder=output/pipe-frames
#
# --pipeframes is --synthetic_input=scripts.pipe_frames --disable_slomo.
# SuperSloMo does not run. Event time is the time in each record.
#
# Each record is little-endian:
#   float64  simulated time in seconds, strictly increasing
#   uint32   number of payload bytes
#   payload  PNG, JPEG, or BMP, gray or color
# Close stdin after the last frame.
#
# The default photoreceptor cutoff is 200 Hz (tau = 1/(2*pi*200) ~= 0.80 ms).
# The frame interval must be at most 0.3*tau, so the source rate must be at
# least (2*pi*cutoff_hz)/0.3. At 200 Hz that is 4189 frames per second of
# simulated time. --cutoff_hz=0 turns the filter off.

import math
import os
import struct
import sys
from typing import Optional, Tuple

import cv2
import numpy as np

from v2ecore.base_synthetic_input import base_synthetic_input
from v2ecore.v2e_utils import min_lowpass_sample_rate_hz, v2e_quit
import logging

logger = logging.getLogger(__name__)

_TIME = struct.Struct("<d")
_LENGTH = struct.Struct("<I")


def pack_frame(time_s: float, image: np.ndarray) -> bytes:
    """One stdin record: time, PNG length, PNG bytes.

    image is grayscale or BGR uint8 in OpenCV layout (row, column).
    """
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("could not encode a frame as PNG")
    payload = encoded.tobytes()
    return _TIME.pack(float(time_s)) + _LENGTH.pack(len(payload)) + payload


def _read_exact(stream, n: int) -> Optional[bytes]:
    """Read n bytes, or None at a clean end of stream."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            if len(buf) == 0:
                return None
            raise EOFError(f"stdin ended after {len(buf)} of {n} expected bytes")
        buf += chunk
    return bytes(buf)


def _cutoff_hz_for_run(parent_args) -> float:
    """Cutoff the emulator will use, including --dvs_params overrides."""
    if parent_args is None:
        return 200.0
    mode = getattr(parent_args, "dvs_params", None)
    if mode == "clean":
        return 0.0
    if mode == "noisy":
        return 30.0
    cutoff = getattr(parent_args, "cutoff_hz", 200.0)
    if cutoff is None:
        return 200.0
    return float(cutoff)


class pipe_frames(base_synthetic_input):
    """Pull rendered frames from stdin. The class name matches the module name."""

    def __init__(self, width: int = 346, height: int = 260, avi_path: Optional[str] = None,
                 preview=False, arg_list=None, parent_args=None) -> None:
        if width is None or height is None:
            logger.error(
                "pipe frames need a DVS size. Pass --dvs346, or set both "
                "--output_width and --output_height.")
            v2e_quit(1)
        super().__init__(width, height, avi_path, preview, arg_list, parent_args)
        self._stdin = sys.stdin.buffer
        if sys.platform == "win32":
            import msvcrt
            msvcrt.setmode(self._stdin.fileno(), os.O_BINARY)
        cutoff = _cutoff_hz_for_run(parent_args)
        if cutoff <= 0:
            logger.info(
                "Photoreceptor lowpass is off. "
                "Frame times on stdin are the DVS timestamps.")
        else:
            min_hz = math.ceil(min_lowpass_sample_rate_hz(cutoff))
            tau_ms = 1000.0 / (2 * math.pi * cutoff)
            logger.info(
                f"Photoreceptor cutoff is {cutoff:g} Hz (tau={tau_ms:.2f} ms). "
                f"Send frames at {min_hz} Hz or faster of simulated time "
                f"so each frame interval stays within 0.3 tau.")
        logger.info(
            "Waiting for frames on stdin. "
            "Each record is little-endian float64 time, uint32 byte count, "
            "then PNG, JPEG, or BMP bytes. Close stdin to finish.")

    def total_frames(self):
        """Length is unknown until stdin closes."""
        return None

    def next_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Return the next (frame, time), or (None, time) when stdin closes."""
        try:
            time_bytes = _read_exact(self._stdin, _TIME.size)
        except EOFError as e:
            logger.error(f"truncated frame time on stdin: {e}")
            return None, self.time
        if time_bytes is None:
            if self.frame_number == 0:
                logger.error("no frames on stdin")
            else:
                logger.info(f"stdin closed after {self.frame_number} frames")
            return None, self.time

        time_s = _TIME.unpack(time_bytes)[0]
        if self.frame_number > 0 and time_s <= self.time:
            logger.error(
                f"frame {self.frame_number} time {time_s} s is not later than "
                f"the previous time {self.time} s")
            v2e_quit(1)

        try:
            length_bytes = _read_exact(self._stdin, _LENGTH.size)
        except EOFError as e:
            logger.error(f"truncated frame length on stdin: {e}")
            return None, self.time
        if length_bytes is None:
            logger.error("stdin ended before the frame byte count")
            return None, self.time
        nbytes = _LENGTH.unpack(length_bytes)[0]
        if nbytes <= 0:
            logger.error(f"frame {self.frame_number} has an empty image payload")
            v2e_quit(1)

        try:
            payload = _read_exact(self._stdin, nbytes)
        except EOFError as e:
            logger.error(f"truncated image on stdin: {e}")
            return None, self.time
        if payload is None:
            logger.error("stdin ended in the middle of an image")
            return None, self.time

        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            logger.error(
                f"frame {self.frame_number} is not a PNG, JPEG, or BMP "
                f"({nbytes} bytes)")
            v2e_quit(1)
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        self.time = float(time_s)
        self.frame_number += 1
        self.pix_arr = frame
        if self.preview:
            cv2.imshow(self.cv2name, frame)
            cv2.waitKey(1)
        self.write_video_frame(frame)
        return frame, self.time
