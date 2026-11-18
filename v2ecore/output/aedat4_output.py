import logging
import struct
from collections import deque

import cv2
import dv_processing as dv
import flatbuffers
import lz4.frame
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


_AEDAT4_VERSION = b"#!AER-DAT4.0\r\n"
# dv-processing 2.x writes a FileDataTable with byte offset, size, and
# element count, but no packet times. jAER reads timestampStart/timestampEnd
# and treats a missing field as 0, so every packet sits at time 0 and the
# playback clock does not move.
_COMPRESSION_LZ4 = 1
_COMPRESSION_LZ4_HIGH = 2


def _i32(buf, offset):
    return struct.unpack_from("<i", buf, offset)[0]


def _u16(buf, offset):
    return struct.unpack_from("<H", buf, offset)[0]


def _u32(buf, offset):
    return struct.unpack_from("<I", buf, offset)[0]


def _i64(buf, offset):
    return struct.unpack_from("<q", buf, offset)[0]


def _vtable_rel(buf, table, field_id):
    vtable = table - _i32(buf, table)
    vlen = _u16(buf, vtable)
    byte = 4 + 2 * field_id
    if byte >= vlen:
        return 0
    return _u16(buf, vtable + byte)


def _field_i64(buf, table, field_id):
    rel = _vtable_rel(buf, table, field_id)
    if not rel:
        return 0
    return _i64(buf, table + rel)


def _size_prefixed_root(buf):
    """Root table of a size-prefixed FlatBuffer."""
    return 4 + _u32(buf, 4)


def _header_layout(data):
    """Return (compression, dataTablePosition, first packet offset)."""
    if not data.startswith(_AEDAT4_VERSION):
        raise ValueError("not an AEDAT-4 file")
    header_size = _i32(data, len(_AEDAT4_VERSION))
    flat = len(_AEDAT4_VERSION) + 4
    root = flat + _u32(data, flat)
    comp_rel = _vtable_rel(data, root, 0)
    pos_rel = _vtable_rel(data, root, 1)
    compression = _i32(data, root + comp_rel) if comp_rel else 0
    table_pos = _i64(data, root + pos_rel) if pos_rel else -1
    return compression, table_pos, flat + header_size


def _decompress_packet(payload, compression):
    if compression == 0:
        return payload
    if compression in (_COMPRESSION_LZ4, _COMPRESSION_LZ4_HIGH):
        return lz4.frame.decompress(payload)
    raise ValueError(
        "AEDAT-4 compression {} has no timestamp index rewriter".format(compression))


def _packet_time_range(payload, compression):
    """Return (t_start_us, t_end_us, num_elements) from one packet payload."""
    decoded = _decompress_packet(payload, compression)
    if len(decoded) < 12:
        return 0, 1, 0
    kind = decoded[8:12]
    root = _size_prefixed_root(decoded)
    if kind == b"FRME":
        start = _field_i64(decoded, root, 0)
        end_exposure = _field_i64(decoded, root, 4)
        exposure = _field_i64(decoded, root, 11)
        end = end_exposure if end_exposure > start else start + max(exposure, 0)
        if end <= start:
            end = start + 1
        return start, end, 1
    if kind == b"EVTS":
        rel = _vtable_rel(decoded, root, 0)
        if not rel:
            return 0, 1, 0
        vector_pos = root + rel
        vector = vector_pos + _u32(decoded, vector_pos)
        count = _u32(decoded, vector)
        if count <= 0:
            return 0, 1, 0
        base = vector + 4
        start = _i64(decoded, base)
        end = _i64(decoded, base + 16 * (count - 1))
        if end < start:
            end = start
        if end == start:
            end = start + 1
        return start, end, count
    return 0, 1, 0


def _build_file_data_table(rows):
    """FlatBuffer FileDataTable with the timestamp fields jAER reads.

    Each row is (payload_offset, stream_id, payload_size, count, t_start, t_end).
    """
    builder = flatbuffers.Builder(1024 + 48 * len(rows))
    offsets = []
    for payload_offset, stream_id, payload_size, count, t_start, t_end in rows:
        builder.Prep(4, 8)
        builder.PrependInt32(int(payload_size))
        builder.PrependInt32(int(stream_id))
        info = builder.Offset()
        builder.StartObject(5)
        # The struct has to be recorded before any field that moves the head.
        builder.PrependStructSlot(1, info, 0)
        builder.PrependInt64Slot(0, int(payload_offset), 0)
        builder.PrependInt64Slot(2, int(count), 0)
        builder.PrependInt64Slot(3, int(t_start), 0)
        builder.PrependInt64Slot(4, int(t_end), 0)
        offsets.append(builder.EndObject())
    builder.StartVector(4, len(offsets), 4)
    for offset in reversed(offsets):
        builder.PrependUOffsetTRelative(offset)
    vector = builder.EndVector()
    builder.StartObject(1)
    builder.PrependUOffsetTRelativeSlot(0, vector, 0)
    root = builder.EndObject()
    builder.Finish(root, b"FTAB")
    flat = bytes(builder.Output())
    return struct.pack("<i", len(flat)) + flat


def rewrite_aedat4_time_index(path):
    """Fill packet timestampStart/timestampEnd in the FileDataTable.

    dv-processing leaves those fields out. jAER then places every frame and
    every event packet at time 0, so the clock stays put and frames do not
    meet the events. The times written here are the ones already stored in
    the packet payloads.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    compression, table_pos, packet_pos = _header_layout(data)
    if table_pos < 0 or table_pos > len(data):
        raise ValueError("AEDAT-4 file {} has no FileDataTable".format(path))
    rows = []
    pos = packet_pos
    while pos + 8 <= table_pos:
        stream_id = _i32(data, pos)
        payload_size = _i32(data, pos + 4)
        payload_at = pos + 8
        if payload_size < 0 or payload_at + payload_size > table_pos:
            break
        t_start, t_end, count = _packet_time_range(
            data[payload_at:payload_at + payload_size], compression)
        rows.append((payload_at, stream_id, payload_size, count, t_start, t_end))
        pos = payload_at + payload_size
    if not rows:
        return 0, 0
    table = _build_file_data_table(rows)
    if compression in (_COMPRESSION_LZ4, _COMPRESSION_LZ4_HIGH):
        blob = lz4.frame.compress(table)
    elif compression == 0:
        blob = table
    else:
        raise ValueError(
            "AEDAT-4 compression {} has no timestamp index rewriter".format(
                compression))
    with open(path, "r+b") as handle:
        handle.seek(table_pos)
        handle.write(blob)
        handle.truncate()
    t0 = min(row[4] for row in rows)
    t1 = max(row[5] for row in rows)
    return t0, t1


def jaer_aedat4_source(output_width: int, output_height: int) -> str:
    """Camera name stored in the AEDAT-4 infoNode for jAER chip detection."""
    source = _JAER_SOURCES.get((output_width, output_height))
    if source is not None:
        return source
    return f"V2E_{output_width}x{output_height}"


class AEDat4Output:
    """Write DVS events as an iniVation AEDAT-4.0 file that jAER can play.

    With write_frames, the file is a DAVIS recording: an events stream and a
    frames stream (plus the empty IMU and trigger streams DAVIS files declare).
    jAER plays that as a Davis346 when the source name is DAVIS346_*.
    """

    def __init__(self, filepath: str, output_width: int, output_height: int,
                 write_frames: bool = False):
        if output_width is None or output_height is None:
            raise ValueError(
                "AEDAT-4.0 output needs output_width and output_height. "
                "Set them or pass a camera size such as --dvs346.")
        self.filepath = filepath
        self.sizex = int(output_width)
        self.sizey = int(output_height)
        self.write_frames = bool(write_frames)
        self.numEventsWritten = 0
        self.numOnEvents = 0
        self.numOffEvents = 0
        self.numFramesWritten = 0
        self.last_us = None
        self._warned_noise_label = False
        # Frames wait here until events reach their timestamp, so a frame is
        # packed next to the events it belongs with instead of in one block
        # at the start of the file.
        self._ready_frames = deque()  # (image, timestamp_us, exposure_us)
        self._holding_frame = None  # (image, timestamp_us) until exposure is known
        self._held_events = []  # event batches waiting on _holding_frame
        self._last_exposure_us = None
        self.source = jaer_aedat4_source(self.sizex, self.sizey)
        logger.info(
            "opening AEDAT-4.0 output file {} ({}x{}, source {}{})".format(
                filepath, self.sizex, self.sizey, self.source,
                ", with monochrome frames" if self.write_frames else ""))
        if self.source.startswith("V2E_"):
            logger.warning(
                "size {}x{} has no standard jAER chip. "
                "Open the file in jAER and select the chip class yourself.".format(
                    self.sizex, self.sizey))

        resolution = (self.sizex, self.sizey)
        if self.write_frames:
            # DAVISConfig declares events, frames, imu, and triggers, which is
            # the stream set jAER expects from a DAVIS recording.
            config = dv.io.MonoCameraWriter.DAVISConfig(
                self.source, resolution, dv.CompressionType.LZ4)
        else:
            config = dv.io.MonoCameraWriter.EventOnlyConfig(
                self.source, resolution, dv.CompressionType.LZ4)
        self.writer = dv.io.MonoCameraWriter(filepath, config)

    def cleanup(self):
        self.close()

    def close(self):
        if self.writer is None:
            return
        self._finish_holding_frame()
        self._drain_held_events()
        while self._ready_frames:
            self._write_ready_frame()
        if self.write_frames:
            logger.info(
                "Closing {} after writing {} events ({} on, {} off) "
                "and {} frames".format(
                    self.filepath,
                    EngNumber(self.numEventsWritten),
                    EngNumber(self.numOnEvents),
                    EngNumber(self.numOffEvents),
                    EngNumber(self.numFramesWritten)))
        else:
            logger.info(
                "Closing {} after writing {} events ({} on, {} off)".format(
                    self.filepath,
                    EngNumber(self.numEventsWritten),
                    EngNumber(self.numOnEvents),
                    EngNumber(self.numOffEvents)))
        # Dropping the writer flushes the background IO thread before the
        # timestamp index is rewritten.
        self.writer = None
        try:
            t0, t1 = rewrite_aedat4_time_index(self.filepath)
            logger.info(
                "AEDAT-4 packet times {} to {} s".format(
                    EngNumber(t0 * 1e-6), EngNumber(t1 * 1e-6)))
        except Exception:
            logger.exception(
                "Could not write AEDAT-4 packet times into {}".format(self.filepath))

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

        if (self.write_frames and self._holding_frame is not None
                and int(timestamp_us[0]) >= self._holding_frame[1]):
            # The frame's exposure is the gap to the next frame, which has
            # not arrived. Hold this batch so the frame is written first.
            self._held_events.append((timestamp_us, x, y, on))
            return
        self._emit_events_with_frames(timestamp_us, x, y, on)

    def appendFrame(self, frame, timestamp_s: float, exposure_s: float = None):
        """Queue one monochrome frame on the same clock as the DVS events.

        The frame is written into the file when events reach its timestamp,
        not in a block before the events. exposure_s is the time until the
        next frame. When it is omitted, the gap to the next appendFrame call
        is used.

        Parameters
        ----------
        frame: np.ndarray
            Grayscale image, shape (height, width), uint8 or a float in 0..255.
            Color frames are converted to luma. Pixel [0, 0] is the top left,
            which is the OpenCV order jAER expects from a DV DAVIS file.
        timestamp_s: float
            Frame time in seconds.
        exposure_s: float or None
            Exposure in seconds. None defers it until the next frame.
        """
        if self.writer is None or not self.write_frames:
            return
        image = np.copy(self._mono8(frame))
        ts_us = int(np.rint(float(timestamp_s) * 1e6))
        if ts_us < 0:
            ts_us = 0
        if self._holding_frame is not None:
            self._promote_holding_frame(ts_us)
        if exposure_s is None:
            self._holding_frame = (image, ts_us)
        else:
            exposure_us = int(np.rint(float(exposure_s) * 1e6))
            if exposure_us <= 0:
                exposure_us = self._last_exposure_us or 1
            self._last_exposure_us = exposure_us
            self._ready_frames.append((image, ts_us, exposure_us))
        self._drain_held_events()

    def _promote_holding_frame(self, next_ts_us):
        image, ts_us = self._holding_frame
        exposure = int(next_ts_us) - int(ts_us)
        if exposure <= 0:
            exposure = self._last_exposure_us or 1
        self._last_exposure_us = exposure
        self._ready_frames.append((image, ts_us, exposure))
        self._holding_frame = None

    def _finish_holding_frame(self):
        if self._holding_frame is None:
            return
        image, ts_us = self._holding_frame
        exposure = self._last_exposure_us or 33333
        self._ready_frames.append((image, ts_us, int(exposure)))
        self._holding_frame = None

    def _drain_held_events(self):
        if not self._held_events:
            return
        held = self._held_events
        self._held_events = []
        for timestamp_us, x, y, on in held:
            self._emit_events_with_frames(timestamp_us, x, y, on)

    def _emit_events_with_frames(self, timestamp_us, x, y, on):
        """Write queued frames into the event stream at their own timestamps."""
        index = 0
        count = len(timestamp_us)
        while index < count:
            if self._ready_frames and self._ready_frames[0][1] <= int(timestamp_us[index]):
                self._write_ready_frame()
                continue
            if self._ready_frames:
                frame_ts = self._ready_frames[0][1]
                end = index
                while end < count and int(timestamp_us[end]) < frame_ts:
                    end += 1
                if end == index:
                    self._write_ready_frame()
                    continue
            else:
                end = count
            self._emit_event_slice(
                timestamp_us[index:end], x[index:end], y[index:end], on[index:end])
            index = end

    def _emit_event_slice(self, timestamp_us, x, y, on):
        if self.writer is None or len(timestamp_us) == 0:
            return
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

    def _write_ready_frame(self):
        image, ts_us, exposure_us = self._ready_frames.popleft()
        self._write_frame(image, ts_us, exposure_us)

    def _write_frame(self, image, ts_us, exposure_us):
        exposure_us = int(max(1, exposure_us))
        self._last_exposure_us = exposure_us
        packet = dv.Frame(
            int(ts_us), exposure_us, 0, 0, image, dv.FrameSource.SENSOR)
        self.writer.writeFrame(packet)
        self.numFramesWritten += 1

    def _mono8(self, frame) -> np.ndarray:
        img = np.asarray(frame)
        if img.ndim == 3:
            channels = img.shape[2]
            if channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif channels == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                img = img[:, :, 0]
        if img.shape != (self.sizey, self.sizex):
            raise ValueError(
                "DAVIS frame shape {} does not match AEDAT-4 size {}x{}".format(
                    img.shape, self.sizex, self.sizey))
        if img.dtype != np.uint8:
            img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img)
