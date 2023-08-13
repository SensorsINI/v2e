import numpy as np
import logging

import torch
from engineering_notation import EngNumber  # only from pip
import atexit
import struct

from v2ecore.v2e_utils import v2e_quit

logger = logging.getLogger(__name__)


class AEDat2Output:
    """
    outputs AEDAT-2.0 jAER format DVS data from v2e
    """

    SUPPORTED_SIZES=((346,260),(240,180),(640,480))

    def __init__(self, filepath: str, output_width=346, output_height=260, label_signal_noise:bool=False):
        """

        Parameters
        ----------
        filepath - full path to output AEDAT file, including ".aedat" or ".aedat2" extension
        output_width - the width of output address space
        output_height - the height of output address space
       :param label_signal_noise: set True to label noise events as 'special'
        """
        self.filepath = filepath
        self.file=None
        # set bit 10 (11'th bit) to 1 to mark special event. Bit number 11 is OFF or ON bit.
        self.noise_special_event_bit= 1 << 10 # https://gitlab.com/inivation/inivation-docs/-/blob/master/Software%20user%20guides/AEDAT_file_formats.md#dvs
        self.label_signal_noise=label_signal_noise
        if self.label_signal_noise:
            logger.info(f'labeling noise events as special events by ORing address with bit 11 set to 1 using bit pattern "{self.noise_special_event_bit:032_b}"')
        # edit below to match https://gitlab.com/inivation/inivation-docs/-/blob/master/Software%20user%20guides/AEDAT_file_formats.md
        # see AEDAT-2.0 format section
        if output_width==346 and output_height==260:
            # DAVIS
            # In the 32-bit address:
            # bit 32 (1-based) being 1 indicates an APS sample
            # bit 11 (1-based) being 1 indicates a special event
            # bits 11 and 32 (1-based) both being zero signals a polarity event
            self.yShiftBits = 22
            self.xShiftBits = 12
            self.polShiftBits = 11  # see https://gitlab.com/inivation/inivation-docs/-/blob/master/Software%20user%20guides/AEDAT_file_formats.md
            self.sizex = output_width
            self.sizey = output_height
            self.flipy = True  # v2e uses computer vision matrix printing convention of UL pixel being 0,0, but jAER uses original graphics and graphing convention that 0,0 is LL
            self.flipx = True # not 100% sure why this is needed. Observed for tennis example
        elif output_width==240 and output_height==180:
            # DAVIS
            # In the 32-bit address:
            # bit 32 (1-based) being 1 indicates an APS sample
            # bit 11 (1-based) being 1 indicates a special event
            # bits 11 and 32 (1-based) both being zero signals a polarity event
            self.yShiftBits = 22
            self.xShiftBits = 12
            self.polShiftBits = 11  # see https://gitlab.com/inivation/inivation-docs/-/blob/master/Software%20user%20guides/AEDAT_file_formats.md#dvs-or-aps
            self.sizex = output_width
            self.sizey = output_height
            self.flipy = True  # v2e uses computer vision matrix printing convention of UL pixel being 0,0, but jAER uses original graphics and graphing convention that 0,0 is LL
            self.flipx = True # not 100% sure why this is needed. Observed for tennis example
        elif output_width==640 and output_height==480: # jAER chip DVS640 final int XSHIFT = 1, XMASK = 0b11_1111_1111<<XSHIFT , YSHIFT = 11, YMASK = 0b11_1111_1111<<YSHIFT;
            # DAVIS
            # In the 32-bit address:
            # bit 32 (1-based) being 1 indicates an APS sample
            # bit 11 (1-based) being 1 indicates a special event
            # bits 11 and 32 (1-based) both being zero signals a polarity event
            self.yShiftBits = 11
            self.xShiftBits = 1
            self.polShiftBits = 0  # see jAER DVS640 class https://github.com/SensorsINI/jaer/blob/master/src/ch/unizh/ini/jaer/chip/retina/DVS640.java
            self.sizex = output_width
            self.sizey = output_height
            self.flipy = True  # v2e uses computer vision matrix printing convention of UL pixel being 0,0, but jAER uses original graphics and graphing convention that 0,0 is LL
            self.flipx = True # not 100% sure why this is needed. Observed for tennis example
        else:
            err_string=f'AEDAT-2.0 output width={output_width} height={output_height} not supported; add your camera to {__name__} or use one of the predefined DVS cameras, e.g. --dvs346 or --dvs240 that have sizes self.SUPPORTED_SIZES={self.SUPPORTED_SIZES}'
            raise ValueError(err_string)

        self.numEventsWritten = 0
        self.numOnEvents=0
        self.numOffEvents=0
        logging.info('opening AEDAT-2.0 output file {} in binary mode'.format(filepath))
        try:
            self.file = open(filepath, 'wb')
            self._writeHeader()
            atexit.register(self.cleanup)
            logger.info('opened {} for DVS output data for jAER'.format(filepath))
        except OSError as err:
            logger.error('caught {}:\n  could not open {} for writing; maybe jAER has it open?'.format(err,filepath))
            v2e_quit(1)

    def cleanup(self):
        self.close()

    def close(self):
        if self.file:
            logger.info("Closing {} after writing {} events ({} on, {} off)".
                        format(self.filepath,
                               EngNumber(self.numEventsWritten),
                               EngNumber(self.numOnEvents),
                               EngNumber(self.numOffEvents),
                               ))
            self.file.close()
            self.file = None

    def _writeHeader(self):
        import datetime, time, getpass
        # CRLF \r\n is needed to not break header parsing in jAER
        date = datetime.datetime.now().strftime('# Creation time: %I:%M%p %B %d %Y\r\n')  # Tue Jan 26 13:57:06 CET 2016
        time = '# Creation time: System.currentTimeMillis() {}\r\n'.format(int(time.time() * 1000.))
        user = '# User name: {}\r\n'.format(getpass.getuser())
        if self.label_signal_noise:
            sn_comment='# noise events are labeled as addressed external input events when the --label_signal_noise option is selected for output\r\n'
        else:
            sn_comment=''
        # IMPORTANT, use \r\n to terminate lines!!!! otherwise the whole file will be corrupted
        header = ('#!AER-DAT2.0\r\n',
                  '# This is a raw AE data file created by AEDat2Output in v2e (see https://github.com/SensorsINI/v2e) as specified at https://inivation.com/support/software/fileformat/#aedat-20\r\n',
                  '# Data format is int32 address, int32 timestamp (8 bytes total), repeated for each event\r\n',
                  '# Timestamps tick is 1 us\r\n',
                  sn_comment,
                  date, time,
                  user,
                  )
        for s in header:
            bytes = s.encode('UTF-8')
            self.file.write(bytes)

    def appendEvents(self, events: np.ndarray, signnoise_label:np.ndarray=None ):
        """Append events to AEDAT-2.0 output

          Parameters
          ----------
          events: np.ndarray if any events, else None
              [N, 4], each row contains [timestamp, x coordinate, y coordinate, sign of event (+1 ON, -1 OFF)].
              NOTE x,y, NOT y,x.
          signnoise: np.ndarray
            [N] each entry is 1 for signal or 0 for noise

          Returns
          -------
          None
          """

        if self.file is None:
            return

        if len(events) == 0:
            return
        n = events.shape[0]
        t = (1e6 * events[:, 0]).astype(np.int32)   # to us from seconds
        if np.any(np.diff(t)<0):
            logger.warning('nonmonontoic timestamp')
        x = events[:, 1].astype(np.int32)
        if self.flipx: x = (self.sizex - 1) - x  # 0 goes to sizex-1
        y = events[:, 2].astype(np.int32)
        if self.flipy: y = (self.sizey - 1) - y
        p = ((events[:, 3] + 1) / 2).astype(np.int32) # 0=off, 1=on

        a = (x << self.xShiftBits | y << self.yShiftBits | p << self.polShiftBits)
        if self.label_signal_noise and not signnoise_label is None:
            noise_mask=np.logical_not(signnoise_label) # true or 1 for noise event elements
            # print(f'\naddr before noise mask {a[0]:032_b}')
            a[np.where(noise_mask)]|=self.noise_special_event_bit # set the special event bits 11 and 10 to 1 for noise events
            # print(f'addr after noise mask  {a[0]:032_b}')
        out = np.empty(2 * n, dtype=np.int32) # for n events allocate 2n int32 because file holds int32 values addr0, timestamp0, addr1, timestamp1
        out[0::2] = a  # put addresses to even positions of out
        out[1::2] = t  # put timestamps to odd positions
        bytes=out.byteswap().tobytes(order='C') # produce c-style bytes in Java big endian format for jAER
        if self.numEventsWritten==0:
            #make sure we don't write comment char as first event
            chopped=False
            while bytes[0:1].decode('utf-8',errors='ignore')=='#':
                logger.warning('first event would write a # comment char, dropping it')
                bytes=bytes[8:]
                chopped=True
        # now out is numpy array holding int32 timestamp,address array, i.e. ts0, ad0, ts1, ad1, etc
        self.file.write(bytes)  # java is big-endian, so  byteswap to get this
        self.numEventsWritten += n
        onCount=np.count_nonzero(p)
        offCount=n-onCount
        self.numOnEvents+=onCount
        self.numOffEvents+=offCount
        self.file.flush()
        # logger.info('wrote {} events'.format(n))

# class AEDat2OutputTest():
#     f = AEDat2Output('aedattest.aedat')
#     e = [[0., 0, 0, 0], [1e-6, 0, 0, 1], [2e-6, 1, 0, 0]]
#     ne = np.array(e)
#     f.appendEvents(ne)
#     f.close()
