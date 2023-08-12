import numpy as np
import logging

import torch
from engineering_notation import EngNumber  # only from pip
import atexit
import struct

# check https://gitlab.com/inivation/dv/dv-processing to install dv-processing-python
import dv_processing as dv

from v2ecore.v2e_utils import v2e_quit

logger = logging.getLogger(__name__)


class AEDat4Output:
    """
    outputs AEDAT-4.0 jAER format DVS data from v2e
    """


    def __init__(self, filepath: str, output_width=640, output_height=480):
        self.filepath = filepath
        self.numEventsWritten = 0
        self.numOnEvents=0
        self.numOffEvents=0
        logging.info('opening AEDAT-4.0 output file {} in binary mode'.format(filepath))

        self.flipy = False 
        self.flipx = False 
        self.sizex = output_width
        self.sizey = output_height
         
        self.store = dv.EventStore()

        resolution = (640, 480)
        # Event only configuration
        config = dv.io.MonoCameraWriter.EventOnlyConfig("DVXplorer_sample", resolution)

        # Create the writer instance, it will only have a single event output stream.
        self.writer = dv.io.MonoCameraWriter(filepath, config)

    def cleanup(self):
        self.close()

    def close(self):
        if self.writer:
            logger.info("Closing {} after writing {} events ({} on, {} off)".
                        format(self.filepath,
                               EngNumber(self.numEventsWritten),
                               EngNumber(self.numOnEvents),
                               EngNumber(self.numOffEvents),
                               ))
            
            self.writer.writeEvents(self.store)
            self.writer = None

    def appendEvents(self, events: np.ndarray, signnoise_label:np.ndarray=None ):
        """Append events to AEDAT-4.0 output

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

        if self.writer is None:
            return

        if len(events) == 0:
            return
        n = events.shape[0]
        for event in events:
            t = int(event[0] * 1e6)
            x = int(event[1])
            if self.flipx: x = (self.sizex - 1) - x  # 0 goes to sizex-1
            y = int(event[2])
            if self.flipy: y = (self.sizey - 1) - y
            p = int((event[3] + 1) / 2) # 0=off, 1=on

            try: 
                 self.store.push_back(t, x, y, p)
            except RuntimeError as e:
                 logger.warning('caught exception event {} to store'.format(e))

            if p==1: self.numOnEvents+=1
            else: self.numOffEvents+=1
            self.numEventsWritten += 1
        
        # logger.info('wrote {} events'.format(n))


if __name__ == '__main__':
    class AEDat4OutputTt():
        f = AEDat4Output('aedattest.aedat4')
        e = [[1, 400, 0, 0], [2, 0, 400, 0], [3, 300, 400, 0], [4, 400, 300, 1], [5, 400, 300, 0]]
        ne = np.array(e)
        eventsNum = 2000 * 5
        nne = np.tile(ne, (int(eventsNum/5), 1))
        nne[:, 0] = np.arange(1, eventsNum + 1)
        f.appendEvents(nne)
        print('wrote {} events'.format(nne.shape[0]))
        f.close()