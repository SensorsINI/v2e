import numpy as np
import logging
from engineering_notation import EngNumber  # only from pip
import atexit

logger = logging.getLogger(__name__)

class DVSTextOutput:
    '''
    outputs text format DVS events to file according to events.txt format in http://rpg.ifi.uzh.ch/davis_data.html
    
    The RPG DVS text file datatset looks like this. Each line has (time(float s), x, y, polarity (0=off,1=on)
        
        0.000000000 33 39 1
        0.000011001 158 145 1
        0.000050000 88 143 0
        0.000055000 174 154 0
        0.000080001 112 139 1
        0.000123000 136 171 0
        0.000130001 173 90 0
        0.000139001 106 140 0
        0.000148001 192 79 1
    '''

    def __init__(self, filepath: str):
        self.filepath = filepath
        # edit below to match your device from https://inivation.com/support/software/fileformat/#aedat-20
        self.numEventsWritten = 0
        logging.info('opening text DVS output file {}'.format(filepath))
        self.file = open(filepath, 'w')
        self._writeHeader()
        atexit.register(self.cleanup)
        self.flipx=False # set both flipx and flipy to rotate TODO replace with rotate180
        self.flipy=False
        self.sizex=346
        self.sizey=260 # adjust to your needs

    def cleanup(self):
        self.close()

    def close(self):
        if self.file:
            logger.info("Closing {} after writing {} events".format(self.filepath, EngNumber(self.numEventsWritten)))
            self.file.close()
            self.file = None

    def _writeHeader(self):
        import datetime, time, getpass
        date = datetime.datetime.now().strftime('# Creation time: %I:%M%p %B %d %Y\n')  # Tue Jan 26 13:57:06 CET 2016
        time = '# Creation time: System.currentTimeMillis() {}\n'.format(int(time.time() * 1000.))
        user = '# User name: {}\n'.format(getpass.getuser())
        header = ('#!events.txt\n',
                  '# This is a text DVS created by v2e (see https://github.com/SensorsINI/v2e)\n',
                  '# Format is time (float s), x, y, polarity (0=off, 1=on) as specified at http://rpg.ifi.uzh.ch/davis_data.html\n',
                  date, time,
                  user
                  )
        for s in header:
            self.file.write(s)

    def appendEvents(self, events: np.ndarray):
        if self.file is None:
            raise Exception('output file closed already')

        if len(events) == 0:
            return
        n = events.shape[0]
        t = (events[:, 0]).astype(np.float)
        x = events[:, 1].astype(np.int32)
        if self.flipx: x = (self.sizex - 1) - x  # 0 goes to sizex-1
        y = events[:, 2].astype(np.int32)
        if self.flipy: y = (self.sizey - 1) - y
        p = ((events[:, 3] + 1) / 2).astype(np.int32) # go from -1/+1 to 0,1
        for i in range(n):
            self.file.write('{} {} {} {}\n'.format(t[i],x[i],y[i],p[i])) # todo there must be vector way
        self.numEventsWritten += n

# class DVSTextOutputTest: # test from src.output.ae_text_output import DVSTextOutputTest
#     f = DVSTextOutput('aedat-text-test.txt')
#     e = [[0., 0, 0, 0], [1e-6, 0, 0, 1], [2e-6, 1, 0, 0]]
#     ne = np.array(e)
#     f.appendEvents(ne)
#     e = [[3e-6, 0, 0, 1], [5e-6, 0, 0, 1], [9e-6, 1, 0, 0]]
#     ne = np.array(e)
#     f.appendEvents(ne)
#     f.close()
