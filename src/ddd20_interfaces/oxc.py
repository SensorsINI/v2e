
'''
Recorder for DAVIS + OpenXC data
Author: J. Binas <jbinas@gmail.com>, 2017

This software is released under the
GNU LESSER GENERAL PUBLIC LICENSE Version 3.
'''

from __future__ import absolute_import, print_function

import time, sys
import multiprocessing as mp
import numpy as np
from openxc.tools import dump as oxc
import queue

class Monitor(mp.Process):
    def __init__(self, bufsize=256):
        super(Monitor, self).__init__()
        arguments = oxc.parse_options()
        source_class, source_kwargs = oxc.select_device(arguments)
        self.source = source_class(callback=self.receive, **source_kwargs)
        self.q = mp.Queue(bufsize)
        self.qsize = 0
        self.maxsize = self.q._maxsize
        self.exit = mp.Event()
        #self.daemon = True
        self.start()

    def run(self):
        self.source.start()
        #self.source.join()
        while not self.exit.is_set():
            try:
                time.sleep(1e-5)
            except KeyboardInterrupt:
                self.source.stop()
                self.exit.set()

    def receive(self, message, **kwargs):
        ''' receive single message from interface '''
        if self.exit.is_set():
            return
        message['timestamp'] = int(time.time() * 1e6)
        try:
            self.q.put_nowait(message)
            self.qsize = max(self.qsize, self.q.qsize())
        except Queue.Full:
            raise Queue.Full('vi buffer overflow')

    def get(self):
        ''' get one message from buffer '''
        return self.q.get_nowait() if not self.q.empty() else False


if __name__ == '__main__':

    vi = Monitor()

    i = 0
    t = time.time()
    while True:
        res = vi.get()
        if res:
            print(res)
        if time.time() - t > 1:
            print('\npolling at', i / (time.time() - t), 'Hz\n')
            i = 0
            t = time.time()
        i += 1

