
from __future__ import print_function
import h5py
import numpy as np
import time
import multiprocessing as mp
import queue

SIZE_INC = 2048
CHUNK_SIZE = 128


class HDF5(mp.Process):
    '''
    Creates a hdf5 file with datasets of specified types.
    Provides an append method.
    '''
    def __init__(self, filename='rec.hdf5', tables={}, bufsize=2048*16, chunksize=0, mode='w-', compression=None):
        super(HDF5, self).__init__()
        self.compression = compression
        self.fname = filename
        self.datasets = {}
        self.outbuffers = {}
        self.ndims = {}
        self.chunk_size = chunksize or CHUNK_SIZE
        self.tables = tables
        self.q = mp.Queue(bufsize)
        self.maxsize = self.q._maxsize
        self.exit = mp.Event()
        self.fmode = mode
        #self.daemon = True
        self.start()

    def init_ds(self):
        self.f = h5py.File(self.fname, self.fmode)
        self.create_datasets(self.tables, compression=self.compression)
        self.ptrs = {k: 0 for k in self.datasets}
        self.size = {k: SIZE_INC for k in self.datasets}

    def run(self):
        self.init_ds()
        f = file('datasets_ioerrors.txt', 'a')
        while not self.exit.is_set() or not self.q.empty():
            try:
                res = self.q.get(False, 1e-3)
                self._save(res)
            except Queue.Empty:
                pass
            except IOError:
                print('IOError, continuing')
                f.write(str(res))
                pass
            except KeyboardInterrupt:
                #print('datasets.run got interrupt')
                self.exit.set()
        f.cleanup()
        self.close()

    def create_datasets(self, tables, compression=None):
        for tname, ttype in tables.iteritems():
            tname_split = tname.split('/')
            if len(tname_split) > 1:
                grpname, subtname = tname_split
                if grpname not in self.f:
                    rnode = self.f.create_group(grpname)
                else:
                    rnode = self.f[grpname]
            else:
                subtname = tname
                rnode = self.f
            tname = tname.replace('/', '_')
            extra_shape = ()
            self.ndims[tname] = 1
            if isinstance(ttype, (tuple, list)):
                extra_shape = ttype[1]
                ttype = ttype[0]
                self.ndims[tname] += 1
            print(tname)
            self.datasets[tname] = rnode.create_dataset(
                subtname,
                (SIZE_INC,) + extra_shape,
                maxshape=(None,) + extra_shape,
                chunks=(self.chunk_size,) + extra_shape,
                dtype=ttype,
                compression=compression)
            self.outbuffers[tname] = []

    def save(self, data):
        try:
            self.q.put_nowait(data)
        except Queue.Full:
            raise Queue.Full('dataset buffer overflow')

    def _save(self, data):
        for col,val in data.iteritems():
            self.outbuffers[col].append(val)
            if len(self.outbuffers[col]) == self.chunk_size:
                self[col][self.ptrs[col]:self.ptrs[col] + self.chunk_size] = \
                        self._get_outbuf(col)
                self.outbuffers[col] = []
                self.ptrs[col] += self.chunk_size
            if self.ptrs[col] == self.size[col]:
                self.size[col] += SIZE_INC
                self[col].resize(self.size[col], axis=0)

    def _get_outbuf(self, col):
        if self.ndims[col] > 1:
            return np.array(self.outbuffers[col])
        else:
            return self.outbuffers[col]

    def __getitem__(self, key):
        return self.datasets[key]

    def close(self):
        self.exit.set()
        self.f.flush()
        self.f.close()
        self.q.close()
        self.q.join_thread()
        print('\nclosed output file')
