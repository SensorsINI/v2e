"""
Reads DDD hdf5 dvs data and return aps frames + events.
@author: Zhe He, Tobi Delbruck
@contact:hezhehz@live.cn, tobi@ini.uzh.ch
@latest update: 2019-May-31
"""
import ctypes
import queue as queue
import time
import numpy as np
import logging
import h5py
from tqdm import tqdm
import multiprocessing as mp
from v2ecore.ddd20_utils.datasets import CHUNK_SIZE
from v2ecore.ddd20_interfaces.caer import unpack_data
from v2ecore.ddd20_interfaces import caer

logger = logging.getLogger(__name__)


class DDD20SimpleReader(object):
    '''
    Simple reader with no multiprocessing threads to read in DDD recording and
    extract data
    '''
    ETYPE_DVS = 'polarity_event'
    ETYPE_APS = 'frame_event'
    ETYPE_IMU = 'imu6_event'

    def __init__(self, fname, rotate180=True):
        """Init

        Parameters
        ----------
        fname: str
            path of input hdf5 file.
        rotate180: bool, True
            rotate input 180 deg (both frames and events)
        """
        logger.info('making reader for DDD recording '+str(fname))
        self.rotate180=rotate180
        self.f_in =h5py.File(fname, 'r')
        logger.info(str(fname)+' contains following keys')
        hasDavisData=False
        dvsKey='dvs'
        for key in self.f_in.keys():
            if key==dvsKey: hasDavisData=True
            print(key)
        if not hasDavisData: raise('file does not contain DAVIS data (key dvs)')

        dvsGroup=self.f_in[dvsKey]
        logger.info('group dvs contains following keys')
        for key in dvsGroup.keys():
            print(key)

        logger.info('group dvs contains following items')
        for item in dvsGroup.items():
            print(item)

        self.davisData=dvsGroup['data']
        logger.info('The DAVIS data has the shape '+str(self.davisData.shape))
        self.shape = None # the shape of the DVS/APS pixel array

        self.numPackets=self.davisData.shape[0] # start here, this is not actual size
        self.firstPacketNumber=0
        firstPacket=self.readPacket(self.firstPacketNumber) # first packet might contain data we can't parse
        while not firstPacket:
            self.firstPacketNumber+=1
            firstPacket = self.readPacket(self.firstPacketNumber)
        self.firstTimeS=firstPacket['timestamp']
        # the last packets in file are actually empty (some consequence of how file is written)
        # just go backards until we get a packet with some data
        lastPacket=self.readPacket(self.numPackets-1)
        while not lastPacket:
            self.numPackets-=1
            lastPacket = self.readPacket(self.numPackets-1)
        self.lastTimeS=lastPacket['timestamp']
        self.durationS= self.lastTimeS - self.firstTimeS
        logger.info('{} has {} packets with start time {:7.2f}s and end time {:7.2f}s (duration {:8.1f}s)'.format(
            fname, self.numPackets, self.firstTimeS, self.lastTimeS, self.durationS
        ))

        self.lastSearchTime=None # cache last search speed up search
        self.lastSearchPacketNumber=None

        # logger.info('Sample DAVIS data is the following')
        # i=0
        # for dat in self.davisData:
        #     headerDat=dat[1] # caer header
        #     header=caer.unpack_header(headerDat) # gets the packet type
        #     data = {'dvs_header': dat[1]} # put it to the dict as header
        #     data.update(caer.unpack_header(data['dvs_header'])) # update the dict?
        #     dat0=dat[0]   # timestamp of the packet?
        #     data['dvs_data'] = dat[2] # put the data payload, dvs_data refers to DAVIS camera data, can be frames or IMU data too
        #     data=caer.unpack_data(data) # use caer to unpack it, store it back to data, which gets timestamp and cooked data
        #     # print some info
        #     if data: # if could not unpack, is False
        #         print('packet #'+str(i)
        #           +' timestamp: '+str(data['timestamp'])
        #           +' etype: '+str(data['etype'])
        #           +' esize: '+str(data['esize'])
        #           +' enumber: '+str(data['enumber'])
        #           )
        #     i+=1
        #     if i>50: break


    def readPacket(self, number):
        """
        Reads packet k in the dataset
        Parameters
        ----------
        number: number of packet, in range(0,numPackets)

        Returns
        -------
        packet of data, or False if packet is outside of range or cannot be extracted
        """
        if number >= self.numPackets or number<0: return False
        dat = self.davisData[number]
        headerDat = dat[1]  # caer header
        if headerDat.shape[0]==0: return False  # empty packet, can happen at end of recording
        packet = {'dvs_header': dat[1]}  # put it to the dict as header
        packet.update(caer.unpack_header(packet['dvs_header']))  # update the dict?
        # dat0 = dat[0]  # timestamp of the packet?
        packet['dvs_data'] = dat[2]  # put the data payload, dvs_data refers to DAVIS camera data, can be frames or IMU data too
        packet = caer.unpack_data(packet, self.rotate180)  # use caer to unpack it, store it back to data, which gets timestamp and cooked data

        # # print some info
        # if data:  # if could not unpack, is False
        #     print('packet #' + str(k)
        #           + ' timestamp: ' + str(data['timestamp'])
        #           + ' etype: ' + str(data['etype'])
        #           + ' esize: ' + str(data['esize'])
        #           + ' enumber: ' + str(data['enumber'])
        #           )
        return packet

    def search(self,timeS):
        """
        Search for a starting time
        Parameters
        ----------
        timeS relative time in s from start of recording (self.startTimeS)

        Returns
        -------
        packet number

        """
        logger.info('searching for time {}'.format(timeS))
        start=self.firstPacketNumber
        if self.lastSearchTime is not None and self.lastSearchPacketNumber is not None and self.lastSearchTime<timeS:
            start=self.lastSearchPacketNumber
        for k in tqdm(range(self.firstPacketNumber,self.numPackets),unit='packet',desc='ddd-h5-search'):
            data=self.readPacket(k)
            if not data: # maybe cannot parse this particular type of packet (e.g. imu6)
                continue
            t=data['timestamp']
            if t>=self.firstTimeS+timeS:
                logger.info('\nfound start time '+str(timeS)+' at packet '+str(k))
                self.lastSearchTime=timeS
                self.lastSearchPacketNumber=k
                return k
        logger.warning('\ncould not find start time '+str(timeS)+' before end of file')
        return False

    def readEntire(self,startTimeS=None, stopTimeS=None):
        sys_ts, t_offset, current = 0, 0, 0
        timestamp = 0
        frames, events = [], []


        start=self.search(startTimeS)
        stop=self.search(stopTimeS)
        for k in tqdm(range(start,stop),desc='read',unit='packet'):
            d = self.readPacket(k)
            if not d:
                continue # some packet type we can't parse
            if d['etype'] == 'special_event':
                unpack_data(d,self.rotate180)
                # this is a timestamp reset
                if any(d['data'] == 0):
                    print('ts reset detected, setting offset', timestamp)
                    t_offset += current
                    # NOTE the timestamp of this special event is not meaningful
                continue
            if d['etype'] == 'frame_event':
                ts = d['timestamp'] + t_offset
                frame = filter_frame(unpack_data(d,self.rotate180))
                data = np.array(
                    [(ts, frame)],
                    dtype=np.dtype(
                        [('ts', np.float64),
                         ('frame', np.uint8, frame.shape)]
                    )
                )
                frames.append(data)
                self.shape = frame.shape

                current = ts
                continue
            if d['etype'] == 'polarity_event':
                unpack_data(d,self.rotate180)
                data = d["data"]
                data = np.hstack(
                    (data[:, 0][:, None] * 1e-6 + t_offset,
                     data[:, 1][:, None],
                     data[:, 2][:, None],
                     data[:, 3].astype(np.int)[:, None] * 2 - 1)
                )
                events.append(data)
                continue

        if frames:
            frames = np.hstack(frames)
            frames["ts"] -= frames["ts"][0]
        if events:
            events = np.vstack(events)
            events[:, 0] -= events[0][0]
        return frames, events



class DDD20ReaderMultiProcessing(object):
    """
    Read aps frames and events from hdf5 files in DDD
    @author: Zhe He
    @contact: hezhehz@live.cn
    @latest update: 2019-May-31
    """

    def __init__(self, fname, startTimeS=None, stopTimeS=None): # todo add rotate180 to mp reader
        """Init

        Parameters
        ----------
        fname: str
            path of input hdf5 file.
        startTimeS: float
            start time of the stream in seconds.
        stopTimeS: float
            stop time of the stream in seconds.
        """
        self.f_in = HDF5Stream(fname, {'dvs'})
        self.m = MergedStream(self.f_in)
        self.start = int(self.m.tmin + 1e6 * startTimeS) if startTimeS else 0
        self.stop = (self.m.tmin + 1e6 * stopTimeS) if stopTimeS else self.m.tmax
        self.m.search(self.start)

    def readEntire(self):
        """
        Read entire file to memory.

        Returns
        frames, events
        -------
        aps_frame: np.ndarray, [n, width, height]
            aps frames
        events: numpy record array.
            events, col names: ["ts", "y", "x", "polarity"], \
                data types: ["<f8", "<i8", "<i8", "<i8"]
        """
        sys_ts, t_offset, current = 0, 0, 0
        timestamp = 0
        frames, events = [], []
        while self.m.has_data and sys_ts <= self.stop * 1e-6:
            try:
                sys_ts, d = self.m.get()
            except queue.Empty:
                # wait for queue to fill up
                time.sleep(0.01)
                continue
            if not d or sys_ts < self.start * 1e-6:
                # skip unused data
                continue
            if d['etype'] == 'special_event':
                unpack_data(d)
                # this is a timestamp reset
                if any(d['data'] == 0):
                    print('ts reset detected, setting offset', timestamp)
                    t_offset += current
                    # NOTE the timestamp of this special event is not meaningful
                continue
            if d['etype'] == 'frame_event':
                ts = d['timestamp'] + t_offset
                frame = filter_frame(unpack_data(d))
                data = np.array(
                    [(ts, frame)],
                    dtype=np.dtype(
                        [('ts', np.float64),
                         ('frame', np.uint8, frame.shape)]
                    )
                )
                frames.append(data)
                self.shape=frame.shape
                current = ts
                continue
            if d['etype'] == 'polarity_event':
                unpack_data(d)
                data = d["data"]
                data = np.hstack(
                    (data[:, 0][:, None] * 1e-6 + t_offset,
                     data[:, 1][:, None],
                     data[:, 2][:, None],
                     data[:, 3].astype(np.int)[:, None] * 2 - 1)
                )
                events.append(data)
                continue
        frames = np.hstack(frames)
        events = np.vstack(events)
        frames["ts"] -= frames["ts"][0]
        events[:, 0] -= events[0][0]
        self.f_in.exit.set()
        self.m.exit.set()
        self.f_in.join()
        self.m.join()

        return frames, events

def filter_frame(d):
    '''
    receives 16 bit frame,
    needs to return unsigned 8 bit img
    '''
    # add custom filters here...
    # d['data'] = my_filter(d['data'])
    frame8 = (d['data'] / 256).astype(np.uint8)
    return frame8

class HDF5Stream(mp.Process):
    def __init__(self, filename, tables, bufsize=64):
        super(HDF5Stream, self).__init__()
        self.f = h5py.File(filename, 'r')
        self.tables = tables
        self.q = {k: mp.Queue(bufsize) for k in self.tables}
        self.run_search = mp.Event()
        self.exit = mp.Event()
        self.done = mp.Event()
        self.skip_to = mp.Value('L', 0)
        self._init_count()
        self._init_time()
        self.daemon = True
        self.start()

    def run(self):
        while self.blocks_rem and not self.exit.is_set():
            blocks_read = 0
            for k in self.blocks_rem.keys():
                if self.q[k].full():
                    time.sleep(1e-6)
                    continue
                i = self.block_offset[k]
                self.q[k].put(self.f[k]['data'][i*CHUNK_SIZE:(i+1)*CHUNK_SIZE])
                self.block_offset[k] += 1
                if self.blocks_rem[k].value:
                    self.blocks_rem[k].value -= 1
                else:
                    self.blocks_rem.pop(k)
                blocks_read += 1
            if not blocks_read:
                time.sleep(1e-6)
            if self.run_search.is_set():
                self._search()
        self.f.close()
        print('closed input file')
        while not self.exit.is_set():
            time.sleep(1e-3)
        # print('[DEBUG] flushing stream queues')
        for k in self.q:
            # print('[DEBUG] flushing', k)
            _flush_q(self.q[k])
            self.q[k].cleanup()
            self.q[k].join_thread()
        # print('[DEBUG] flushed all stream queues')
        self.done.set()
        print('stream done')

    def get(self, k, block=True, timeout=None):
        return self.q[k].get(block, timeout)

    def _init_count(self, offset={}):
        self.block_offset = {k: offset.get(k, 0) / CHUNK_SIZE
                             for k in self.tables}
        self.size = {k: len(self.f[k]['data']) - v * CHUNK_SIZE
                     for k, v in self.block_offset.items()}
        self.blocks = {k: v / CHUNK_SIZE for k, v in self.size.items()}
        self.blocks_rem = {
            k: mp.Value(ctypes.c_double, v) for k, v in self.blocks.items() if v}

    def _init_time(self):
        self.ts_start = {}
        self.ts_stop = {}
        self.ind_stop = {}
        for k in self.tables:
            ts_start = self.f[k]['timestamp'][self.block_offset[k]*CHUNK_SIZE]
            self.ts_start[k] = mp.Value('L', ts_start)
            b = self.block_offset[k] + self.blocks_rem[k].value - 1
            while b > self.block_offset[k] and \
                    self.f[k]['timestamp'][b*CHUNK_SIZE] == 0:
                b -= 1
            print(k, 'final block:', b)
            self.ts_stop[k] = mp.Value(
                'L', self.f[k]['timestamp'][(b + 1) * CHUNK_SIZE - 1])
            self.ind_stop[k] = b

    def init_search(self, t):
        ''' start streaming from given time point '''
        if self.run_search.is_set():
            return
        self.skip_to.value = np.uint64(t)
        self.run_search.set()

    def _search(self):
        t = self.skip_to.value
        offset = {k: self._bsearch_by_timestamp(k, t) for k in self.tables}
        for k in self.tables:
            _flush_q(self.q[k])
        self._init_count(offset)
        # self._init_time()
        self.run_search.clear()

    def _bsearch_by_timestamp(self, k, t):
        '''performs binary search on timestamp, returns closest block index'''
        l, r = 0, self.ind_stop[k]
        print('searching', k, t)
        while True:
            if r - l < 2:
                print('selecting block', l)
                return l * CHUNK_SIZE
            if self.f[k]['timestamp'][(l + (r - l) / 2) * CHUNK_SIZE] > t:
                r = l + (r - l) / 2
            else:
                l += (r - l) / 2


class MergedStream(mp.Process):
    ''' Unpacks and merges data from HDF5 stream '''
    def __init__(self, fbuf, bufsize=256):
        super(MergedStream, self).__init__()
        self.fbuf = fbuf
        self.ts_start = self.fbuf.ts_start
        self.ts_stop = self.fbuf.ts_stop
        self.q = mp.Queue(bufsize)
        self.run_search = mp.Event()
        self.skip_to = mp.Value('L', 0)
        self._init_state()
        self.done = mp.Event()
        self.fetched_all = mp.Event()
        self.exit = mp.Event()
        self.daemon = True
        self.start()

    def run(self):
        while self.blocks_rem and not self.exit.is_set():
            # find next event
            if self.q.full():
                time.sleep(1e-4)
                continue
            next_k = min(self.current_ts, key=self.current_ts.get)
            self.q.put((self.current_ts[next_k], self.current_dat[next_k]))
            self._inc_current(next_k)
            # get new blocks if necessary
            for k in {k for k in self.blocks_rem if self.i[k] == CHUNK_SIZE}:
                self.current_blk[k] = self.fbuf.get(k)
                self.i[k] = 0
                if self.blocks_rem[k]:
                    self.blocks_rem[k] -= 1
                else:
                    self.blocks_rem.pop(k)
                    self.current_ts.pop(k)
            if self.run_search.is_set():
                self._search()
        self.fetched_all.set()
        self.fbuf.exit.set()
        while not self.fbuf.done.is_set():
            time.sleep(1)
            # print('[DEBUG] waiting for stream process')
        while not self.exit.is_set():
            time.sleep(1)
            # print('[DEBUG] waiting for merger process')
        _flush_q(self.q)
        # print('[DEBUG] flushed merger q ->', self.q.qsize())
        self.q.close()
        self.q.join_thread()
        # print('[DEBUG] joined merger q')
        self.done.set()

    def close(self):
        self.exit.set()

    def _init_state(self):
        keys = self.fbuf.blocks_rem.keys()
        self.blocks_rem = {k: self.fbuf.blocks_rem[k].value for k in keys}
        self.current_blk = {k: self.fbuf.get(k) for k in keys}
        self.i = {k: 0 for k in keys}
        self.current_dat = {}
        self.current_ts = {}
        for k in keys:
            self._inc_current(k)

    def _inc_current(self, k):
        ''' get next event of given type and increment row pointer '''
        row = self.current_blk[k][self.i[k]]
        if k == 'dvs':
            ts, d = caer_event_from_row(row)
        else:  # vi event
            ts = row[0] * 1e-6
            d = {'etype': k, 'timestamp': row[0], 'data': row[1]}
        if not ts and k in self.current_ts:
            self.current_ts.pop(k)
            self.blocks_rem.pop(k)
            return False
        self.current_ts[k], self.current_dat[k] = ts, d
        self.i[k] += 1

    def get(self, block=False):
        return self.q.get(block)

    @property
    def has_data(self):
        return not (self.fetched_all.is_set() and self.q.empty())

    @property
    def tmin(self):
        return self.ts_start['dvs'].value

    @property
    def tmax(self):
        return self.ts_stop['dvs'].value

    def search(self, t, block=True):
        if self.run_search.is_set():
            return
        self.skip_to.value = np.uint64(t)
        self.run_search.set()

    def _search(self):
        self.fbuf.init_search(self.skip_to.value)
        while self.fbuf.run_search.is_set():
            time.sleep(1e-6)
        _flush_q(self.q)
        self._init_state()
        self.q.put((0, {'etype': 'timestamp_reset'}))
        self.run_search.clear()


def caer_event_from_row(row):
    '''
    Takes binary dvs data as input,
    returns unpacked event data or False if event type does not exist.
    '''
    sys_ts, head, body = (v.tobytes() for v in row)
    if not sys_ts:
        # rows with 0 timestamp do not contain any data
        return 0, False
    d = caer.unpack_header(head)
    d['dvs_data'] = body
    return int(sys_ts) * 1e-6, unpack_data(d)


def _flush_q(q):
    ''' flush queue '''
    while True:
        try:
            q.get(timeout=1e-3)
        except queue.Empty:
            if q.empty():
                break
