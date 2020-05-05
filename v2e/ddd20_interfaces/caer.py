#!/usr/bin/python

'''
Recorder for DAVIS + OpenXC data
Author: J. Binas <jbinas@gmail.com>, 2017

This software is released under the
GNU LESSER GENERAL PUBLIC LICENSE Version 3.
'''

from __future__ import print_function
import time
import numpy as np
import socket, struct
import multiprocessing as mp
import queue

HOST = "127.0.0.1"
PORT = 7777
PORT_CTL = 4040

SENSOR = 'DAVIS346B'
#SENSOR = 'DAVIS640'

DVS_SHAPE = (260, 346)
#DVS_SHAPE = (640, 480)

HEADER_FIELDS = (
        'etype',
        'esource',
        'esize',
        'eoffset',
        'eoverflow',
        'ecapacity',
        'enumber',
        'evalid',
        )

EVENT_TYPES = {
        'special_event': 0,
        'polarity_event': 1,
        'frame_event': 2,
        'imu6_event': 3,
        'imu9_event': 4,
        }

etype_by_id = {v: k for k,v in EVENT_TYPES.items()}


def unpack_events(p, rotate180=True):
    '''
    Extract events from binary data,
    returns list of event tuples.
    '''
    if not p['etype'] == 'polarity_event':
        return False
    p_arr = np.fromstring(p['dvs_data'], dtype=np.uint32)
    p_arr = p_arr.reshape((p['ecapacity'], p['esize'] // 4))
    data, ts = p_arr[:,0], p_arr[:,1]
    pol = data >> 1 & 0b1
    y = data >> 2 & 0b111111111111111
    x = data >> 17
    if rotate180:
        x=DVS_SHAPE[1]-x-1
        y=DVS_SHAPE[0]-y-1
    return ts[0] * 1e-6, np.array([ts, x, y, pol]).T

def unpack_header(header_raw):
    '''
    Extract header info from binary data,
    returns dict object.
    '''
    vals = struct.unpack('hhiiiiii', header_raw)
    obj = dict(zip(HEADER_FIELDS, vals))
    obj['etype'] = etype_by_id.get(obj['etype'], obj['etype'])
    return obj

def unpack_frame(p, rotate180=True):
    '''
    Extract image from binary data, returns timestamp and 2d np.array.
    '''
    if not p['etype'] == 'frame_event':
        return False
    img_head = np.fromstring(p['dvs_data'][:36], dtype=np.uint32)
    img_data = np.fromstring(p['dvs_data'][36:], dtype=np.uint16)
    img_data=img_data.reshape(DVS_SHAPE)
    if rotate180:
        img_data=np.rot90(img_data,k=2)
    return img_head[2] * 1e-6, img_data

def unpack_special(p, rotate180=True):
    '''
    Extract special event data (only return type id).
    '''
    if not p['etype'] == 'special_event':
        return False
    p_arr = np.fromstring(p['dvs_data'], dtype=np.uint32)
    p_arr = p_arr.reshape((p['ecapacity'], p['esize'] // 4))
    data, ts = p_arr[:,0], p_arr[:,1]
    typeid = data & 254
    #valid = data & 1
    #opt = data >> 8
    return ts[0] * 1e-6, typeid

unpack_func = {
        'polarity_event': unpack_events,
        'frame_event': unpack_frame,
        'special_event': unpack_special,
        }


def unpack_data(d, rotate180=True):
    '''
    Unpack data for given caer packet,
    return False if event type does not exist.
    '''
    _get_data = unpack_func.get(d['etype'])
    if _get_data:
        d['timestamp'], d['data'] = _get_data(d,rotate180)
        return d
    return False




class Monitor(mp.Process):
    def __init__(self, bufsize=2048):
        super(Monitor, self).__init__()
        self.sock = socket.socket()
        self.sock.connect((HOST, PORT))

        # network header (contains: magic number, sequence number,
        # version number, format number, source ID)
        hdata = self.sock.recv(20, socket.MSG_WAITALL)  # header of aer stream
        self.hdata = struct.unpack('llbbh', hdata)
        print('opened connection:', self.hdata)
        self.q = mp.queue(bufsize)
        self.maxsize = self.q._maxsize
        self.qsize = 0
        self.exit = mp.Event()
        #self.daemon = True
        self.start()

    def run(self):
        while not self.exit.is_set():
            try:
                self.q.put_nowait(self._get())
                self.qsize = max(self.qsize, self.q.qsize())
            except Queue.Full:
                raise Full('caer buffer overflow')
            except KeyboardInterrupt:
                self.exit.set()

    def _get(self):
        # read packet header
        data = {'dvs_header': self.sock.recv(28, socket.MSG_WAITALL)}
        data['dvs_timestamp'] = int(time.time() * 1e6)
        data.update(unpack_header(data['dvs_header']))
        # read full packet
        psize = data['ecapacity'] * data['esize']
        data['dvs_data'] = self.sock.recv(psize, socket.MSG_WAITALL)
        return data

    def get(self):
        return self.q.get_nowait() if not self.q.empty() else False

    def get_events(self):
        return unpack_events(self.get())

    def shutdown(self):
        self.exit.set()


class Controller(object):
    def __init__(self):
        self.data_buffer_size = 4069 * 30
        self.max_cmd_parts = 5
        self.cmd_part_action = 0
        self.NODE_EXISTS = 0
        self.ATTR_EXISTS = 1
        self.GET = 2
        self.PUT = 3
        self.cmd_part_node = 1
        self.cmd_part_key = 2
        self.cmd_part_type = 3
        self.cmd_part_value = 4
        self.type_action = {
                'bool': 0,
                'byte': 1,
                'short': 2,
                'int': 3,
                'long': 4,
                'float': 5,
                'double': 6,
                'string': 7
                }
        self.actions = [
                ("node_exists", 11, self.NODE_EXISTS),
                ("attr_exists", 11, self.ATTR_EXISTS),
                ("get", 3, self.GET),
                ("put", 3, self.PUT)
                ]
        try:
            self.s_commands = socket.socket()
            self.s_commands.connect((HOST, PORT_CTL))
        except socket.error as msg:
            print('Failed to create socket ' + str(msg))
            quit()

    def parse_command(self, command):
        '''
        parse string command
        e.g. string: put /1/1-DAVISFX2/'+str(sensor)+'/aps/ Exposure int 10
        (copied from https://svn.code.sf.net/p/jaer/code/scripts/python/cAER_utils/imagers_characterization/caer_communication.py)
        '''
        databuffer = bytearray(b'\x00' * self.data_buffer_size)
        node_length = 0
        key_length = 0
        action_code = -1
        cmd_parts = command.split()
        if(len(cmd_parts) > self.max_cmd_parts):
            print('Error: command is made up of too many parts')
            return
        else:
            if(cmd_parts[self.cmd_part_action] != None):
                for i in range(len(self.actions)):
                    if(cmd_parts[self.cmd_part_action] == self.actions[i][0]):
                        action_code = self.actions[i][2]
                if(action_code == -1):
                    print("Please specify an action to perform as: get/put..")
                    return
                #do action based on action_code
                if(action_code == self.NODE_EXISTS):
                    node_length = len(cmd_parts[self.cmd_part_node]) + 1
                    databuffer[0] = action_code
                    databuffer[1] = 0 #unused
                    databuffer[10:10+node_length] = self.cmd_parts[self.cmd_part_node]
                    databuffer_lenght = 10 + node_length
                if(action_code == self.PUT):
                    node_length  = len(cmd_parts[self.cmd_part_node]) + 1
                    key_length = len(cmd_parts[self.cmd_part_key]) + 1
                    value_length = len(cmd_parts[self.cmd_part_value]) + 1
                    databuffer[0] = action_code
                    databuffer[1] = self.type_action[cmd_parts[self.cmd_part_type]]
                    databuffer[2:3] = struct.pack('H', 0)
                    databuffer[4:5] = struct.pack('H', node_length)
                    databuffer[6:7] = struct.pack('H', key_length)
                    databuffer[8:9] = struct.pack('H', value_length)
                    databuffer[10:10+node_length] = str(cmd_parts[self.cmd_part_node])
                    databuffer[10+node_length:10+node_length+key_length] = str(cmd_parts[self.cmd_part_key])
                    databuffer[10+node_length+key_length:10+node_length+key_length+value_length] = str(cmd_parts[self.cmd_part_value])
                    databuffer_length = 10 + node_length + key_length + value_length
                    #raise Exception
        return databuffer[0:databuffer_length]

    def send_command(self, string):
        '''
        parse input command and send it to the device
        print the answer
        input string - ie. 'put /1/1-DAVISFX2/'+str(sensor)+'/aps/ Exposure int 100'
        '''
        cmd = self.parse_command(string)
        self.s_commands.sendall(cmd)
        msg_header = self.s_commands.recv(4)
        msg_packet = self.s_commands.recv(struct.unpack('H', msg_header[2:4])[0])
        action = struct.unpack('B',msg_header[0])[0]
        second = struct.unpack('B',msg_header[1])[0]
        #print(string+' action='+str(action)+' type='+str(second)+' message='+msg_packet)

    def set_aps(self, name, dtype, value):
        self.send_command('put /1/1-DAVISFX3/%s/aps/ %s %s %s' % (SENSOR, name, dtype, value))


class ExposureCtl(Controller):
    '''
    Automatic exposure control
    * fps -- update frequency
    * target -- target average pixel value (between 0 and 255)
    * cutoff_top -- number of pixels at the top of image to be ignored
    * cutoff_bot -- number of pixels at the bottom of image to be ignored
    
    We generally want to expose for road which is bottom of image. Therefore the default
    is to ignore the top 100 pixels (cutoff_top) and to ignore the bottom 50 (cutoff_bot) 
    which might be the hood. 
    Howvever, if sensor is mounted upside down, then we should ignore a lot of the bottom
    (sky) and maybe a bit of the top (hood).
    '''
    def __init__(self, fps=5, target=100, cutoff_top=10, cutoff_bot=200):
        super(ExposureCtl, self).__init__()
        self.fps = fps
        self.dt = 1. / self.fps
        self.target = float(target * 255)
        self.t_pre = 0
        self.cutoff_top = cutoff_top
        self.cutoff_bot = cutoff_bot
        self.exp_now = 1000 # dummy

    def update(self, packet):
        if time.time() - self.t_pre < self.dt:
            return
        if packet['etype'] != 'frame_event':
            return
        ts, frame = unpack_frame(packet)
        m = frame[self.cutoff_top:-self.cutoff_bot].mean()
        upd = (self.target / m - m / self.target) / 2
        exp_new = self.exp_now * (1 + 0.5 * upd)
        exp_new = np.clip(exp_new, 100, 100000).astype(int)
        self.set_aps('Exposure', 'int', int(exp_new))
        self.exp_now = exp_new
        self.t_pre = time.time()



if __name__ == '__main__':

    aer = Monitor()
    exposure = ExposureCtl(cutoff_bot=80, cutoff_top=0) # set for upside down camera where sky will be at bottom
    while True:
        packet = aer.get()
        if not packet:
            continue
        exposure.update(packet)
        evts = unpack_events(packet)
        if evts is not False:
            print('\nreceived events:', evts)

