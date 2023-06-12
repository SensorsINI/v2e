# plots the recorded single pixel states
import pickle

#%% load data
from easygui import fileopenbox
import pickle
import numpy as np
f=fileopenbox('choose recorded .npz file',default='*.dat')
states=None
with open(f,'rb') as infile:
    states=pickle.load(infile)
states

#%% test cell
t=states['time']
base_log_frame=states['base_log_frame']
new_frame=states['new_frame']
lp_log_frame=states['lp_log_frame']
log_new_frame=states['log_new_frame']
pos_thres=states['pos_thres']
neg_thres=states['neg_thres']
diff_frame=states['diff_frame']
final_neg_evts_frame=states['final_neg_evts_frame']
final_pos_evts_frame=states['final_pos_evts_frame']

#%% plot data
import matplotlib.pyplot as plt

plt.subplot(3,1,1)
plt.plot( t,new_frame)
plt.xlabel('time (s)')
plt.ylabel('new frame')
plt.plot( t,new_frame)

plt.subplot(3,1,2)
plt.plot( t, base_log_frame, t,log_new_frame, t,lp_log_frame)
plt.xlabel('time (s)')
plt.ylabel('log')
plt.legend([ 'base_log_frame', 'log_new_frame', 'lp_log_frame'])

plt.subplot(3,1,3)
plt.plot(t,pos_thres, t,-neg_thres, t,diff_frame,t,final_pos_evts_frame,t,-final_pos_evts_frame)
plt.xlabel('time (s)')
plt.legend(['pos_thres','-neg_thres','diff_frame','final_pos_evts_frame', '-final_neg_evts_frame'])
plt.show()