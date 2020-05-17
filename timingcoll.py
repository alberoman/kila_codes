#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:23:31 2020

@author: aroman
"""

def read_csv(filename):
  
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        A = []
        for row in readCSV:
            A.append(row)
    return A
def format_tilt(A):
    A = A[4:]
    t = []
    d = []
    for i in A:
        t.append(float(i[0]))
        d.append(float(i[1]))
    t = np.array(t)
    d = np.array(d)
    return t,d
def get_data_segment(date_start,date_end,dates):
    #extract data between to dates expresses in human format
    number_start = mdates.date2num(datetime.strptime(date_start,'%Y-%m-%d'))
    number_end = mdates.date2num(datetime.strptime(date_end,'%Y-%m-%d'))
    ind = (dates>number_start) & (dates<number_end)
    return ind,number_start,number_end

stations = pickle.load(open('../data/tilt/tilt_dictionary_01may.pickle','rb'))
b, a = signal.butter(2, 0.03)
name = 'UWD'
east = stations[name]['east']
north = stations[name]['north']
time = stations[name]['time']
indices = (np.isnan(north) == False) & (np.isnan(east) == False)
time = time[indices] 
east = east[indices] 
north = north[indices]
index,start,end = get_data_segment(date_initial_str,date_final_str,time)
east = east[index]
north = north[index]
time = time[index]
east = (east - east[0]) 
north = (north - north[0])
time = time[::1]
north = north[::1]
east = east[::1]
time = time - time[0]
u1,w1,u2,w2,PCAmean = PCA_vectors(east,north) #Extract PCA vector and PCA mean
proj_max = east * u1 + north * w1
proj_max = signal.filtfilt(b, a, proj_max)
tilt = proj_max[:]
fig = plt.figure(constrained_layout=True, figsize = (7,4))
gs = fig.add_gridspec(3, 3)
ax0 = fig.add_subplot(gs[0, :2])
ax0.plot(time,tilt)
ax0.set_xlim([0,96])
ax0.set_ylabel('UWD [$\mu$rad]',fontsize= 12)
data = read_csv(path_tilt_paper + 'reunion_seis.csv')
days,tilt = format_tilt(data)
tilt = tilt[np.argsort(days)]
days = days[np.argsort(days)]

#days = days - days[0]
tilt = tilt - tilt[0]
ax1 = fig.add_subplot(gs[1, :2])
ax1.plot(days,tilt)
ax1.set_xlim([4,14.5])
ax1.set_ylim([-0.87,-0.10])
ax1.tick_params(labelsize = 10)
ax1.set_ylabel('RER [$\mu$rad]',fontsize= 12)
data = read_csv(path_tilt_paper + 'myake_data.csv')
days,tilt = format_tilt(data)

#days = days - days[0]
tilt = tilt - tilt[0]
tilt = tilt[np.argsort(days)]
days = days[np.argsort(days)]
tilt = - tilt
ax2 = fig.add_subplot(gs[2, :2])
ax2.plot(days,tilt)
ax2.set_xlabel('Days',fontsize= 12)
ax2.set_ylabel('MKS [$\mu$rad]',fontsize= 12)
ax2.set_xlim([0,33])
ax2.set_ylim([-55,-10])
fig.align_ylabels([ax0,ax1,ax2])
plt.savefig(path_figs + 'Tilt_Examples.pdf')

plt.show()