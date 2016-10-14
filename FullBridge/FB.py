__author__ = 'PatDen00'

import matplotlib.pyplot as plt
import numpy as np
from functions import *

# general
fs = 200e3
t = np.arange(0, 4/fs, 1/fs/10e3)

# CVA-VM
# system parameters
Vdc = 450
L = 200e-6*2
C = (680e-9/2)*3

# inductor current ripple is max. at Da = Db = 0.5 due to the common mode currents
Da = 0.5
Db = 0.5
Va = Vdc*pwm(t, Da, fs, centeralign = True, shift=0) - Vdc/2
Vb = Vdc*pwm(t, Db, fs, centeralign = True, shift=180) - Vdc/2
Ia = calcripple(t, Va)/(L/2)
Ib = calcripple(t, Vb)/(L/2)

# plot
axlist = []
fig = plt.figure(figsize=(8, 6), dpi=80)
fig.suptitle('CVA-VM max. inductor current ripple')
ax = fig.add_subplot(2, 1, 1)
ax.plot(t, Va)
ax.plot(t, Vb)
ax.set_ylim(createLimits(.1, Va))
ax.set_ylabel('voltage [V]')
ax.grid(True)
axlist.append(ax)
ax = fig.add_subplot(2, 1, 2)
ax.plot(t, Ia)
ax.plot(t, Ib)
ax.set_ylabel('current [A]')
annotate_ripple(ax, t, Ia, 0)
ax.set_xlabel('time [sec]')
ax.grid(True)
#plt.show()
fig.savefig('PDFs\CVA_VM_L_ripple.pdf')

# max. capacitor current ripple is determined by the differential mode inductor current ripple
# it is max. at Da = 75, Db = 0.25 -> Vout = +-Vdc/2
Da = 0.75
Db = 0.25
Va = Vdc*pwm(t, Da, fs, centeralign = True)
Vb = Vdc*pwm(t, Db, fs, centeralign = True)
Vdm = Va - Vb
Il = calcripple(t, Vdm)/L
Vc = calcripple(t, Il)/C

# plot
fig = plt.figure(figsize=(8, 6), dpi=80)
fig.suptitle('CVA-VM max. capacitor voltage ripple')
ax = fig.add_subplot(3, 1, 1)
ax.plot(t, Vdm)
ax.set_ylim(createLimits(.1, Va))
ax.set_ylabel('voltage [V]')
ax.grid(True)
ax = fig.add_subplot(3, 1, 2)
ax.plot(t, Il)
ax.set_ylabel('current [A]')
ax.grid(True)
annotate_ripple(ax, t, Il, 0)
ax = fig.add_subplot(3, 1, 3)
ax.plot(t, Vc)
ax.set_ylabel('voltage [V]')
ax.grid(True)
annotate_ripple(ax, t, Vc, 0)
ax.set_xlabel('time [sec]')
#plt.show()
fig.savefig('PDFs\CVA_VM_C_ripple.pdf')

# input capacitor
# init
Da, Db = 0.75, 0.25
D0 = (Da + Db)/2
iL = 0

# create time vector
t = np.arange(0,4,0.0005)

# show ripple in inductor and current drawn from dc link capacitor
fig = plt.figure(figsize=(8, 6), dpi=80)
showpwmripple2(fig, t, Da, Db, iL, centeralign=True)
plt.grid()
#plt.show()

#
fig = plt.figure(figsize=(8, 6), dpi=80)
split_Icdc_1(fig, t, Da, Db, iL, centeralign=True)
plt.grid()
#plt.show()