__author__ = 'PatDen00'

import matplotlib.pyplot as plt
import numpy as np
from functions import *

# general
fs = 200e3
t = np.arange(0, 4/fs, 1/fs/40e3)

# HCA-CM
# system parameters
Vdc = 2*38
L = 3e-6
C = 3.3e-6*6

# inductor current ripple is max. at Da = Db = 0.5
D = 0.5
Va = Vdc*pwm(t, D, fs, centeralign = True, shift=0) - Vdc/2
Vb = Vdc*pwm(t, D, fs, centeralign = True, shift=120) - Vdc/2
Vc = Vdc*pwm(t, D, fs, centeralign = True, shift=240) - Vdc/2

Ia = calcripple(t, Va)/L
Ib = calcripple(t, Vb)/L
Ic = calcripple(t, Vc)/L

# plot
fig = plt.figure(figsize=(8, 6), dpi=80)
fig.suptitle('HCA-CM max. inductor current ripple')
ax = fig.add_subplot(2, 1, 1)
ax.plot(t, Va)
ax.plot(t, Vb)
ax.plot(t, Vc)
ax.set_ylim(createLimits(.1, Va))
ax.set_ylabel('voltage [V]')
ax.grid(True)
ax = fig.add_subplot(2, 1, 2)
ax.plot(t, Ia)
ax.plot(t, Ib)
ax.plot(t, Ic)
ax.set_ylabel('current [A]')
annotate_ripple(ax, t, Ia, 0)
ax.set_xlabel('time [sec]')
ax.grid(True)
#plt.show()
fig.savefig('PDFs\HCA_CM_L_ripple.pdf')

# capacitor voltage ripple is max. at D = 1/2 or 1/6 or 5/6
# assuming that the total ripple flows over the capacitor
D = 1/2
Va = Vdc*pwm(t, D, fs, centeralign = True, shift=0) - Vdc/2
Vb = Vdc*pwm(t, D, fs, centeralign = True, shift=120) - Vdc/2
Vc = Vdc*pwm(t, D, fs, centeralign = True, shift=240) - Vdc/2

Ia = calcripple(t, Va)/L
Ib = calcripple(t, Vb)/L
Ic = calcripple(t, Vc)/L
iC = Ia + Ib + Ic
vC = calcripple(t, iC)/C

# plot
fig = plt.figure(figsize=(8, 6), dpi=80)
fig.suptitle('HCA-CM max. capacitor voltage ripple')
ax = fig.add_subplot(3, 1, 1)
ax.plot(t, Va)
ax.plot(t, Vb)
ax.plot(t, Vc)
ax.set_ylim(createLimits(.1, Va))
ax.set_ylabel('voltage [V]')
ax.grid(True)
ax = fig.add_subplot(3, 1, 2)
ax.plot(t, Ia)
ax.plot(t, Ib)
ax.plot(t, Ic)
ax.plot(t, iC)
ax.set_ylabel('current [A]')
ax.grid(True)
annotate_ripple(ax, t, iC, 0)
ax = fig.add_subplot(3, 1, 3)
ax.plot(t, vC)
ax.set_ylabel('voltage [V]')
ax.grid(True)
annotate_ripple(ax, t, vC, 0)
ax.set_xlabel('time [sec]')
#plt.show()
fig.savefig('PDFs\HCA_CM_C_ripple.pdf')