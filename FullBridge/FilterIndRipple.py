__author__ = 'PatDen00'

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

from functions import *

# general
t = np.arange(0, 4, 1e-4)

# system parameters
Vdc = 450
fs = 200e3
L = 2*200e-6
C = (680e-9/2)*3

Ir0 = Vdc/fs/L
Vr0 = Ir0/fs/C

# voltage mode
# create pwm signals
# inductor current ripple is max. at Da = Db = 0.5 due to the common mode currents
Da = 0.5
Db = 0.5
Va = pwm(t, Da, centeralign = True)
Vb = pwm(t, Db, centeralign = True)

# plot common mode inductor current ripple
fig = plt.figure(figsize=(8, 6), dpi=80)
showCMripple(fig,t,Va,Vb, 'Max. inductor current ripple in voltage mode')
plt.show()
fig.savefig('VM_L_ripple.pdf')

# create pwm signals
# DM inductor current ripple is max. at Da = 75, Db = 0.25 -> Vout = +-Vdc/2
Da = 0.75
Db = 0.25
Va = pwm(t, Da, centeralign = True)
Vb = pwm(t, Db, centeralign = True)

# plot differential mode inductor current ripple and capacitor voltage ripple
fig = plt.figure(figsize=(8, 8), dpi=80)
showDMripple(fig,t,Va,Vb, 'Max. capacitor voltage ripple in voltage mode')
plt.show()
fig.savefig('VM_C_ripple.pdf')

# system parameters
Vdc = 38
fs = 200e3
L = 10e-6
C = 680e-9*2

Ir0 = Vdc/fs/L
Vr0 = Ir0/fs/C

# current mode
# create pwm signals
# inductor current ripple is max. at Da = Db = 0.5
Da = 0.5
Db = 0.5
Va = pwm(t, Da, centeralign = True, shift=True)
Vb = pwm(t, Db, centeralign = True, shift=False)

# plot inductor current ripple
fig = plt.figure(figsize=(8, 6), dpi=80)
showripple_HB1(fig,t,Va,Vb, 'Max. inductor current ripple in current mode')
plt.show()
fig.savefig('CM_L_ripple.pdf')

# create pwm signals
# capacitor voltage ripple is max. at Da = Db = 0.75
Da = 0.75
Db = 0.75
Va = pwm(t, Da, centeralign = True, shift=True)
Vb = pwm(t, Db, centeralign = True, shift=False)

# plot capacitor voltage ripple
fig = plt.figure(figsize=(8, 8), dpi=80)
showripple_HB2(fig,t,Va,Vb, 'Max. capacitor voltage ripple in current mode')
plt.show()
fig.savefig('CM_C_ripple.pdf')


# # plot ripple over duty cycle
# D = np.arange(-1,1,0.005)
# plt.plot(D,abs(D)*(1-abs(D)))
# plt.xlabel('net duty cycle $D = D_a - D_b$',fontsize=16)
# plt.ylabel('ripple current / $I_{R0}$',fontsize=16)
# plt.grid()
# plt.show()