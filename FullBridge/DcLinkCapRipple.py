__author__ = 'PatDen00'

from functions import *

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
plt.show()

#
fig = plt.figure(figsize=(8, 6), dpi=80)
split_Icdc_1(fig, t, Da, Db, iL, centeralign=True)
plt.grid()
plt.show()