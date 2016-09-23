__author__ = 'PatDen00'

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

from functions import *

# init
Da, Db = 0.8, 0.2
D0 = (Da + Db)/2
iL = 0

# create a time vector
t = np.arange(0,4,0.0005)
pwmA = pwm(t,0.5,centeralign=False)
# plot inductor current ripple
fig = plt.figure(figsize=(8, 6), dpi=80)
showpwmripple(fig,t,Da,Db, iL, centeralign=True);
plt.show()

# # plot ripple over duty cycle
# D = np.arange(-1,1,0.005)
# plt.plot(D,abs(D)*(1-abs(D)))
# plt.xlabel('net duty cycle $D = D_a - D_b$',fontsize=16)
# plt.ylabel('ripple current / $I_{R0}$',fontsize=16)
# plt.grid()
# plt.show()
#
# #
# fig = plt.figure(figsize=(8, 6), dpi=80)
# intcodes = 'ABCD'
# axlist = showpwmripple(fig,t,Da,Db,iL,centeralign=True)
# for (i,tcenter) in enumerate([0.5, 1-(Da+Db)/4, 1, 1+(Da+Db)/4]):
#     y = (i/2) * 1.1 + 0.55
#     intcode = '[%s]' % intcodes[i]
#     axlist[0].annotate(intcode,xy=(tcenter,y),
#         horizontalalignment='center',
#         verticalalignment='center')
#     axlist[2].annotate(intcode,xy=(tcenter,0.05),
#         horizontalalignment='center',
#         verticalalignment='center')
# plt.grid()
# plt.show()