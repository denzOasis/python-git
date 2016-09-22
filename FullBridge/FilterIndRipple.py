__author__ = 'PatDen00'

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

from functions import *

# init
Da, Db = 0.75, 0.25
D0 = (Da + Db)/2
iL = 0

# create a time vector
t = np.arange(0,4,0.0005)

T = t[-1] - t[0]
print(T)
yint0 = np.append([0], scipy.integrate.cumtrapz(Da - Db, t))
meanval0 = yint0[-1] / T
yint1 = yint0 - (t - t[0]) * meanval0
meanval1 = scipy.integrate.trapz(yint1, t) / T

# plot inductor current ripple
fig = plt.figure(figsize=(8, 6), dpi=80)
showpwmripple(fig,t,Da,Db,iL,centeralign=True);
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