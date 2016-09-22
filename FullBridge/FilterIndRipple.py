__author__ = 'PatDen00'

from functions import *

# init
Da, Db = 0.75, 0.25
D0 = (Da + Db)/2
iL = 0

# create a time vector
t = np.arange(0,4,0.0005)

# create PWM signal and plot it
sigA = ('A',pwm(t,D0,centeralign=True))
fig1 = plt.figure(1)
ax = fig1.add_subplot(1,1,1)
digitalplotter(t, sigA+(sawtooth(t),'k:'))(ax)
plt.grid()
plt.show()

# create PWM signal for both branches
pwmA = pwm(t,Da,centeralign=True)
pwmB = pwm(t,Db,centeralign=True)

# plot resulting signal at switching node which has double the frequency
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
digitalplotter(t,
  ('pwmA',pwmA),
  ('pwmB',pwmB),
  ('pwmA-pwmB',pwmA-pwmB))(ax)
plt.grid()
plt.show()

# plot inductor current ripple
fig = plt.figure(figsize=(8, 6), dpi=80)
showpwmripple(fig,t,Da,Db,iL,centeralign=True);
plt.grid()
plt.show()

# plot ripple over duty cycle
D = np.arange(-1,1,0.005)
plt.plot(D,abs(D)*(1-abs(D)))
plt.xlabel('net duty cycle $D = D_a - D_b$',fontsize=16)
plt.ylabel('ripple current / $I_{R0}$',fontsize=16)
plt.grid()
plt.show()

#
fig = plt.figure(figsize=(8, 6), dpi=80)
intcodes = 'ABCD'
axlist = showpwmripple(fig,t,Da,Db,iL,centeralign=True)
for (i,tcenter) in enumerate([0.5, 1-(Da+Db)/4, 1, 1+(Da+Db)/4]):
    y = (i/2) * 1.1 + 0.55
    intcode = '[%s]' % intcodes[i]
    axlist[0].annotate(intcode,xy=(tcenter,y),
        horizontalalignment='center',
        verticalalignment='center')
    axlist[2].annotate(intcode,xy=(tcenter,0.05),
        horizontalalignment='center',
        verticalalignment='center')
plt.grid()
plt.show()

# show one period
show1period(Da,Db)
plt.grid()
plt.show()

# SYMPY stuff
from IPython.display import display
from sympy import init_printing
init_printing('mathjax')

import sympy
#%load_ext sympy.interactive.ipythonprinting
D, D0 = sympy.symbols("D D_0")
Da =  D/2 + D0
Db = -D/2 + D0
# now let's verify that we've defined D and D0 properly:
display('Da-Db=',Da-Db)
display('(Da+Db)/2=',(Da+Db)/2)