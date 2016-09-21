__author__ = 'PatDen00'

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from pwm import ramp, sawtooth, pwm

def digitalplotter(t,*signals):
  '''return a plotting function that takes an axis and plots
digital signals (or other signals in the 0-1 range)'''
  def f(ax):
    n = len(signals)
    for (i,sig) in enumerate(signals):
      ofs = (n-1-i)*1.1
      plotargs = []
      for y in sig[1:]:
        if isinstance(y,str):
          plotargs += [y]
        else:
          plotargs += [t,y+ofs]
      ax.plot(*plotargs)
    ax.set_yticks((n-1-np.arange(n))*1.1+0.55)
    ax.set_yticklabels([sig[0] for sig in signals])
    ax.set_ylim(-0.1,n*1.1)
  return f

def extendrange(ra, rb):
    '''return a tuple (x1,x2) representing the interval from x1 to x2,
  given two input ranges of the same form, or None (representing no input).'''
    if ra is None:
        return rb
    elif rb is None:
        return ra
    else:
        return (min(ra[0], rb[0]), max(ra[1], rb[1]))


def createLimits(margin, *args):
    '''add proportional margin to an interval:
createLimits(0.1, (1,3),(2,4),(0,2)) calculates the maximum extent
of the ranges provided, in this case (0,4), and adds another 0.1 (10%)
to the extent symmetrically, thus returning (-0.2, 4.2).'''
    r = None
    for x in args:
        r = extendrange(r, (np.min(x), np.max(x)))
    rmargin = (r[1] - r[0]) * margin / 2.0
    return (r[0] - rmargin, r[1] + rmargin)


def calcripple(t, y):
    ''' calculate ripple current by integrating the input,
then adjusting it by a linear function to put both endpoints
at the same value. The slope of the linear function is the
mean value of the input; the offset is chosen to make the mean value
of the output ripple current = 0.'''
    T = t[-1] - t[0]
    yint0 = np.append([0], scipy.integrate.cumtrapz(y, t))
    # cumtrapz produces a vector of length N-1
    # so we need to add one element back in at the beginning
    meanval0 = yint0[-1] / T
    yint1 = yint0 - (t - t[0]) * meanval0
    meanval1 = scipy.integrate.trapz(yint1, t) / T
    return yint1 - meanval1


def showripple(fig, t, Va, Vb, titlestring):
    '''plot ripple current as well as phase duty cycles and load voltage'''
    axlist = []
    Iab = calcripple(t, Va - Vb)
    margin = 0.1
    ax = fig.add_subplot(3, 1, 1)
    digitalplotter(t, ('Va', Va), ('Vb', Vb))(ax)
    ax.set_ylabel('Phase duty cycles')
    axlist.append(ax)

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(t, Va - Vb)
    ax.set_ylim(createLimits(margin, Va - Vb))
    ax.set_ylabel('Load voltage')
    axlist.append(ax)

    ax = fig.add_subplot(3, 1, 3)
    ax.plot(t, Iab)
    ax.set_ylim(createLimits(margin, Iab))
    ax.set_ylabel('Ripple current')
    axlist.append(ax)

    fig.suptitle(titlestring, fontsize=16)

    # annotate with peak values
    tlim = [min(t), max(t)]
    tannot0 = tlim[0] + (tlim[1] - tlim[0]) * 0.5
    tannot1 = tlim[0] + (tlim[1] - tlim[0]) * 0.6
    for y in [min(Iab), max(Iab)]:
        ax.plot(tlim, [y] * 2, 'k:')
        # see:
        # http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.annotate
        ax.annotate('%.5f' % y, xy=(tannot0, y), xytext=(tannot1, y * 0.3),
                    bbox=dict(boxstyle="round", fc="0.9"),
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc,angleA=0,armA=20,angleB=%d,armB=15,rad=7"
                                                    % (-90 if y > 0 else 90))
                    )
    return axlist


def showpwmripple(fig, t, Da, Db, centeralign=False, titlestring=''):
    return showripple(fig, t,
                      pwm(t, Da, centeralign),
                      pwm(t, Db, centeralign),
                      titlestring='%s-aligned pwm, $D_a$=%.3f, $D_b$=%.3f' %
                                  ('Center' if centeralign else 'Edge', Da, Db))

def show1period(Da,Db):
  t1period = np.arange(-0.5,1.5,0.001)
  pwmA = pwm(t1period,Da,centeralign=True)
  pwmB = pwm(t1period,Db,centeralign=True)
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  Ir = calcripple(t1period,pwmA-pwmB)
  ax.plot(t1period,Ir)
  ax.set_xlim(-0.1,1.1)
  ax.set_ylim(min(Ir)*1.2,max(Ir)*1.2)
  # now annotate
  (D1,D2,V) = (Db,Da,1) if Da > Db else (Da,Db,-1)
  tpts = np.array([0, D1/2, D2/2, 0.5, 1-(D2/2), 1-(D1/2), 1])
  dtpts = np.append([0],np.diff(tpts))
  vpts = np.array([0, 0, V, 0, 0, V, 0]) - (Da-Db)
  ypts = np.cumsum(vpts*dtpts)
  ax.plot(tpts,ypts,'.',markersize=8)
  for i in range(7):
    ax.annotate('$t_%d$' % i,xy=(tpts[i]+0.01,ypts[i]), fontsize=16)

t = np.arange(0,4,0.0005)
sigA = ('A',pwm(t,0.5,centeralign=True))
sigB = ('B',pwm(t,0.5,centeralign=False))
sigC = ('C',pwm(t,0.9,centeralign=False))

fig1 = plt.figure(1)
ax = fig1.add_subplot(1,1,1)
digitalplotter(t,
  sigA+(sawtooth(t),'k:'),
  sigB+(ramp(t),'k:'),
  sigC+(ramp(t),'k:'))(ax)
plt.grid()

pwmA = pwm(t,0.75,centeralign=True)
pwmB = pwm(t,0.25,centeralign=True)

fig2 = plt.figure(2)
ax = fig2.add_subplot(1,1,1)
digitalplotter(t,
  ('pwmA',pwmA),
  ('pwmB',pwmB),
  ('pwmA-pwmB',pwmA-pwmB))(ax)
plt.grid()

fig3 = plt.figure(figsize=(8, 6), dpi=80)
showpwmripple(fig3,t,0.75,0.25,centeralign=True);

fig4 = plt.figure(figsize=(8, 6), dpi=80)
Da,Db = 0.75, 0.25
intcodes = 'ABCD'
axlist = showpwmripple(fig4,t,Da,Db,centeralign=True)
for (i,tcenter) in enumerate([0.5, 1-(Da+Db)/4, 1, 1+(Da+Db)/4]):
    y = (i/2) * 1.1 + 0.55
    intcode = '[%s]' % intcodes[i]
    axlist[0].annotate(intcode,xy=(tcenter,y),
        horizontalalignment='center',
        verticalalignment='center')
    axlist[2].annotate(intcode,xy=(tcenter,0.05),
        horizontalalignment='center',
        verticalalignment='center')

plt.show()
show1period(0.75,0.25)
