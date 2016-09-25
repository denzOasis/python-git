__author__ = 'PatDen00'

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

def ramp(t): return t % 1


def sawtooth(t, shift=False):
    return 1 - abs(2 * (t % 1) - 1) if shift else abs(2 * (t % 1) - 1)


def pwm(t, D, centeralign=False, shift=False):
    # generate PWM signals with duty cycle D
    return ((sawtooth(t, shift) if centeralign else ramp(t)) <= D) * 1.0


def rms(t, y):
    return np.sqrt(np.trapz(y * y, t) / (np.amax(t) - np.amin(t)))


def digitalplotter(t, *signals):
    # return a plotting function that takes an axis and plots
    # digital signals (or other signals in the 0-1 range)

    def f(ax):
        n = len(signals)
        for (i, sig) in enumerate(signals):
            ofs = (n - 1 - i) * 1.1
            plotargs = []
            for y in sig[1:]:
                if isinstance(y, str):
                    plotargs += [y]
                else:
                    plotargs += [t, y + ofs]
            ax.plot(*plotargs)
        ax.set_yticks((n - 1 - np.arange(n)) * 1.1 + 0.55)
        ax.set_yticklabels([sig[0] for sig in signals])
        ax.set_ylim(-0.1, n * 1.1)

    return f


def extendrange(ra, rb):
    # return a tuple (x1,x2) representing the interval from x1 to x2,
    # given two input ranges of the same form, or None (representing no input).'''
    if ra is None:
        return rb
    elif rb is None:
        return ra
    else:
        return (min(ra[0], rb[0]), max(ra[1], rb[1]))


def createLimits(margin, *args):
    # add proportional margin to an interval:
    # createLimits(0.1, (1,3),(2,4),(0,2)) calculates the maximum extent
    # of the ranges provided, in this case (0,4), and adds another 0.1 (10%)
    # to the extent symmetrically, thus returning (-0.2, 4.2).
    r = None
    for x in args:
        r = extendrange(r, (np.min(x), np.max(x)))
    rmargin = (r[1] - r[0]) * margin / 2.0
    return (r[0] - rmargin, r[1] + rmargin)


def calcripple(t, y):
    # calculate ripple current by integrating the input,
    # then adjusting it by a linear function to put both endpoints
    # at the same value. The slope of the linear function is the
    # mean value of the input; the offset is chosen to make the mean value
    # of the output ripple current = 0.
    T = t[-1] - t[0]
    yint0 = np.append([0], scipy.integrate.cumtrapz(y, t))
    # cumtrapz produces a vector of length N-1
    # so we need to add one element back in at the beginning
    meanval0 = yint0[-1] / T
    yint1 = yint0 - (t - t[0]) * meanval0
    meanval1 = scipy.integrate.trapz(yint1, t) / T
    return yint1 - meanval1


def annotate_level(ax, t, yline, ytext, text, style='k:'):
    tlim = [min(t), max(t)]
    tannot0 = tlim[0] + (tlim[1] - tlim[0]) * 0.5
    tannot1 = tlim[0] + (tlim[1] - tlim[0]) * 0.6
    ax.plot(tlim, [yline] * 2, style)
    # see:
    # http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.annotate
    ax.annotate(text, xy=(tannot0, yline), xytext=(tannot1, ytext),
                bbox=dict(boxstyle="round", fc="0.9"),
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc,angleA=0,armA=20,angleB=%d,armB=15,rad=7"
                                                % (-90 if ytext < yline else 90)))


def annotate_ripple(ax, t, Xab, VI):
    # annotate with peak values
    for y in [min(Xab), max(Xab)]:
        yofs = y
        if (VI == 0):
            annotate_level(ax, t, y, yofs * 0.3, '$I_{Ldc} %+.5f$' % yofs)
        else:
            annotate_level(ax, t, y, yofs * 0.3, '$V_{Cdc} %+.5f$' % yofs)


def showripple_HB1(fig, t, Va, Vb, titlestring):

    axlist = []
    margin = 0.1

    vCMp = Va - 0.5
    iCMripplep = 2*calcripple(t, vCMp)
    vCMn = Vb - 0.5
    iCMripplen = 2 * calcripple(t, vCMn)
    iC = iCMripplep + iCMripplen

    ax = fig.add_subplot(3, 1, 1)
    digitalplotter(t, ('Da', Va), ('Db', Vb))(ax)
    ax.set_ylabel('branch duty cycles')
    ax.grid(True)
    axlist.append(ax)

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(t, vCMp)
    ax.plot(t, vCMn)
    ax.set_ylim(createLimits(margin, vCMp))
    ax.set_ylabel('CM voltage [V]')
    ax.grid(True)
    axlist.append(ax)

    ax = fig.add_subplot(3, 1, 3)
    ax.plot(t, iCMripplep)
    ax.plot(t, iCMripplen)
    ax.plot(t, iC)
    ax.set_ylim(createLimits(margin, iCMripplep))
    ax.set_ylabel('CM current [A]')
    ax.grid(True)
    axlist.append(ax)
    annotate_ripple(ax, t, iCMripplep, 0)

    fig.suptitle(titlestring, fontsize=16)
    return axlist

def showripple_HB2(fig, t, Va, Vb, titlestring):

    axlist = []
    margin = 0.1

    vCMp = Va - 0.5
    iCMripplep = 2*calcripple(t, vCMp)
    vCMn = Vb - 0.5
    iCMripplen = 2 * calcripple(t, vCMn)
    iC = iCMripplep + iCMripplen
    vC = calcripple(t, iC)

    ax = fig.add_subplot(4, 1, 1)
    digitalplotter(t, ('Da', Va), ('Db', Vb))(ax)
    ax.set_ylabel('branch duty cycles')
    ax.grid(True)
    axlist.append(ax)

    ax = fig.add_subplot(4, 1, 2)
    ax.plot(t, vCMp)
    ax.plot(t, vCMn)
    ax.set_ylim(createLimits(margin, vCMp))
    ax.set_ylabel('CM voltage [V]')
    ax.grid(True)
    axlist.append(ax)

    ax = fig.add_subplot(4, 1, 3)
    ax.plot(t, iCMripplep)
    ax.plot(t, iCMripplen)
    ax.plot(t, iC)
    ax.set_ylim(createLimits(margin, iCMripplep))
    ax.set_ylabel('CM current [A]')
    ax.grid(True)
    axlist.append(ax)
    annotate_ripple(ax, t, iCMripplep, 0)

    ax = fig.add_subplot(4, 1, 4)
    ax.plot(t, vC)
    ax.set_ylim(createLimits(margin, vC))
    ax.set_ylabel('Capacitor ripple [V]')
    ax.grid(True)
    axlist.append(ax)
    annotate_ripple(ax, t, vC, 1)

    fig.suptitle(titlestring, fontsize=16)
    return axlist


def showCMripple(fig, t, Va, Vb, titlestring):

    axlist = []
    margin = 0.1

    # ripple voltage in inductor is determined by the common mode voltage across it
    vCMp = Va - 0.5
    iCMripplep = 2*calcripple(t, vCMp)    # twice because half the DM inductance
    vCMn = -Vb + 0.5
    iCMripplen = 2 * calcripple(t, vCMn)  # twice because half the DM inductance

    ax = fig.add_subplot(3, 1, 1)
    digitalplotter(t, ('Da', Va), ('Db', Vb))(ax)
    ax.set_ylabel('branch duty cycles')
    ax.grid(True)
    axlist.append(ax)

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(t, vCMp)
    ax.plot(t, vCMn)
    ax.set_ylim(createLimits(margin, vCMp))
    ax.set_ylabel('CM voltage [V]')
    ax.grid(True)
    axlist.append(ax)

    ax = fig.add_subplot(3, 1, 3)
    ax.plot(t, iCMripplep)
    ax.plot(t, iCMripplen)
    ax.set_ylim(createLimits(margin, iCMripplep))
    ax.set_ylabel('CM current [A]')
    ax.grid(True)
    axlist.append(ax)
    annotate_ripple(ax, t, iCMripplep, 0)

    fig.suptitle(titlestring, fontsize=16)
    return axlist


def showDMripple(fig, t, Va, Vb, titlestring):

    axlist = []
    margin = 0.1

    # only the differential part of the common mode ripple causes a capacitor
    # voltage ripple since the common mode part cancels out
    vDM = Va - Vb
    iDMripple = calcripple(t, vDM)
    vDMripple = calcripple(t, iDMripple)

    ax = fig.add_subplot(4, 1, 1)
    digitalplotter(t, ('Da', Va), ('Db', Vb))(ax)
    ax.set_ylabel('branch duty cycles')
    ax.grid(True)
    axlist.append(ax)

    ax = fig.add_subplot(4, 1, 2)
    ax.plot(t, vDM)
    ax.set_ylim(createLimits(margin, vDM))
    ax.set_ylabel('DM voltage [V]')
    ax.grid(True)
    axlist.append(ax)

    ax = fig.add_subplot(4, 1, 3)
    ax.plot(t, iDMripple)
    ax.set_ylim(createLimits(margin, iDMripple))
    ax.set_ylabel('DM current [A]')
    ax.grid(True)
    axlist.append(ax)
    annotate_ripple(ax, t, iDMripple, 0)

    ax = fig.add_subplot(4, 1, 4)
    ax.plot(t, vDMripple)
    ax.set_ylim(createLimits(margin, vDMripple))
    ax.set_ylabel('Capacitor ripple [V]')
    ax.grid(True)
    axlist.append(ax)
    annotate_ripple(ax, t, vDMripple, 1)

    fig.suptitle(titlestring, fontsize=16)
    return axlist


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
  ax.grid(True)
  # now annotate
  (D1,D2,V) = (Db,Da,1) if Da > Db else (Da,Db,-1)
  tpts = np.array([0, D1/2, D2/2, 0.5, 1-(D2/2), 1-(D1/2), 1])
  dtpts = np.append([0],np.diff(tpts))
  vpts = np.array([0, 0, V, 0, 0, V, 0]) - (Da-Db)
  ypts = np.cumsum(vpts*dtpts)
  ax.plot(tpts,ypts,'.',markersize=8)
  for i in range(7):
    ax.annotate('$t_%d$' % i,xy=(tpts[i]+0.01,ypts[i]), fontsize=16)


def calc_capacitor_ripple(pwmA,pwmB,I_L,I_S):
    return (pwmA-pwmB)*I_L - I_S


def showripple2(fig, t, pwmA, pwmB, I_Ldc, I_S, titlestring=''):
    # plot ripple current in inductor and capacitor
    # as well as phase duty cycles and load voltage
    axlist = []
    Iabripple = calcripple(t, pwmA - pwmB)
    Iab = Iabripple + I_Ldc
    Icdc = calc_capacitor_ripple(pwmA, pwmB, Iab, I_S)

    margin = 0.1
    ax = fig.add_subplot(4, 1, 1)
    digitalplotter(t, ('Va', pwmA), ('Vb', pwmB))(ax)
    ax.set_ylabel('Phase duty cycles')
    ax.grid(True)
    axlist.append(ax)

    ax = fig.add_subplot(4, 1, 2)
    ax.plot(t, pwmA - pwmB)
    ax.set_ylim(createLimits(margin, pwmA - pwmB))
    ax.set_ylabel('Load voltage')
    ax.grid(True)
    axlist.append(ax)

    ax = fig.add_subplot(4, 1, 3)
    ax.plot(t, Iab)
    ax.set_ylim(createLimits(margin, Iab))
    ax.set_ylabel('Ripple current')
    ax.grid(True)
    axlist.append(ax)

    ax = fig.add_subplot(4, 1, 3)
    ax.plot(t, Iab)
    ax.set_ylim(createLimits(margin, Iab))
    ax.set_ylabel('$I_L$', fontsize=14)
    ax.grid(True)
    axlist.append(ax)
    annotate_ripple(ax, t, Iab, 0)

    ax = fig.add_subplot(4, 1, 4)
    ax.plot(t, Icdc)
    ax.set_ylim(createLimits(margin, Icdc))
    ax.set_ylabel('$I_{CDC}$', fontsize=14)
    ax.grid(True)
    axlist.append(ax)

    fig.suptitle(titlestring, fontsize=16)

    return axlist


def showpwmripple2(fig, t, Da, Db, I_Ldc, centeralign=False):
    return showripple2(fig, t,
                       pwm(t, Da, centeralign),
                       pwm(t, Db, centeralign),
                       I_Ldc,
                       (Da - Db) * I_Ldc,
                       titlestring='%s-aligned pwm, $D_a$=%.3f, $D_b$=%.3f, $I_{Ldc}$=%.3f' %
                                   ('Center' if centeralign else 'Edge', Da, Db, I_Ldc))


def split_Icdc_1(fig, t, Da, Db, I_Ldc, centeralign=True):
    axlist = []
    pwmA = pwm(t, Da, centeralign)
    pwmB = pwm(t, Db, centeralign)
    I_S = (Da - Db) * I_Ldc
    s = np.sign(Da - Db)
    Iabripple = calcripple(t, pwmA - pwmB)
    Iab = Iabripple + I_Ldc
    Icdc = calc_capacitor_ripple(pwmA, pwmB, Iab, I_S=0)
    Icdc_mean = np.mean(Icdc)

    ax = fig.add_subplot(3, 1, 1)
    ax.plot(t, -I_S * np.ones_like(t))
    ax.set_ylabel('$-I_S$', fontsize=15)
    ax.grid(True)
    axlist.append(ax)

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(t, Icdc, t, Iab * s, '--r')
    ax.set_ylabel('$I_{CDC}: I_S = 0$', fontsize=15)
    ax.grid(True)
    annotate_level(ax, t, Icdc_mean, np.max(Icdc) / 3, 'mean = %.3f' % Icdc_mean)
    axlist.append(ax)
    fig.text(0.1, 0.5, '+', fontsize=28)

    ax = fig.add_subplot(3, 1, 3)
    ax.plot(t, Icdc - I_S)
    ax.set_ylabel('$I_{CDC}$', fontsize=15)
    ax.grid(True)
    annotate_level(ax, t, Icdc_mean - I_S, (np.min(Icdc) - I_S) / 3,
                   'mean = %.3f' % (Icdc_mean - I_S))
    axlist.append(ax)
    fig.text(0.1, 0.22, '=', fontsize=28)

    plt.subplots_adjust(left=0.25)
    return axlist