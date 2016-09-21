__author__ = 'PatDen00'

import numpy as np

def digitalplotter(t, *signals):
# return a plotting function that takes an axis and plots
# digital signals (or other signals in the 0-1 range)
  def f(ax):
    n = len(signals)
    for (i,sig) in enumerate(signals):
      ofs = (n - 1 - i)*1.1
      plotargs = []
      for y in sig[1:]:
        if isinstance(y,str):
          plotargs += [y]
        else:
          plotargs += [t, y + ofs]
      ax.plot(*plotargs)
    ax.set_yticks((n - 1 - np.arange(n))*1.1 + 0.55)
    ax.set_yticklabels([sig[0] for sig in signals])
    ax.set_ylim(-0.1, n*1.1)
  return f