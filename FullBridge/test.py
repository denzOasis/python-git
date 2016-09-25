__author__ = 'PatDen00'

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

from functions import *

# create a time vector
t = np.arange(0,4,1e-4)
x1 = sawtooth1(t, shift=False)
x2 = sawtooth1(t, shift=True)
plt.plot(x1)
plt.plot(x2)
plt.show()


