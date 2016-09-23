__author__ = 'PatDen00'

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

# create a time vector
t = np.arange(0,4,0.5)
t = np.linspace(0,4,100)

T = t[-1] - t[0]
print(T)
