__author__ = 'PatDen00'

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

from functions import *


cpd = 61.09
d = 4 + 31 + 30 + 31 + 31 + 28 + 31 + 30 + 31 + 28
c = cpd*d
print('Cash total: EUR %.2f' % c)

months = 21
cpm = c/months
print('Cash per month: EUR %.2f' % cpm)

sal = 1700
mon1 = 850
perc = mon1/sal*100
print('Percent: %.2f' % perc)

mon2 = cpm*perc/100
print('Überweisung: %.2f' % mon2)

perc2 = mon2/mon1*100/2
print('Percent: %.2f' % perc2)


