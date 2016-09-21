__author__ = 'PatDen00'

def ramp(t): return t % 1

def sawtooth(t): return 1 - abs(2*(t % 1) - 1)

def pwm(t, D, centeralign = False):
# generate PWM signals with duty cycle D
  return ((sawtooth(t) if centeralign else ramp(t)) <= D) * 1.0