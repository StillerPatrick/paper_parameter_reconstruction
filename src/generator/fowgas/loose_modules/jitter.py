import numpy as np
from matplotlib import pyplot as plt

def make_jitter(amplitude, wavelength):
    """
    Make function that represents the change of the edge postion yp along zp axis. Modelled as sine function.
    
    Arguments:
    -----
    amplitude = Difference between maximum and minimum of jitter function value.
    wavelength = Difference between two neighboring maximmum positions.
    
    Return:
    -----
    Function (of target zp coordinate).
    
    """
    f = lambda zp : amplitude * 0.5 * np.sin(2.*np.pi * zp / wavelength)
    return f
