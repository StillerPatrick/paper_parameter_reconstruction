import numpy as np
from matplotlib import pyplot as plt

from auxiliary import theta

def make_density(grating, jitter):
    """
    Make density function.
    
    Arguments:
    -----
    grating = Function whose graph represents the grating edge in the zp=0=const slice.
    jitter = Function that represents the change of the edge postion yp along zp axis.
    
    Results:
    -----
    Density function f(xp, yp, zp) of grating.
    """
    f = lambda xp, yp, zp : 1 - theta(xp - grating(yp - jitter(zp))) - theta(-xp)
    return f

def make_foil_density(thick):
    """
    Make density function of foil.
    
    Arguments:
    -----
    thick = Thickness of foil
    
    Results:
    -----
    Density function f(xp, yp, zp) of foil, foil is right below xp=0.
    """
    f = lambda xp, yp, zp : theta(xp+thick) * theta(-xp)
    return f