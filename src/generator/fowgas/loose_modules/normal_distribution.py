import numpy as np

# Definition of normal distribution density function
def make_normal_distribution_density(sigma=1, mu=0, **kwargs):
    normalize = kwargs.pop('normalize', True)
    prefactor = kwargs.pop('prefactor', 1.)
    if normalize:
        alpha = 1. / sigma / np.sqrt(2.*np.pi)
    else:
        alpha = 1.
    
    f = lambda x : prefactor * alpha * np.exp(-0.5*((x-mu)/sigma)**2)
    return f

def aaa():
    return 1