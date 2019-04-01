import numpy as np

def perfect_grating(pitch, horiz_feat_size):
#    return np.vectorize(lambda x: 1. if (x%pitch)<horiz_feat_size else 0.)
    return lambda x: (x%pitch)<horiz_feat_size

def symm_feat(hfs, f):
    """
    Make a symmetrical feature function from an edge function.
    
    Arguments:
    hfs -- Horizontal feature size.
    f   -- The input edge function. It's assumed to start from 0. at -0.5 and rise to 1. at 0.5
    
    Return:
    A lambda function object of a function of x. It starts from 0. at -1. and ends with 0. at 1.
    """
#    return np.vectorize(lambda x: f(x+0.5*hfs) if x<0. else f(-x+0.5*hfs))
    return lambda x: f(x+0.5*hfs) * (x<0.) + f(-x+0.5*hfs) * (x>=0.)

def feature_grating(pitch, f):
    """
    Make a grating function from a feature function.
    
    Arguments:
    pitch -- The distance between two neighboring features.
    f     -- The input feature function. It's assumed to start from 0. at -1. and end with 0. at 1.
    
    Return:
    A lambda function object of a function of x that covers the real axis.
    """
#    return np.vectorize(lambda x: f((x%pitch)-0.5*pitch))
    return lambda x: f((x%pitch)-0.5*pitch)
