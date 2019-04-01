from scipy import ndimage
import numpy as np

def minInputLength(lc):
    """
    """
    return np.ceil(np.sqrt(2)*lc)

def cropRotate(a, lc, alpha):
    """
    """
    ax, ay = a.shape
    if ax != ay:
        raise("Only square images are supported.")
    if lc > np.floor(np.sqrt(2)*ax):
        raise("Output image dimension must not exceed (1/sqrt(2))*(input image dimension).")
    
    temp = ndimage.rotate(a, alpha)
    temp = temp[0.5*(temp.shape[0]-lc):0.5*(temp.shape[0]+lc), 0.5*(temp.shape[1]-lc):0.5*(temp.shape[1]+lc)]
    return temp[0:lc, 0:lc]
