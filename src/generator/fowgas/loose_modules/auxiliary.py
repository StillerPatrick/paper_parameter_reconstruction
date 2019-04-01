import numpy as np
import os
from matplotlib import pyplot as plt

def array2image(a):
    return np.swapaxes(a, 0, 1)

def logimage(a):
    return np.log(a+1)

def circle(r, l):
    """
    """
    arr = np.zeros((l, l))
    for i in np.arange(0, l):
        for j in np.arange(0, l):
            if ((0.5*l - i)**2+(0.5*l - j)**2) < r**2:
                arr[i,j] = 1
    return arr

def imshow(a, **kwargs):
    """
    """
    if not 'interpolation' in kwargs:
        kwargs['interpolation']='nearest'
    if not 'origin' in kwargs:
        kwargs['origin']='lower'
    if not 'cmap' in kwargs:
        kwargs['cmap']=plt.cm.gray
    
    plt.imshow(a, **kwargs)

theta = lambda x : 0.5*(1. + np.sign(x))

def get_representation_of_complex(x):
    """
    """
    real = np.real(x)
    imag = np.imag(x)
    abso = np.abs(x)
    pi   = np.max(abso)*np.ones(shape=x.shape)
    angl = np.angle(x)/np.pi*pi
    return real, imag, abso, angl, pi

def plot_complex(x, y, **kwargs):
    """
    """
    axes  = kwargs.pop('axes', None)
    preal = kwargs.pop('real', True)
    pimag = kwargs.pop('imag', True)
    pabso = kwargs.pop('abso', True)
    pangl = kwargs.pop('angl', True)
    
    real, imag, abso, angl, pi = get_representation_of_complex(y)
    def inner(t):
        if pangl:
            t.plot(x, pi,   'k',  label='+pi, scaled to equal max(absolute)', **kwargs)
            t.plot(x, -pi,  'k',  label='-pi, scaled to equal -max(absolute)', **kwargs)
            t.plot(x, angl, 'c',  label='phase, scaled like +/- pi', **kwargs)
        if preal:
            t.plot(x, real, 'b',  label='real part', **kwargs)
        if pimag:
            t.plot(x, imag, 'g',  label='imaginary part', **kwargs)
        if pabso:
            t.plot(x, abso, 'r',  label='absolute', **kwargs)

    if axes:
        inner(axes)
        
    else:
        inner(plt)
        
def swap_center_border(x):
    """
    """
    temp         = np.zeros(shape=x.shape)
    n            = len(x)
    temp[0:n//2] = x[n//2:n]
    temp[n//2:n] = x[0:n//2]
    return temp

def rcoords2tcoords(x, y, z):
    """
    Tranform ray/scattering coordinates to target coordinates.
    """
    return (-x-z)/np.sqrt(2), y, (x-z)/np.sqrt(2)

def tcoords2rcoords(xp, yp, zp):
    """
    Tranform target coordinates to ray/scattering coordinates.
    """
    return (-xp+zp)/np.sqrt(2), yp, (-xp-zp)/np.sqrt(2)

def make_tcoords2ttcoords(alpha):
    """
    Transform untilted target coordinates to tilted target coordinates.
    """
    T = lambda xp, yp, zp : (np.cos(alpha)*xp + np.sin(alpha)*yp, -np.sin(alpha)*xp + np.cos(alpha)*yp, zp)
    return T

def make_ttcoords2tcoords(alpha):
    """
    Transform tilted target coordinates to untilted target coordinates.
    """
    T = lambda xp, yp, zp : (np.cos(-alpha)*xp + np.sin(-alpha)*yp, -np.sin(-alpha)*xp + np.cos(-alpha)*yp, zp)
    return T

def theta(x):
    return 0.5*(1. + np.sign(x))

def i2x(i, n, extent):
    """
    Calculate the real position of the left border of a sub-interval given by its index within the full interval.
    
    Arguments:
    i = index of sub-interval
    n = number of sub-intervals within the interval
    extent = real positions of left border of the left-most sub-interval and of the right border of the right-most sub-interval.
    
    Return:
    Real position of the left border of the sub-interval given by i.
    """
    return extent[0] + i * (extent[1]-extent[0])/n

def x2i(x, n, extent):
    """
    Calculate the (zero-based) index of a value in a discretized interval.
    
    Arguments:
    x = postion
    n = number of sub-intervals within the interval
    extent = real positions of left border of the left-most sub-interval and of the right border of the right-most sub-interval.
    
    Return:
    Zero-based index of x within the interval. Left sub-interval borders belong to the sub-interval, right sub-interval borders don't.
    """
    return int((x-extent[0])*n/(extent[1]-extent[0]))

def feature_pixels(feature_size, total_size, total_pixels):
    """
    Calculate the number of pixels within a real length.
    
    Arguments:
    feature_size = input length
    total_size   = reference length
    total_pixels = number f pixels within the reference length.
    
    Return:
    Number of pixels (rounded up).
    """
    return int((total_pixels * feature_size + total_size - 1)/ total_size)

def path_exists(path, **kwargs):
    """
    Test if a given path already exists on the os.
    
    Arguments:
    -----
    path = string that is tested as potential pathname
    
    Keyword arguments:
    -----
    mode = 'Fail'     - an exception is raised if the path exists
           'Continue' - a message is printed if the path exists
    
    Results:
    -----
    True if path exists, False otherwise.
    """
    mode = kwargs.pop('mode', 'Fail')
    if not mode in ('Fail', 'Continue'):
        raise Exception('Invalid argument for keyword \'mode\'.')
    
    if os.path.lexists(path):
        if mode == 'Fail':
            raise Exception('Path \"{}\" exists. An exception was raised.'.format(path))
        elif mode == 'Continue':
            print('Path \"{}\" exists. Execution continues normally.'.format(path))
        return True
    else:
        return False

def evp(a):
    """
    Evaluate parameter.
    """
    if callable(a):
        return a()
    else:
        return a

def transpose_extent(a):
    return [a[2], a[3], a[0], a[1]]