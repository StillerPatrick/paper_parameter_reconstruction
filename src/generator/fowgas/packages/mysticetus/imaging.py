import numpy as np
from PIL import Image
from scipy.signal import fftconvolve as convolve



def center_and_rotate(array, center, angle):
    """
    Translate the input array so that the pixel indicated by 'center' coincides with the center of the output array
    and rotate by a given angle around that pixel.
    
    * Output center:
        Let 'size' be the number of pixels along one axis. In zero-based indexing, the index of the output 'center pixel' along that axis is defined as
        (size + 1) // 2,
        '//' is the integer division operator.
    * Output shape:
        same as array.shape
    * Interpolation for rotation:
        nearest
    * padding:
        input is assumed to be surrounded by pixels of constant value zero
    
    array  - input array, first dimension is y axis, second is x axis
    center - center position as pixel index pair, fist is y index, second is x index
    angle  - rotation angle in radians, positive angles refer to (the usual) counterclockwise rotation right-handed x,y axes
    """
    sizeY, sizeX = array.shape
    centerY, centerX = center
    
    padXLow     = sizeX - centerX - 1
    padXHigh    = centerX
    padYLow     = sizeY - centerY - 1
    padYHigh    = centerY
    padding     = [[padYLow, padYHigh], [padXLow, padXHigh]]
    padded      = np.pad(array, padding, 'constant', constant_values=0)
    
    rotated     = np.array(Image.fromarray(padded).rotate(-angle*180/np.pi, expand=False))
    
    cropXLow    = (padXLow + padXHigh)//2
    cropXHigh   =  padXLow + padXHigh - cropXLow
    cropYLow    = (padYLow + padYHigh)//2
    cropYHigh   =  padYLow + padYHigh - cropYLow
    
    cropped     = rotated[cropYLow:rotated.shape[0]-cropYHigh, cropXLow:rotated.shape[1]-cropXHigh]
    return cropped



def central_similar_portion_xyslice(array, factor):
    """
    Return slice tuple for the central section of array.
    
    array  - the array to slice
    factor - relative part that is kept of each axis
    """
    sizeY   = array.shape[0]
    sizeX   = array.shape[1]
    
    startY  = int((sizeY*(1-factor))//2) 
    stopY   = sizeY - startY
    startX  = int((sizeX*(1-factor))//2) 
    stopX   = sizeX - startX
    
    sx      = slice(startX, stopX)
    sy      = slice(startY, stopY)
    
    return (sy, sx)



def central_xyslice(array, ssize):
    """
    Return slice tuple for the central section of array.
    
    array - the array to slice
    ssize - numbers of pixels in x and y direction as (ny, nx)
    """
    sizeY   = array.shape[0]
    sizeX   = array.shape[1]
    
    ny      = ssize[0]
    nx      = ssize[1]
    
    startX  = (sizeX - nx)//2
    stopX   = startX + nx
    startY  = (sizeY - ny)//2
    stopY   = startY + ny
    
    sx      = slice(startX, stopX)
    sy      = slice(startY, stopY)
    
    return (sy, sx)



def circular_increase_mask(mask, diameter):
    """
    Increase mask by pushing its edge out.
    
    Outward pushing is done in a circular manner.
    True or positive values are considered as inside, False or zero values are considered as outside.
    
    mask   - mask array to increase
    radius - number of pixels to push the mask's edge
    """
    def circle(diameter, a):
        """
        Create circular mask.
        
        diameter - circle's diameter as number of pixels
        a        - parameter for pixel discretisation
        """
        assert(type(diameter)==int)

        #    a          = np.exp(-1e-16) - 1./np.sqrt(2)
        radius     = diameter/2

        #x          = np.linspace(-radius+0.5, radius-0.5, diameter, endpoint=True)
        x          = np.linspace(-radius-0.5, radius+0.5, diameter+2, endpoint=True)
        y          = x
        xx, yy     = np.meshgrid(x, y)
        circle     = np.array(((xx**2+yy**2)<=((radius+a)**2)), dtype=int)

        return circle
    
    n           = np.sum(mask)
    c           = circle(diameter, 0.)
    convolved   = convolve(mask, c, mode='same')
    r           = np.logical_not(np.isclose(convolved, np.zeros(shape=convolved.shape), rtol=0.1/n))
    return np.array(r, dtype=int)



def calc_angle(start, end):
    """
    Calculate the angle in radians between a line segment and the x axis.
    
    start - coordinates of line segment start point as (y, x)
    end   - coordinates of line esement end point as (y, x)
    """
    startY, startX = start
    endY, endX     = end
    distanceX   = endX - startX
    distanceY   = endY - startY
    angle       = np.arctan2(distanceY, distanceX)
    return angle
