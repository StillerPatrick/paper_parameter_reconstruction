import numpy as np
from scipy import signal

# append path for loading own packages
import sys
sys.path.append("/home/mz6084/2016-09_Ipython-Notebooks_zu_LN04/packages")

# import own packages
import auxiliary as aux

def gaussian_kernel(sigmax, sigmay, pixelsize, nsigma=3):
    """
    Make a Gaussian blurring kernel.
    
    Arguments:
    -----
    sigmax    = Gaussian sigma parameter for x axis
    sigmay    = Gaussian sigma parameter for y axis
    pixelsize = real size of on pixel - assumed the same in both directions
    nsigma    = number of sigma covered by the kernel - same for both directions
    
    Result:
    -----
    The kernel.
    """
    nx = 0
    ny = 0
    f = 0
    
    if (sigmax != 0) and (sigmay != 0):
        f = lambda x, y : np.exp(-x**2/(2*sigmax**2)-y**2/(2*sigmay**2))
        nx = int(2*nsigma*sigmax/pixelsize)
        ny = int(2*nsigma*sigmay/pixelsize)
    
    if sigmax == 0 and sigmay != 0:
        f = lambda x, y : (x==0) * np.exp(-y**2/(2*sigmay**2))
        nx = 1
        ny = int(2*nsigma*sigmay/pixelsize)
    
    if sigmax != 0 and sigmay == 0:
        f = lambda x, y : np.exp(-x**2/(2*sigmax**2)) * (y==0)
        nx = int(2*nsigma*sigmax/pixelsize)
        ny = 1
    
    if sigmax == 0 and sigmay == 0:
        f = lambda x, y : (x==0) * (y==0)
        nx = 1
        ny = 1
        
    xmin = -nsigma * sigmax
    xmax =  nsigma * sigmax
    ymin = -nsigma * sigmay
    ymax =  nsigma * sigmay
    
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    
    xx, yy = np.meshgrid(x, y)
    
    # meshgrid compensates for 'imshow axes swap' behavior, which is unwanted here -> undo with .T
    ff = f(xx.T, yy.T)
    return ff/ff.sum()

def expand_coordinates(c, kernelsize):
    """
    Expand an array c that represents a linear coordinate axis with correctly spaced values to left and right
    so that smoothing an array that holds values over the expanded coordinate axis with a kernel of size
    kernelsize along that axis will result in the smoothed values over the original array c.
    
    Arguments:
    -----
    c = original linear coordinate array
    kernelsize = size of the kernel that will be used for smoothing with function 'smooth' (or scipy.signal.convolve(mode='valid'))
    
    Result:
    -----
    Expanded array.
    """
    c_pxsize = (c.max()-c.min())/(c.size-1)
    c_newmin = c.min() - int((kernelsize-1)/2) * c_pxsize
    c_newmax = c.max() + int((kernelsize  )/2) * c_pxsize
    c_new    = np.linspace(c_newmin, c_newmax, c.size + kernelsize - 1)
    return c_new

def smooth(a, b, extent):
    """
    Convolve the 2D image of a sharp edge with some blurring kernel.
    
    Arguments:
    a = image
    b = blurring kernel
    extent = real positions of:
             - left border of the left-most pixel
             - right border of the right-most pixel
             - bottom border of the bottom-most pixel
             - top border of the top-most pixel
             of the input image
    
    Return:
    Convolved image, extent of convolved image.
    """
    out = signal.convolve2d(a, b, mode='valid')
    
    ori_shape = a.shape
    out_shape = out.shape
    
    out_xmin = aux.i2x((ori_shape[0]-out_shape[0])/2,                ori_shape[0], extent[0:2])
    out_xmax = aux.i2x((ori_shape[0]-out_shape[0])/2 + out_shape[0], ori_shape[0], extent[0:2])
    out_ymin = aux.i2x((ori_shape[1]-out_shape[1])/2,                ori_shape[1], extent[2:4])
    out_ymax = aux.i2x((ori_shape[1]-out_shape[1])/2 + out_shape[1], ori_shape[1], extent[2:4])
    out_extent = [out_xmin, out_xmax, out_ymin, out_ymax]
    
    return out, out_extent

def pad(a, nx, ny, extent):
    """
    Symmetrically pad an input image at its borders copying the outmost pixel values to have exactly the desired array shape. 
    
    Arguments:
    a = input image
    nx = desired number of pixels in x-direction
    ny = desired number of pixels in y-direction
    extent = real positions of:
             - left border of the left-most pixel
             - right border of the right-most pixel
             - bottom border of the bottom-most pixel
             - top border of the top-most pixel
             of the input image 
    
    Return:
    Padded image, extent of padded image.
    """
    is_nx = a.shape[0]
    is_ny = a.shape[1]
    
    pad_nx = nx - is_nx
    pad_ny = ny - is_ny
    
    left_pad_nx = pad_nx//2
    right_pad_nx = (pad_nx+1)//2
    bottom_pad_ny = pad_ny//2
    top_pad_ny = (pad_ny+1)//2
    
    left_pad_ix = 0
    right_pad_ix = is_nx + left_pad_nx
    bottom_pad_iy = 0
    top_pad_iy = is_ny + bottom_pad_ny
    
    original_ix = left_pad_nx
    original_iy = bottom_pad_ny
    
    t = np.zeros(shape=(nx, ny))
    t[original_ix:original_ix + is_nx, original_iy:original_iy + is_ny] = a
    for ix in range(left_pad_nx):
        t[ix, original_iy:original_iy + is_ny] = a[0, :]
    for ix in range(right_pad_nx):
        t[ix + right_pad_ix, original_iy:original_iy + is_ny] = a[-1, :]
    for iy in range(bottom_pad_ny):
        t[original_ix:original_ix + is_nx, iy] = a[:, 0]
    for iy in range(top_pad_ny):
        t[original_ix:original_ix + is_nx, iy + top_pad_iy] = a[:, -1]
    t[left_pad_ix:left_pad_nx, bottom_pad_iy:bottom_pad_ny] = a[0, 0]
    t[right_pad_ix:right_pad_nx+right_pad_ix, bottom_pad_iy:bottom_pad_ny] = a[-1, 0]
    t[left_pad_ix:left_pad_nx, top_pad_iy:top_pad_ny+top_pad_iy] = a[0, -1]
    t[right_pad_ix:right_pad_nx+right_pad_ix, top_pad_iy:top_pad_ny+top_pad_iy] = a[-1, -1]
    
    out_extent = []
    out_extent.append(extent[0] - left_pad_nx   * (extent[1]-extent[0])/is_nx)
    out_extent.append(extent[1] + right_pad_nx  * (extent[1]-extent[0])/is_nx)
    out_extent.append(extent[2] - bottom_pad_ny * (extent[3]-extent[2])/is_ny)
    out_extent.append(extent[3] + top_pad_ny    * (extent[3]-extent[2])/is_ny)
    
    return t, out_extent

def symm_feat(a, extent):
    """
    Make an x-symmetrical by mirroring a given image at its right border.
    
    Arguments:
    a = input image
    extent = real positions of:
             - left border of the left-most pixel
             - right border of the right-most pixel
             - bottom border of the bottom-most pixel
             - top border of the top-most pixel
             of the input image 
    
    Return:
    Symmetrical image, extent of symmetrical image.
    """
    f = np.zeros(shape=(2*a.shape[0], a.shape[1]))
    f[0:a.shape[0], :] = a
    f[a.shape[0]:f.shape[0], :] = a[::-1, :]
    return f, [extent[0], 2*extent[1]-extent[0], extent[2], extent[3]]

def crop(a, extent, crop_extent):
    """
    Take a rectangular part out of a given image.
    
    Argument:
    -----
    a           = input image
    extent      = real positions of:
                  - left border of the left-most pixel
                  - right border of the right-most pixel
                  - bottom border of the bottom-most pixel
                  - top border of the top-most pixel
                  of the input image 
    crop_extent = desired extent of cropped image
    
    Return:
    -----
    Cropped image, extent of cropped image - usually slightly differs from the desired extent!
    """
    nx = a.shape[0]
    ny = a.shape[1]
    
    ixmin = aux.x2i(crop_extent[0], nx, extent[0:2])
    ixmax = aux.x2i(crop_extent[1], nx, extent[0:2])
    iymin = aux.x2i(crop_extent[2], ny, extent[2:4])
    iymax = aux.x2i(crop_extent[3], ny, extent[2:4])
    
    out_extent = [aux.i2x(ixmin, nx, extent[0:2]), \
                  aux.i2x(ixmax, nx, extent[0:2]), \
                  aux.i2x(iymin, ny, extent[2:4]), \
                  aux.i2x(iymax, ny, extent[2:4]), \
                  ]
    return a[ixmin:ixmax, iymin:iymax], out_extent

def tile(a, extent, nx, ny):
    """
    """
    out = np.tile(a, (nx, ny))
    out_extent = [extent[0],\
                   nx*extent[1] - (nx-1)*extent[0],\
                   extent[2],\
                   ny*extent[3] - (ny-1)*extent[2]]
    return out, out_extent

def translate(a, extent, dx, dy):
    """
    """
    pixelsize_x = (extent[1] - extent[0]) / a.shape[0]
    pixelsize_y = (extent[3] - extent[2]) / a.shape[1]
    
    dix = int(dx/pixelsize_x)
    diy = int(dy/pixelsize_y)
    
    out_extent = [aux.i2x(             dix, a.shape[0], extent[0:2]), \
                  aux.i2x(a.shape[0] + dix, a.shape[0], extent[0:2]), \
                  aux.i2x(             diy, a.shape[1], extent[2:4]), \
                  aux.i2x(a.shape[1] + diy, a.shape[1], extent[2:4])]
    
    return a, out_extent

def roll(a, extent, dx, dy):
    """
    """
    temp1, temp1_extent = tile(a, extent, 2, 2)
    temp2, temp2_extent = crop(temp1, temp1_extent, [extent[0]+dx, extent[1]+dx, extent[2]+dy, extent[3]+dy])
    return translate(temp2, temp2_extent, dx, dy)

def concatenate(a, extent, **kwargs):
    """
    """
    if (type(a) != tuple) or (type(extent) != tuple):
        raise Exception('2 positional tuple arguments are exspected: image tuple, extent tuple.')
    
    if kwargs['axis'] == 0:
        out_extent = [extent[0][0], extent[-1][1], extent[0][2], extent[0][3]]
    elif kwargs['axis'] == 1:
        out_extent = [extent[0][0], extent[0][1], extent[0][2], extent[-1][3]]
    
    #error = 0
    #for i in range(len(a)):
    #    if kwargs['axis'] == 0:
    #        error += (extent[0][2] != extent[i][2])
    #        error += (extent[0][3] != extent[i][3])
    #    elif kwargs['axis'] == 1:
    #        error += (extent[0][0] != extent[i][0])
    #        error += (extent[0][1] != extent[i][1])
    #
    #if error != 0:
    #    for i in range(len(a)):
    #        print(extent[i])
    #    raise Exception('Extents of images do not match')
    #
    #if not (kwargs['axis'] == 0) or (kwargs['axis'] == 1):
    #    raise Exception('Invalid axis argument')
            
    return np.concatenate(a, **kwargs), out_extent