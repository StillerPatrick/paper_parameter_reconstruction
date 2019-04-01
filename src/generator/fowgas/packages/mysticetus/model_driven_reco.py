import numpy as np
import random
from scipy import fftpack as fft
from scipy.optimize import basinhopping
from scipy import signal

from mysticetus.grating_base_functions import *

def dist_model_grating_1D(sigma, pitch, fsize, x):
    """
    Distribution model: 1D grating function
    """
    fct_edge = make_erf_edge(sigma, 0)
    fct_feat = make_box(pitch, fsize, fct_edge)
    fct_grat = make_periodical(pitch, fct_feat, 0.)
    return fct_grat(x)

def dist_model_grating_2D(sigma, pitch, fsize, xx, yy):
    """
    Distribution model: 2D grating function
    """
    return fct_grat(pitch, fsize, sigma, xx, yy)

def meas_model_basic_far_field_intensity_1D(distribution):
    """
    Measurement model: input (1D) -> 1D FFT -> absolute squared -> out
    """
    temp = fft.fft(distribution)
    temp = fft.fftshift(temp)
    temp = np.abs(temp)**2
    return temp

def meas_model_far_field_intensity_and_psf_2D(distribution, **kwargs):
    """
    Measurement model: input (2D) -> 2D FFT -> absolute squared -> convolve with psf -> out
    """
    psf = kwargs.pop('psf')
    
    temp = fft.fft2(distribution)
    temp = fft.fftshift(temp)
    temp = np.abs(temp)**2
    temp = convolve_with_psf(temp, psf)
    return temp

def meas_model_far_field_intensity_and_psf_and_illumination_2D(distribution, illumination, **kwargs):
    """
    Measurement model: input (2D) -> mult with illumination -> 2D FFT -> absolute squared -> convolve with psf -> out
    """
    psf = kwargs.pop('psf')
    
    temp = distribution * illumination
    temp = fft.fft2(temp)
    temp = fft.fftshift(temp)
    temp = np.abs(temp)**2
    temp = convolve_with_psf(temp, psf)
    return temp

def q_measure(synth, meas, **kwargs):
    """
    Quality measure: Squared differences, average
    """
    normalize = kwargs.pop('normalize', True)
    
    # check if mask is present
    has_mask = False
    if 'mask' in kwargs.keys():
        mask = kwargs.pop('mask')
        has_mask = True
    
    # determine overall norms of operands
    norm_meas  = 1
    norm_synth = 1
    if normalize:
        if has_mask:
            norm_meas  = np.sum(meas  * mask)
            norm_synth = np.sum(synth * mask)
        else:
            norm_meas  = np.sum(meas)
            norm_synth = np.sum(synth)
    
    # determine dimensionality
    dim = 0
    if has_mask:
        dim = np.sum(mask)
    else:
        dim = meas.size
    
    # calc squared differences
    diff_sq = ((meas/norm_meas) - (synth/norm_synth))**2
    if has_mask:
        diff_sq = diff_sq * mask
    
    return np.sqrt(np.sum(diff_sq) * dim)

def q_measure_relative_norm(synth, meas, rn, **kwargs):
    """
    Quality measure: Squared differences, average
    """
    normalize = kwargs.pop('normalize', True)
    
    # check if mask is present
    has_mask = False
    if 'mask' in kwargs.keys():
        mask = kwargs.pop('mask')
        has_mask = True
    
    # determine overall norms of operands
    norm_meas  = 1
    norm_synth = 1
    if normalize:
        if has_mask:
            norm_meas  = np.sum(meas  * mask)
            norm_synth = np.sum(synth * mask)
        else:
            norm_meas  = np.sum(meas)
            norm_synth = np.sum(synth)
    
    # determine dimensionality
    dim = 0
    if has_mask:
        dim = np.sum(mask)
    else:
        dim = meas.size
    
    # calc squared differences
    diff_sq = ((meas/norm_meas) - (synth/norm_synth*rn))**2
    if has_mask:
        diff_sq = diff_sq * mask
    
    return np.sqrt(np.sum(diff_sq) * dim)

def q_measure_relative_norm_log(synth, meas, rn, **kwargs):
    """
    Quality measure: Squared differences of log, average
    """
    normalize = kwargs.pop('normalize', True)
    
    # check if mask is present
    has_mask = False
    if 'mask' in kwargs.keys():
        mask = kwargs.pop('mask')
        has_mask = True
    
    # determine overall norms of operands
    norm_meas  = 1
    norm_synth = 1
    if normalize:
        if has_mask:
            norm_meas  = np.sum(meas  * mask)
            norm_synth = np.sum(synth * mask)
        else:
            norm_meas  = np.sum(meas)
            norm_synth = np.sum(synth)
    
    # determine dimensionality
    dim = 0
    if has_mask:
        dim = np.sum(mask)
    else:
        dim = meas.size
    
    # calc squared differences
    rel = np.abs(meas/synth/norm_meas*norm_synth/rn)
    diff_sq = (np.log(rel + 10**(-10)))**2
    if has_mask:
        diff_sq = diff_sq * mask
    
    return np.sqrt(np.sum(diff_sq) * dim)

def make_objective_1D(x, meas):
    """
    Simple objective function maker: Returns objective function for the combination of the 1D grating distribution model with the
    basic 1D far field intensity measurement model and the summed squared difference quality measure
    """
    fct = lambda p :    q_measure(
                            meas_model_basic_far_field_intensity_1D(
                                dist_model_grating_1D(
                                    p[0], p[1], p[2], x
                                )
                            ),
                            meas
                        )
    return fct

def make_objective_2D(xx, yy, meas, **kwargs):
    """
    Simple objective function maker: Returns objective function for the combination of the 2D grating distribution model with the
    basic 2D far field intensity measurement model and the summed squared difference quality measure
    """
    fct = lambda p :    q_measure(
                            meas_model_basic_far_field_intensity_2D(
                                dist_model_grating_2D(
                                    p[0], p[1], p[2], xx, yy
                                )
                            ),
                            meas,
                            **kwargs
                        )
    return fct

def make_objective_2D_with_psf(xx, yy, meas, **kwargs):
    """
    Objective function maker: Returns objective function for the combination of
    the 2D grating distribution model with
    the 2D far field intensity measurement model that takes into account the detector's (fixed) psf and
    the summed square differences quality measure
    """
    fct = lambda p :    q_measure(
                            meas_model_far_field_intensity_and_psf_2D(
                                dist_model_grating_2D(
                                    p[0], p[1], p[2], xx, yy
                                ),
                                **kwargs
                            ),
                            meas,
                            **kwargs
                        )
    return fct

def make_objective_2D_with_psf_and_illumination(xx, yy, meas, **kwargs):
    """
    Objective function maker: Returns objective function for the combination of
    the 2D grating distribution model with
    the 2D far field intensity measurement model that takes into account the detector's (fixed) psf and the illumination function and
    the summed square differences quality measure
    """
    fct = lambda p :    q_measure(
                            meas_model_far_field_intensity_and_psf_and_illumination_2D(
                                dist_model_grating_2D(
                                    p[0], p[1], p[2], xx, yy
                                ),
                                create_illumination(xx, yy, p[3]),
                                **kwargs
                            ),
                            meas,
                            **kwargs
                        )
    return fct

def make_objective_2D_with_psf_and_illumination_and_norm(xx, yy, meas, **kwargs):
    """
    Objective function maker: Returns objective function for the combination of
    the 2D grating distribution model with
    the 2D far field intensity measurement model that takes into account the detector's (fixed) psf and the illumination function and a normalization factor (relative sum norm) and
    the summed square differences quality measure
    """
    fct = lambda p :    q_measure_relative_norm(
                            meas_model_far_field_intensity_and_psf_and_illumination_2D(
                                dist_model_grating_2D(
                                    p[0], p[1], p[2], xx, yy
                                ),
                                create_illumination(xx, yy, p[3]),
                                **kwargs
                            ),
                            meas,
                            p[4],
                            **kwargs
                        )
    return fct

def make_objective_2D_with_psf_and_illumination_and_norm_log(xx, yy, meas, **kwargs):
    """
    Objective function maker: Returns objective function for the combination of
    the 2D grating distribution model with
    the 2D far field intensity measurement model that takes into account the detector's (fixed) psf and the illumination function and a normalization factor (relative sum norm) and
    the summed square differences quality measure
    """
    fct = lambda p :    q_measure_relative_norm_log(
                            meas_model_far_field_intensity_and_psf_and_illumination_2D(
                                dist_model_grating_2D(
                                    p[0], p[1], p[2], xx, yy
                                ),
                                create_illumination(xx, yy, p[3]),
                                **kwargs
                            ),
                            meas,
                            p[4],
                            **kwargs
                        )
    return fct




################################################################################
# Helper functions
################################################################################

def freq_helper(x):
    temp = fft.fftfreq(len(x), x[1]-x[0])
    temp = fft.fftshift(temp)
    return temp

def accepted_minima_only(minima, accepted):
    """
    Create a list that has as each element the last minimum value that was accepted by basinhopping
    
    minima : iterable of all minima basinhopping found
    accepted : iterable of boolean values that indicate if the minimum of the same index was accepted by basinhopping
    """
    temp = list()
    for i in range(len(accepted)):
        if accepted[i] == 1:
            temp.append(minima[i])
        else:
            temp.append(temp[i-1])
    return temp

def get_params(ret):
    """
    Put the reconstructed parameters in a dictionary
    """
    if len(ret.x) is 3:
        return {'sigma':ret.x[0], 'pitch':ret.x[1], 'fsize':ret.x[2]}
    if len(ret.x) is 4:
        return {'sigma':ret.x[0], 'pitch':ret.x[1], 'fsize':ret.x[2], 'sill':ret.x[3]}
    if len(ret.x) is 5:
        return {'sigma':ret.x[0], 'pitch':ret.x[1], 'fsize':ret.x[2], 'sill':ret.x[3], 'rn':ret.x[4]}
    
    
def bisect(data):
    """
    Discriminate image to data to either True or False to get an image mask
    """
    minimum = np.min(data)
    maximum = np.max(data)
    return (data > 0.5*(minimum+maximum))

def domain_scales(x):
    """
    Return as a dictionary the minimal and maximal spatial scale the domain allows to represent
    """
    dx = x[1]-x[0]
    Dx = x[-1]-x[0] + dx
    return {'dx':dx, 'Dx':Dx}

def random_bound_true_paramset(**kwargs):
    """
    Return as a dictionary a reasonable random set of model parameters
    """
    dx = kwargs.pop('dx')
    Dx = kwargs.pop('Dx')
    
    rpitch = (4.*dx, 0.5*Dx)
    pitch  = random.uniform(rpitch[0], rpitch[1])
    
    rfsize = (2.*dx, pitch)
    fsize  = random.uniform(rfsize[0], rfsize[1])
    
    rsigma = (2.*dx, fsize)
    sigma  = random.uniform(rsigma[0], rsigma[1])
    
    return {'sigma':sigma, 'pitch':pitch, 'fsize':fsize}

def random_start_paramset(**kwargs):
    """
    Return as a dictionary a model parameter set that can be used to start a reconstruction
    """
    Dx = kwargs.pop('Dx')
    
    pitch = random.uniform(0, Dx)
    sigma = random.uniform(0, Dx)
    fsize = random.uniform(0, Dx)
    
    return {'sigma':sigma, 'pitch':pitch, 'fsize':fsize}

def random_paramset(pmin, pmax, accept_test, nreadjust=10000, **kwargs):
    scales = pmax - pmin
    def draw():    
        r = np.random.uniform(0., 1., scales.shape)
        return pmin + r*scales
    
    p = draw()
    i = 0
    while not accept_test(x_new=p):
        if i>=nreadjust:
            raise ValueError('scaled_random_paramset failed to produce a valid paramset!')
            break
        p = draw()
        i += 1
    
    if len(p) is 3:
        return {'sigma':p[0], 'pitch':p[1], 'fsize':p[2]}
    if len(p) is 4:
        return {'sigma':p[0], 'pitch':p[1], 'fsize':p[2], 'sill':p[3]}
    if len(p) is 5:
        return {'sigma':p[0], 'pitch':p[1], 'fsize':p[2], 'sill':p[3], 'rn':p[4]}

def draw_paramsets(N, **kwargs):
    """
    Draw a number of random parameter sets and return them as a list
    """
    ensemble = kwargs.pop('ensemble', random_bound_true_paramset)
    
    params = []
    
    for i in range(N):
        params.append(ensemble(**kwargs))
    
    return params

def create_psf(xx, yy, sigma_psf):
    """
    Create circular gaussian point spread function
    
    sigma_psf : sigma of gaussian, in number of pixels
    """
    shape = xx.shape
    small_y = (shape[0])//2
    large_y = shape[0] - small_y
    small_x = (shape[1])//2
    large_x = shape[1] - small_x
    pxidsy = np.arange(-small_y, large_y)
    pxidsx = np.arange(-small_x, large_x)
    pxidsxx, pxidsyy = np.meshgrid(pxidsx, pxidsy)
    if sigma_psf == 0:
        g = np.logical_and(pxidsxx==0, pxidsyy==0)
    else:
        g = np.exp(-0.5*((pxidsxx+1)**2 + (pxidsyy+1)**2)/sigma_psf**2)
    return np.array(g/g.sum())

def binary_ceil(x):
    """
    Return the smallest integer power of 2 that is bigger or equal to x
    """
    return 2**(np.ceil(np.log(x)/np.log(2)))

class ProcessResult(object):
    def __init__(self, pl, consts):
        self.pl = pl
        self.consts = consts
    
    def __call__(self, result):
        processed = {}
        self.pl(result, processed, self.consts)
        return processed

class Mask(object):
    def __init__(self, limit, margin):
        self.limit = limit
        self.margin = margin
    def __call__(self, image):
        minval = image.min()
        maxval = image.max()
        upperlimit = minval + self.limit*(maxval-minval)
        outliers = np.where(image<upperlimit, 0, 1)
        kernel = self._circle(self.margin)
        grown = signal.convolve(outliers, kernel, mode='same')
        mask = np.where(grown == 0, 1, 0)
        return mask
    def _circle(self, r):
        d = 2*r
        a = np.zeros(shape=(d, d))
        for i in range(d):
            for j in range(d):
                if (i-r+0.5)**2+(j-r+.5)**2 <=r**2:
                    a[i, j] = 1
        return a

def crop_to_centered_binary_array(a, xbegin, xend, ybegin, yend, xcent, ycent):
    """
    Select rectangular part containing a special point from a 2D array.
    Copy selected part into a new array of binary shape so that the special point is centerd within the new array.
    
    Return: New array, position mask of original selection within the new array
    """
    # left and right pixel distances from selection edge to the special point
    left_lx     = xcent - xbegin
    right_lx    = xend - xcent
    # bigger of the distances
    big_lx      = np.max([left_lx, right_lx])
    # binary ceil'd
    bin_big_lx  = binary_ceil(big_lx)
    # begin and end of new array in old pixel coordinates, x direction
    new_xbegin  = int(xcent - bin_big_lx)
    new_xend    = int(xcent + bin_big_lx)
    new_lenx    = new_xend - new_xbegin
    
    # down and up pixel distances from selection edge to the special point
    down_ly     = ycent - ybegin
    up_ly       = yend - ycent
    # bigger of the distances
    big_ly      = np.max([down_ly, up_ly])
    # binary ceil'd
    bin_big_ly  = binary_ceil(big_ly)
    # begin and end of new array in old pixel coordinates, y direction
    new_ybegin  = int(ycent - bin_big_ly)
    new_yend    = int(ycent + bin_big_ly)
    new_leny    = new_yend - new_ybegin
    
    # create new array and mask
    b           = np.zeros(shape=(new_lenx, new_leny))
    mask        = np.zeros(shape=(new_lenx, new_leny))
    
    b[xbegin-new_xbegin:xend-new_xbegin, ybegin-new_ybegin:yend-new_ybegin]\
                = a[xbegin:xend, ybegin:yend]
    mask[xbegin-new_xbegin:xend-new_xbegin, ybegin-new_ybegin:yend-new_ybegin]\
                = 1
    
    return b, mask
