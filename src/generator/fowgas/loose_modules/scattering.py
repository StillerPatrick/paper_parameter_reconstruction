from scipy import fftpack
import numpy as np

def propagate(target, photonnumber, wavelength, target_size, trans, eff, fftsampling=1, re_sq=7.9408e-30):
    """
    Propagate a plane wave through a target with a given distribution of electrons to a detector
    plane in the far field.
    
    Args:
        target = array with number of electrons in each pixel
        photonnumber = total number of incoming photons I_0
        wavelength = wavelength of x-rays in m
        target_size = edgelength of target in m
        trans = transmission through the target from 0. to 1. Absorbed photons will simply 
                    be subtracted from number of incoming photons
        eff = efficiency of detection from 0. to 1.
        
    Return:
        I = detected intensity in number of photons in the detector plane
    """
    I = fftpack.fft2(target)
    I = fftpack.fftshift(I)
    #fft_norm = 1./np.product(target.shape)
    physics_norm = photonnumber * re_sq * wavelength**2 / target_size**4 * trans * eff
    #I = fft_norm * physics_norm * np.abs(I)**2.0
    I = physics_norm * np.abs(I)**2.0

    return I

def propagate_Melanie(target, photonnumber, target_pixsize, dist_det, trans, eff, wavelength, fftsampling=1, re_sq=7.9408e-30):
    """
    Propagate a plane wave through a target with a given distribution of electrons to a detector
    plane in the far field.
    
    Args:
        target = array with number of electrons in each pixel
        photonnumber = total number of incoming photons I_0
        target_pixsize = pixel size of the target array in m
        dist_det = distance target - detector plane
        trans = transmission through the target (absorbed photons will simply be subtracted from number of incoming photons)
        eff = efficiency of detection
        wavelength = wavelength of x-rays in m
        re_sq = square of classical electron radius in m^2
    
    Return:
        I = detected intensity in number of photons in the detector plane
    """
    # calculate the minimal scattering theta_min
    # theta_min = asin(lambda/(2*biggest feature size))
    # (biggest feature size =
    #   size of the simulated target =
    #   number of pixels along one axis in the target array * pixelsize
    #   -> corresponds to smallest q = one pixel in diffraction pattern)
    theta_min = np.arcsin(wavelength / (2. * len(target) * target_pixsize))
    
    # calculate pixel size of simulated diffraction pattern on the detector
    det_pixsize = dist_det * np.tan(2.0 * theta_min)
    
    # propagate to the Fraunhofer regime by simple FFT
    F1 = fftpack.fft2(target, shape=(target.shape * np.ones(2) * fftsampling))
    # shift intensity from the corners to the center (standard FFT procedure)
    F2 = fftpack.fftshift(F1) / len(target)
    
    # calculate prefactor of FFT in correct units 
    norm = re_sq * photonnumber / target_pixsize**2.0 * det_pixsize**2.0 / dist_det**2.0 * trans * eff
    
    # calculate intensity in number of photons
    #   physics:  I = |amplitude of wavefield|^2
    #   numerics: I = norm * |FFT|^2
    I = norm * np.abs(F2)**2.0

    #print "Sum over target array:", target.sum()
    #print "Pixel size of simulated diffraction pattern on the detector: %.2e" % det_pixsize, " m"
    #print "Normalization factor: ", norm
    #print "Maximum of the Fourier transform 2: ", '%.2e' %np.amax(abs(F2))
    #print "Maximum of the Fourier transform 1: ", '%.2e' %np.amax(abs(F1))
    
    return I

def maskcenter(a, n):
    nx, ny = a.shape
    mask = np.ones((nx, ny))
    mask[(nx-n)/2:(nx+n)/2, (ny-n)/2:(ny+n)/2] = 0.
    return mask

