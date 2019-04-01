from mysticetus import Models

from scipy import fftpack as fft
import numpy as np
from scipy import ndimage


DistModelGrating2D = Models.DistModelGrating2D
ConvolveIntensityWithConstantPSF = Models.ConvolveIntensityWithConstantPSF


class MeasModelFarField(Models.BaseModel):
    """
    Calculate the far field by FFT of electron distribution. sum(abs(.)) of far field is the same as of el. dist.
    """
    def calc(self, params, interims, consts):
        dist = interims['dist']
        ff   = fft.fftshift(fft.fft2(dist))
        n    = dist.shape[0]*dist.shape[1]
        interims['far_field'] = ff/np.sqrt(n)



class RotateFF(Models.BaseModel):
    """
    Rotate far field. Angle in radians.
    """
    def calc(self, params, interims, consts):
        ff = interims['far_field']
        angle = params['ff_angle']
        real = np.real(ff)
        imag = np.imag(ff)
        rot_real = ndimage.rotate(real, angle*180/np.pi, order=1, reshape=False, cval=np.nan)
        rot_imag = ndimage.rotate(imag, angle*180/np.pi, order=1, reshape=False, cval=np.nan)
        interims['far_field'] = rot_real + 1j*rot_imag



class DisplaceMeas(Models.BaseModel):
    """
    Displace the far field.
    """
    def calc(self, params, interims, consts):
        ff = interims['far_field']
        disp = params['disp']
        dispd_real = ndimage.shift(ff.real, disp, order=1, cval=np.nan)
        dispd_imag = ndimage.shift(ff.imag, disp, order=1, cval=np.nan)
        interims['far_field'] = dispd_real + 1j*dispd_imag



class CropFarfield(Models.BaseModel):
    """
    Crop far field.
    """
    def calc(self, params, interims, consts):
        cropslice = consts['crop_slice']
        interims['far_field'] = interims['far_field'][cropslice[0], cropslice[1]]



class DrawExperimentalPhotonCount(Models.BaseModel):
    """
    interims['far_field'] needs to be normalized to count photon number expectation values before applying this Functor!
    """
    def calc(self, params, interims, consts):
        photon_statistics = consts['photon_statistics']
        interims['far_field'] = photon_statistics(interims['far_field'])



class Farfield2Intensity(Models.BaseModel):
    """
    Compute the intensity of the far field.
    """
    def calc(self, params, interims, consts):
        interims['meas'] = np.abs(interims['far_field'])**2



class CutoffIntensity(Models.BaseModel):
    """
    Set intensity values that are above the cutoff value to the cutoff value.
    """
    def calc(self, params, interims, consts):
        cutoff = consts['cutoff']
        selection = np.where(interims['meas']>cutoff, True, False)
        interims['meas'][selection] = cutoff
