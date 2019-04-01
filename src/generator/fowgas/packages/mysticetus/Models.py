import numpy as np
from mysticetus.density import density_1d
from mysticetus.density import edge_functions_1d
from mysticetus.density import density_2d



class BaseModel(object):
    def __init__(self, **kwargs):
        save = kwargs.pop('save', None)
        self.save = save
    
    def __call__(self, params, interims, consts):
        self.calc(params, interims, consts)
        if self.save is not None:
            self.save()
    
    def calc(self, params, interims, consts):
        pass



class DistModelGrating2D(BaseModel):
    def calc(self, params, interims, consts):
        edge = edge_functions_1d.NormalEdge(params['sigma'], mode='sigma')
        symm = density_1d.Symmetric1dFromEdge(edge, params['fsize'])
        feat = density_1d.Finite1dSection(symm, -0.5*(params['pitch']-params['fsize']), 0.5*(params['pitch']+params['fsize']))
        grat = density_1d.Periodic1dFromFinite(feat)
        grat2d = density_2d.Wrap1dTo2d(grat)
        dist = grat2d(consts['domain']['xx'], consts['domain']['yy'])
        interims['dist'] = dist



class CreateIllumination(BaseModel):
    def calc(self, params, interims, consts):
        xx = consts['domain']['xx']
        yy = consts['domain']['yy']
        ill = self._create_illumination(xx, yy, params['sill'])
        interims['illumination'] = ill
    
    def _create_illumination(self, xx, yy, sigma_ill):
        """
        Create circular square root of gaussian illumination function

        sigma_ill : sigma of gaussian, in number of pixels
        """
        shape = xx.shape
        small_y = (shape[0])//2
        large_y = shape[0] - small_y
        small_x = (shape[1])//2
        large_x = shape[1] - small_x
        pxidsy = np.arange(-small_y, large_y)
        pxidsx = np.arange(-small_x, large_x)
        pxidsxx, pxidsyy = np.meshgrid(pxidsx, pxidsy)
        g = np.exp(-0.5*((pxidsxx+1)**2 + (pxidsyy+1)**2)/sigma_ill**2)
        g = np.array(g/g.sum())
        g = np.sqrt(g)
        return g



class MultDistWithInterimIllumination(BaseModel):
    def calc(self, params, interims, consts):
        illumination = interims['illumination']
        dist = interims['dist']
        dist = dist * illumination
        interims['dist'] = dist



import scipy.fftpack as fft

class MeasModelFarField2D(BaseModel):
    def calc(self, params, interims, consts):
        temp = fft.fft2(interims['dist'])
        temp = fft.fftshift(temp)
        interims['field_det'] = temp



class MeasModelFarFieldIntensity2D(BaseModel):
    """
    Measurement model: input (2D) -> 2D FFT -> absolute squared -> out
    """
    def calc(self, params, interims, consts):
        temp = fft.fft2(interims['dist'])
        temp = fft.fftshift(temp)
        meas = np.abs(temp)**2
        interims['meas'] = meas



from scipy import signal

def _convolve_with_psf(sharp_intensity, convolution_kernel):
    """
    Convolve the sharp intensity signal with the detector's PSF
    """
    out = signal.fftconvolve(sharp_intensity, convolution_kernel, mode='same')
    return out    



class ConvolveIntensityWithConstantPSF(BaseModel):
    def calc(self, params, interims, consts):
        psf = consts['psf']
        meas = interims['meas']
        meas = _convolve_with_psf(meas, psf)
        interims['meas'] = meas



class ConvolveIntensityWithInterimPSF(BaseModel):
    def calc(self, params, interims, psf):
        psf = interims['psf']
        meas = interims['meas']
        meas = _convolve_with_psf(meas, psf)
        interims['meas'] = meas



class NormalizeToMeasurement(BaseModel):
    def calc(self, params, interims, consts):
        synth = interims['meas']
        meas = consts['meas']
        mask = consts['mask'] if 'mask' in consts.keys() else None
        
        use_synth = (synth*mask) if mask is not None else synth
        use_meas  = (meas *mask) if mask is not None else meas
        
        meas = synth/np.sum(use_synth)*np.sum(use_meas)
        interims['meas'] = meas



class MultMeasWithRelNorm(BaseModel):
    def calc(self, params, interims, consts):
        meas = interims['meas']
        meas = meas * params['rn']
        interims['meas'] = meas



class AverageSquaredDifferences(BaseModel):
    def calc(self, params, interims, consts):
        mask = consts['mask'] if 'mask' in consts.keys() else None
        synth = interims['meas']
        meas = consts['meas']
        use_synth = synth*mask if mask is not None else synth
        use_meas  = meas *mask if mask is not None else meas
        dim = np.sum(mask)     if mask is not None else meas.size
        diff_sq = (use_meas - use_synth)**2
        q = np.sqrt(np.sum(diff_sq)*dim)
        interims['q'] = q



class MaximumSquaredDifferences(BaseModel):
    def calc(self, params, interims, consts):
        mask = consts['mask'] if 'mask' in consts.keys() else None
        synth = interims['meas']
        meas = consts['meas']
        use_synth = synth*mask if mask is not None else synth
        use_meas  = meas *mask if mask is not None else meas
        diff_sq = (use_meas - use_synth)**2
        q = np.sqrt(np.max(diff_sq))
        interims['q'] = q
