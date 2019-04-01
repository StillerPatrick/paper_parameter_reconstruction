import numpy as np
from scipy.special import erf
from . import density_1d



class SineEdge(density_1d.Abstract1dEdge):
    def __init__(self, slope_at_zero):
        self.slope_at_zero = slope_at_zero
        self.left = -np.pi/2/(2*slope_at_zero)
        self.right = np.pi/2/(2*slope_at_zero)
        
    def function(self, x):
        return self._f(x) * np.array(self._within(x), dtype=float) +\
                            np.array(self._above(x),  dtype=float)
    
    def _f(self, x):
        return 0.5*(1+np.sin(2.*self.slope_at_zero*x))
    
    def _below(self, x):
        return x<self.left
    
    def _within(self, x):
        return (self.left<=x) & (x<self.right)
    
    def _above(self, x):
        return self.right<=x



class StraightEdge(density_1d.Abstract1dEdge):
    def __init__(self, slope_at_zero):
        self.slope_at_zero  = slope_at_zero
        self.left           = -1./(2.*slope_at_zero)
        self.right          =  1./(2.*slope_at_zero)
    
    def function(self, x):
        return self._f(x) * np.array(self._within(x), dtype=float) +\
                            np.array(self._above(x),  dtype=float)
    
    def _f(self, x):
        return 0.5*(1+2*self.slope_at_zero*x)
    
    def _below(self, x):
        return self._f(x)<0
    
    def _within(self, x):
        return (0<self._f(x))&(self._f(x)<1)
    
    def _above(self, x):
        return 1<self._f(x)



class NormalEdge(density_1d.Abstract1dEdge):
    def __init__(self, param, **kwargs):
        mode                = kwargs.pop('mode', 'slope')
        if mode == 'slope':
            self.slope_at_zero = param
        if mode == 'sigma':
            self.slope_at_zero = 1./(param * np.sqrt(2.*np.pi))
        
        self.left           = -np.infty
        self.right          = np.infty
    
    def function(self, x):
        return 0.5*(1+erf(x/(self.sigma()*np.sqrt(2))))
    
    def sigma(self):
        return 1./(self.slope_at_zero * np.sqrt(2.*np.pi))
