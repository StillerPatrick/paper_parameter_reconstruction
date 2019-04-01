import numpy as np



class Abstract1d(object):
    def __call__(self, x):
        return self.function(x)



class Abstract1dEdge(Abstract1d):
    """
    Abstract base class for edge models. Functor.
    """
    def __init__(self, slope_at_zero):
        """
        Initializer. Abstract, needs to be reimplemented by child classes.
        
        slope_at_zero: function slope at x=1 that this functor object will have.
        """
        self.function = None
        raise NotImplementedError
        
    def __call__(self, x):
        """
        Return density at x.
        
        x: spatial postion at which the 1D density function is evaluated.
        """
        return self.function(x)



class AbstractFinite1d(Abstract1d):
    """
    Defined for a continuous finite interval only.
    """
    def __call__(self, x):
        self.range_check(x)
        return self.function(x)
    
    def range_check(self, x):
        if (x<self.lower_lim).any():
            raise ValueError
        if (self.upper_lim<=x).any():
            raise ValueError



class Symmetric1dFromEdge(Abstract1d):
    def __init__(self, edge, offset):
        if not issubclass(type(edge), Abstract1dEdge):
            raise TypeError
        self.edge   = edge
        self.offset = offset
    
    def function(self, x):
        return self.edge(x)              * np.array(self._lower_part(x), dtype=float)+\
               self.edge(-x+self.offset) * np.array(self._upper_part(x), dtype=float)
    
    def _lower_part(self, x):
        return x<0.5*self.offset
    
    def _upper_part(self, x):
        return 0.5*self.offset<=x



class Finite1dSection(AbstractFinite1d):
    def __init__(self, f1d, lower, upper):
        if lower >= upper:
            raise ValueError
        self.lower_lim  = lower
        self.upper_lim  = upper
        self.function   = f1d



class Periodic1dFromFinite(Abstract1d):
    def __init__(self, finite_feature):
        if not issubclass(type(finite_feature), AbstractFinite1d):
            raise TypeError
        self.finite_feature = finite_feature
        self.period_length = finite_feature.upper_lim - finite_feature.lower_lim
        self.function = lambda x : self.finite_feature(self.periodic_mapping(x))
    
    def periodic_mapping(self, x):
        return (x-self.finite_feature.lower_lim)%self.period_length + self.finite_feature.lower_lim
