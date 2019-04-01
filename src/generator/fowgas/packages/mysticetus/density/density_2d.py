from . import density_1d




class Abstract2d(object):
    def __call__(self, x, y):
        return self.function(x, y)



class Wrap1dTo2d(Abstract2d):
    def __init__(self, wrap_1d, **kwargs):
        if not issubclass(type(wrap_1d), density_1d.Abstract1d):
            raise TypeError
        
        self.mapping_xy2x = kwargs.pop('mapping_xy2x', (lambda x,y : x))
        self.wrapped_1d = wrap_1d
    
    def function(self, x, y):
        t = self.mapping_xy2x(x, y)
        return self.wrapped_1d(t)



