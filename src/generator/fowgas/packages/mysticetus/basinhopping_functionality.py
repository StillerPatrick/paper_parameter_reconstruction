import numpy as np



class ParamBounds(object):
    """
    Custom accept_test for basinhopping that enforces bounds on the parameter values
    """
    def __init__(self):
        self.use_bounds = False
    
    def __init__(self, pmax, pmin):
        self.use_bounds = True
        self.pmax = np.array(pmax)
        self.pmin = np.array(pmin)
    
    def __call__(self, **kwargs):
        if self.use_bounds:
            p = kwargs['x_new']
            tmax = bool(np.all(p <= self.pmax))
            tmin = bool(np.all(p >= self.pmin))
            consistent = bool(p[2]<p[1])
            return tmax and tmin and consistent
        else:
            return True

class NonNegativityTest(object):
    """
    Functor that returns true if and only if all parameter values are non-negative.
    """
    def __init__(self):
        pass
    def __call__(self, **kwargs):
        p = kwargs['x_new']
        non_negativ = bool(np.all(p >= 0))
        return non_negativ

class ConsistencyTest(object):
    """
    Functor that returns true if and only if the parameters 'sigma' and 'fsize' are consistent with the 'pitch'.
    TODO: Make independent of explicit meaning of parameter vector!!!
    """
    def __init__(self):
        pass
    def __call__(self, **kwargs):
        p = kwargs['x_new']
        sigma_consistent = bool(p[0] < p[1])
        fsize_consistent = bool(p[2] < p[1])
        return sigma_consistent and fsize_consistent

class BoundsTest(object):
    """
    Functor that is set up with upper and lower bounds for the parameters and returns true if and only if all parameters are within those bounds.
    """
    def __init__(self, pmin, pmax):
        self.pmin = pmin
        self.pmax = pmax
    def __call__(self, **kwargs):
        p = kwargs['x_new']
        min_pass = bool(np.all(p >= self.pmin))
        max_pass = bool(np.all(p <= self.pmax))
        return min_pass and max_pass

class UpperBoundTest(object):
    """
    Functor that is set up with upper bounds for the parameters and returns true if and only no parameter is above its upper bound.
    """
    def __init__(self, pmax):
        self.pmax = pmax
    def __call__(self, **kwargs):
        p = kwargs['x_new']
        max_pass = bool(np.all(p <= self.pmax))
        return max_pass

class HardTest(object):
    """
    Functor that returns true if and only if NonNegativityTest() AND ConsistencyTest() return true.
    """
    def __init__(self):
        self.nonNegativity = NonNegativityTest()
        self.consistency   = ConsistencyTest()
    def __call__(self, **kwargs):
        return self.nonNegativity(**kwargs) and self.consistency(**kwargs)

class SoftTest(object):
    """
    Functor returns true if and only if an UpperBoundsTest() AND a BoundsTest() return true. It is set up with respective arguments.
    """
    def __init__(self, domain_size, pmin, pmax):
        self.domain = UpperBoundTest(domain_size)
        self.bounds = BoundsTest(pmin, pmax)
    def __call__(self, **kwargs):
        return self.domain(**kwargs) and self.bounds(**kwargs)

class CombinedTest(object):
    """
    AND combination of a HardTest() with a SoftTest().
    """
    def __init__(self, domain_size, pmin, pmax):
        self.soft = SoftTest(domain_size, pmin, pmax)
        self.hard = HardTest()
    def __call__(self, **kwargs):
        return self.soft(**kwargs) and self.hard(**kwargs)

class ScaledAxesDisplacement(object):
    """
    Step taking functor that allows for different step size scaling factors in the parameters.
    """
    def __init__(self, scales, stepsize=0.5):
        self.scales   = scales
        self.stepsize = stepsize
    def __call__(self, x):
        displacement = np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
        x += self.scales * displacement
        return x

class TestedDisplacement(object):
    """
    Step taking functor that wraps another step taking functor and additionally enforces the parameters to pass a test.
    """
    def __init__(self, takestep, test, nreadjust=10000):
        self.takestep  = takestep
        self.test      = test
        self.nreadjust = nreadjust
    def __call__(self, x):
        return self.take_step(x)
    def take_step(self, x):
        self.takestep(x)
        i = 0
        while not self.test(**{'x_new':x}):
            if i>=self.nreadjust:
                raise ValueError('take_step failed to produce a valid displacement!')
                break
            self.takestep(x)
            i += 1
        return x

class CallbackSaveAll(object):
    """
    Callback functor for basinhopping routine that saves all minima found, their objective function values and the acceptance information.
    """
    def __init__(self):
        self.x          = []
        self.f          = []
        self.accepted   = []
    
    def __call__(self, x, f, accepted):
        self.x       .append(x)
        self.f       .append(f)
        self.accepted.append(accepted)

    def get_x(self):
        return self.x
    
    def get_f(self):
        return self.f
    
    def get_accepted(self):
        return self.accepted
    
    def clear(self):
        self.x          = []
        self.f          = []
        self.accepted   = []
