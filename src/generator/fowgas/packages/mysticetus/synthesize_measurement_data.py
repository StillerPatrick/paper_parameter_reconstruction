import numpy as np
import copy
from . import Measurement

class SyntheticMeasurement(Measurement.Measurement):
    """
    """
    def __init__(self, params, domain, dist, meas):
        self.params     = params
        self.domain     = domain
        self.dist       = dist
        self.meas       = meas

class Synthesizer(object):
    """
    """
    def __init__(self, dist_model, meas_model, domain):
        self.dist_model = dist_model
        self.meas_model = meas_model
        self.domain     = domain
    
    def __call__(self, params, **kwargs):
        dist            = self.dist_model(params, self.domain)
        meas            = self.meas_model(params, dist, **kwargs)
        return SyntheticMeasurement(params, self.domain, dist, meas)
