from mysticetus import model_driven_reco
from mysticetus import Models
from mysticetus import Objective

import copy


# Pipeline: From params to distribution
dm_pl   = Objective.Pipeline([\
            Models.DistModelGrating2D(),])

# Pipeline: From distribution to synthetic measurement
mm_pl   = Objective.Pipeline([\
            Models.CreateIllumination(),
            Models.MultDistWithInterimIllumination(),
            Models.MeasModelFarFieldIntensity2D(),
            Models.ConvolveIntensityWithConstantPSF(),
            Models.NormalizeToMeasurement()])

# Pipeline: From params to quality measure
pl      = Objective.Pipeline([\
            dm_pl,
            mm_pl,
            Models.AverageSquaredDifferences()])

def make_objective(pl, consts, **kwargs):
    return Objective.Objective(pl, consts, **kwargs)

# Parameter representation conversions between tuple and dict
def p2params(p):
    return {'sigma':p[0], 'pitch':p[1], 'fsize':p[2], 'sill':p[3]}

def params2p(params):
    return (params['sigma'], params['pitch'], params['fsize'], params['sill'])

# Wrapper for producing the synthetic distribution
class SynthDist(object):
    def __init__(self, pl):
        self.pl = pl
    
    def __call__(self, params, consts):
        interims = {}
        self.pl(params, interims, consts)
        return interims['dist']

# Wrapper for producing the synthetic measurement
class SynthMeas(object):
    def __init__(self, pl):
        self.pl = pl
    
    def __call__(self, params, dist, consts):
        interims = {'dist':copy.deepcopy(dist)}
        self.pl(params, interims, consts)
        return interims['meas']

calc_sd = SynthDist(dm_pl)
calc_sm = SynthMeas(mm_pl)



class ResultProcessingPipeline(object):
    def __call__(self, result, interims, consts):
        opt_params              = p2params(result.bh_return.x)
        interims['opt_dist']    = calc_sd(opt_params, consts)
        interims['opt_meas']    = calc_sm(opt_params, interims['opt_dist'], consts)
        interims['acc_minima']  = model_driven_reco.accepted_minima_only(result.fs, result.accs)
        interims['true_dist']   = result.m.dist   if hasattr(result.m, 'dist')   else None
        interims['meas']        = result.m.meas
        interims['mask']        = result.m.mask   if hasattr(result.m, 'mask')   else None