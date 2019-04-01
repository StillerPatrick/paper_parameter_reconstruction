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
    Models.NormalizeToMeasurement(),
    Models.MultMeasWithRelNorm(),
])

# Pipeline: From params to quality measure
pl = Objective.Pipeline([\
    dm_pl,
    mm_pl,
    Models.AverageSquaredDifferences()
])

def make_objective(pl, consts, **kwargs):
    return Objective.Objective(pl, consts, **kwargs)

def p2params(p):
    return {'sigma':p[0], 'pitch':p[1], 'fsize':p[2], 'sill':p[3], 'rn':p[4]}

def params2p(params):
    return (params['sigma'], params['pitch'], params['fsize'], params['sill'], params['rn'])

class ResultProcessingPipeline(object):
    def __call__(self, result, interims, consts):
        opt_params              = p2params(result.bh_return.x)
        interims['opt_params']  = opt_params
        Models.DistModelGrating2D()(              opt_params, interims, consts)
        interims['opt_dist']    = interims['dist']
        Models.CreateIllumination()(              opt_params, interims, consts)
        Models.MultDistWithInterimIllumination()( opt_params, interims, consts)
        Models.MeasModelFarFieldIntensity2D()(    opt_params, interims, consts)
        Models.ConvolveIntensityWithConstantPSF()(opt_params, interims, consts)
        interims['opt_meas']    = interims['meas']
        
        interims['acc_minima']  = model_driven_reco.accepted_minima_only(result.fs, result.accs)
        interims['true_dist']   = result.m.dist   if hasattr(result.m, 'dist')   else None
        interims['meas']        = result.m.meas
        interims['mask']        = result.m.mask   if hasattr(result.m, 'mask')   else None
