# Pipeline functor to wrap several calculation stages together
class Pipeline(object):
    def __init__(self, steps):
        self.steps = steps
    
    def __call__(self, params, interims, consts):
        for step in self.steps:
            step(params, interims, consts)

# Objective function functor
class Objective(object):
    def __init__(self, pl, consts, **kwargs):
        self.p_wrap = kwargs['p_wrap'] if 'p_wrap' in kwargs.keys() else None
        self.pl = pl
        self.consts = consts
    
    def __call__(self, p):
        interims = {}
        params = self.p_wrap(p)
        self.pl(params, interims, self.consts)
        return interims['q']
