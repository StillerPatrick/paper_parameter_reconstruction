from mysticetus import Objective
from mysticetus import InspectObjective as IO
import Models



class SavingObject(object):
    pass    



interims = {}
savings = SavingObject()

pl = Objective.Pipeline([\
    Models.DistModelGrating2D(
        save=IO.GetSetChain([\
            IO.GetSet(
                IO.DeepCopyKeyGetter(interims, 'dist'),
                IO.AttributeSetter(savings, 'dist')
            )
        ])
    ),
    Models.MeasModelFarField(
        save=IO.GetSetChain([\
            IO.GetSet(
                IO.DeepCopyKeyGetter(interims, 'far_field'),
                IO.AttributeSetter(savings, 'untouched_far_field')
            )
        ])
    ),
    Models.RotateFF(),
    Models.DisplaceMeas(),
    Models.CropFarfield(
        save=IO.GetSetChain([\
            IO.GetSet(
                IO.DeepCopyKeyGetter(interims, 'far_field'),
                IO.AttributeSetter(savings, 'tailored_far_field')
            )
        ])
    ),
    Models.DrawExperimentalPhotonCount(),
    Models.Farfield2Intensity(),
    Models.ConvolveIntensityWithConstantPSF(),
    Models.CutoffIntensity(
        save=IO.GetSetChain([\
            IO.GetSet(
                IO.DeepCopyKeyGetter(interims, 'meas'),
                IO.AttributeSetter(savings, 'endproduct')
            )
        ])
    )
])


pl2 = Objective.Pipeline([\
    Models.DistModelGrating2D(
        save=IO.GetSetChain([\
            IO.GetSet(
                IO.DeepCopyKeyGetter(interims, 'dist'),
                IO.AttributeSetter(savings, 'dist')
            )
        ])
    ),
    Models.MeasModelFarField(),
    Models.CropFarfield(),
    Models.DrawExperimentalPhotonCount(),
    Models.Farfield2Intensity(
        save=IO.GetSetChain([\
            IO.GetSet(
                IO.DeepCopyKeyGetter(interims, 'meas'),
                IO.AttributeSetter(savings, 'endproduct')
            )
        ])
    )
])



