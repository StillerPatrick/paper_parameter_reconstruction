import copy



class DeepCopyAttributeGetter:
    """
    Functor for getting an attribute from a deepcopy of an object.
    """
    def __init__(self, object_, attribute):
        """
        Initialize functor.
        
        Parameters
        -----
        object_: The object whose attribute this instance gets.
        attribute: The attribute this instance gets.
        """
        self.object_    = object_
        self.attribute  = attribute
    
    def __call__(self):
        """
        Return object's attribute.
        
        Deepcopy object before getting the attribute to make sure the return value is 'decoupled' from the original object.
        """
        return getattr(copy.deepcopy(self.object_), self.attribute)



class DeepCopyKeyGetter:
    """
    Functor for getting a mapped value from a deepcopy of a map-like object.
    """
    def __init__(self, object_, key):
        """
        Initialize functor.
        
        Parameters
        -----
        object_: The object whose mapped value this instance gets.
        key: The key this instance looks up.
        """
        self.object_    = object_
        self.key        = key
    
    def __call__(self):
        """
        Return object's mapped value.
        
        Deepcopy object before looking up the key to make sure the return value is 'decoupled' from the original object.
        """
        return copy.deepcopy(self.object_[self.key])



class AttributeSetter:
    """
    Functor fo setting an objects attribute.
    """
    def __init__(self, object_, attribute):
        """
        Initialize functor.
        
        Parameters
        -----
        object_: The object whose attribute this instance sets.
        attribute: The attribute this instance sets.
        """
        self.object_    = object_
        self.attribute  = attribute
    
    def __call__(self, value):
        """
        Set object's attribute to desired value.
        
        Parameters
        -----
        value: The value that will be set.
        """
        setattr(self.object_, self.attribute, value)



class KeySetter:
    """
    Functor for setting one key-value-pair of a map-like object.
    """
    def __init__(self, object_, key):
        """
        Initialize functor.
        
        Parameters
        -----
        object_: The object whose mapping this instance sets.
        key: The key this instance sets the mapped value for.
        """
        self.object_    = object_
        self.key        = key
    
    def __call__(self, value):
        """
        Map object's key to desired value.
        
        Parameters
        -----
        value: The value that will be set.
        """
        self.object_[self.key] = value



class GetSet:
    """
    Functor for getting a value and then setting it elsewhere.
    """
    def __init__(self, getter, setter):
        """
        Initialize functor.
        
        Parameters
        ----
        getter: Use this getter.
        setter: Use this setter.
        """
        self.getter = getter
        self.setter = setter
    
    def __call__(self):
        """
        Get and set.
        """
        temp = self.getter()
        self.setter(temp)



class GetSetChain:
    """
    Functor that performs multple get-and-set actions.
    """
    def __init__(self, chain):
        """
        Initialize functor.
        
        Parameters
        -----
        chain: Iterable of getsetters to use.
        """
        self.chain = chain
    
    def __call__(self):
        """
        Get and set.
        """
        for gs in self.chain:
            gs()



if __name__ == '__main__':
    import copy

    class CalculationStep:
        def __init__(self, **kwargs):
            save = kwargs.pop('save', None)
            self.save = save

        def __call__(self, params, interims, consts):
            self.calc(params, interims, consts)
            if self.save is not None:
                self.save()

        def calc(self, params, interims, consts):
            pass

    class StepCopyParamsIntoInterims(CalculationStep):
        def calc(self, params, interims, consts):
            for k in params.keys():
                interims[k] = params[k]

    class StepDoubleAllInterims(CalculationStep):
        def calc(self, params, interims, consts):
            for k in interims.keys():
                interims[k] = 2.*interims[k]

    class Pipeline(CalculationStep):
        def __init__(self, steps):
            self.steps = steps

        def __call__(self, params, interims, consts):
            for s in self.steps:
                s(params, interims, consts)

    class Dummy:
        pass

    params = {'a':1., 'b':2.}
    interims = {}
    consts = {}

    jotDown = Dummy()

    pl = Pipeline([ StepCopyParamsIntoInterims(
                        save=GetSetChain([
                            GetSet(DeepCopyKeyGetter(interims, 'a'), AttributeSetter(jotDown, 'a1')),
                            GetSet(DeepCopyKeyGetter(interims, 'b'), AttributeSetter(jotDown, 'b1'))
                        ])
                    ),
                    StepDoubleAllInterims(
                        save=GetSetChain([
                            GetSet(DeepCopyKeyGetter(interims, 'a'), AttributeSetter(jotDown, 'a2')),
                            GetSet(DeepCopyKeyGetter(interims, 'b'), AttributeSetter(jotDown, 'b2'))
                        ])
                    ),
                    StepDoubleAllInterims(
                        save=GetSetChain([
                            GetSet(DeepCopyKeyGetter(interims, 'a'), AttributeSetter(jotDown, 'a3')),
                            GetSet(DeepCopyKeyGetter(interims, 'b'), AttributeSetter(jotDown, 'b3'))
                        ])
                    ),
                    StepCopyParamsIntoInterims(
                        save=GetSetChain([
                            GetSet(DeepCopyKeyGetter(interims, 'a'), AttributeSetter(jotDown, 'a4')),
                            GetSet(DeepCopyKeyGetter(interims, 'b'), AttributeSetter(jotDown, 'b4'))
                        ])
                    ),
                  ])

    pl(params, interims, consts)

    print(interims)
    print(vars(jotDown))