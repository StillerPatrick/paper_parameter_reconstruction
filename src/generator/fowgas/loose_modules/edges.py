import numpy as np
from scipy import special

#@np.vectorize
def perf_edge(x):
    return 0.5*(1+np.sign(x))

def make_perf_edge(b=1., h=1.):
    f = lambda x : h*0.5*(1+np.sign(x))
    return f

#@np.vectorize
def erf95_edge(x):
    y = x*(2.*1.38590382435)
    return 0.5*(1.+special.erf(y)/0.95) * (np.abs(y) < 1.38590382435) +\
           0.5*(1.+np.sign(y)) * (np.abs(y) >= 1.38590382435)

#@np.vectorize
def erf99_edge(x):
    y = x*(2.*1.82138636773)
    return 0.5*(1.+special.erf(y)/0.99) * (np.abs(y) < 1.82138636773) +\
           0.5*(1.+np.sign(y)) * (np.abs(y) >= 1.82138636773)

#@np.vectorize
def erf999_edge(x):
    y = x*(2.*2.3267537655)
    return 0.5*(1.+special.erf(y)/0.999) * (np.abs(y) < 2.3267537655) +\
           0.5*(1.+np.sign(y)) * (np.abs(y) >= 2.3267537655)

#@np.vectorize
def cos_edge(x):
    y = (x+1.5)*np.pi
    return 0.5*(1.+np.cos(y)) * (np.abs(x) < 0.5) +\
           0.5*(1.+np.sign(x)) * ( np.abs(x) >= 0.5)

#@np.vectorize
def circular_edge(x, r):
    if r>0.5:
        raise("Radius must not exceed 0.5")
    
#    return 0.5*(1.+np.sign(x)) if np.abs(x)>=r else \
#            r - np.sqrt(-2*r*x-x**2) if (x<=0) else \
#            1. - r + np.sqrt(2*r*x-x**2)    
    return 0.5*(1.+np.sign(x)) * (np.abs(x)>=r) +\
            ((r - np.sqrt(-2*r*x-x**2)) * (np.abs(x)<r) * (x<=0) +\
             (1. - r + np.sqrt(2*r*x-x**2)) * (x>0))

def make_circular_edge(r):
    if r>0.5:
        raise("Radius must not exceed 0.5")
    
#    return np.vectorize(lambda x : 0.5*(1.+np.sign(x)) if np.abs(x)>=r else \
#                                   r - np.sqrt(-2.*r*x-x**2) if (x<=0) else \
#                                   1. - r + np.sqrt(2.*r*x-x**2))
    return lambda x : 0.5*(1.+np.sign(x)) if np.abs(x)>=r else \
                             r - np.sqrt(-2.*r*x-x**2) if (x<=0) else \
                             1. - r + np.sqrt(2.*r*x-x**2)

#@np.vectorize
def elliptical_edge(x, r, alpha=1.):
    if r*alpha>0.5:
        raise("Radius times alpha must not exceed 0.5")
    
    return 0.5*(1.+np.sign(x)) if np.abs(x)>=r else \
           alpha * (r - np.sqrt(-2*r*x-x**2)) if (x<=0) else \
           1. - alpha * (r - np.sqrt(2*r*x-x**2))

def make_elliptical_edge(r, alpha=1.):
    if r*alpha>0.5:
        raise("Radius times alpha must not exceed 0.5")
    
    f = lambda x : 0.5*(1.+np.sign(x)) if np.abs(x)>=r else \
                   alpha * (r - np.sqrt(-2.*r*x-x**2)) if (x<=0) else \
                   1. - alpha * (r - np.sqrt(2.*r*x-x**2))
    
#    return np.vectorize(f)
    return f

def make_skewed_edge(b=1., h=1.):
    """
    Return an edge function modelled by a linear function.
    
    Arguments:
    w -- Width of edge. For x in (-w/2, w/2), the returned function is linear. Outside it's constant.
    h -- Height of edge.
    
    Return:
    A function object. The function covers the whole real axis and has the asymptotic values 0. and h.
    """
#    f = lambda x : 0.5*h*(np.sign(x)+1.) if np.abs(x)>0.5*b else h/b*(x+0.5*b)
    if b != 0:
        f = lambda x : 0.5*h*(np.sign(x)+1.) * (np.abs(x)>0.5*b) + h/b*(x+0.5*b) * (np.abs(x)<=0.5*b)
    else:
        f = lambda x : 0.5*h*(np.sign(x)+1.)
    
#    return np.vectorize(f)
    return f

def make_erf_edge(w=1., h=1.):
    """
    Return an edge function modelled by the error function of width w and height h.
    
    Arguments:
    w -- Width of edge. Width is defined as the width of the skewed edge that has the same tangent at x=0.
    h -- Height of edge.
    
    Return:
    A function object. The function covers the whole real axis and has the asymptotic values 0. and h.
    """
    a = 2./np.pi
    f = lambda x : h*0.5*((1+special.erf(x/a/w)))
#    return np.vectorize(f)
    return f
