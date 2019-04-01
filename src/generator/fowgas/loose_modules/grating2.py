from scipy import special

def make_comb(pitch, reach):
    f = lambda x : 1.*((x%pitch) < reach)
    return f

def make_erf_edge(sigma, mu):
    f = lambda x : 0.5 * (1. + special.erf((x-mu)/sigma))
    return f

def make_periodical(pitch, fct, offset=0.):
    f = lambda x : fct((x-offset)%pitch)
    return f

def make_box(pitch, fsize, fct_edge):
    f = lambda x :\
        (0. <= x)                * (x < 0.5*fsize)         * fct_edge(x) +\
        (0.5*fsize <= x)         * (x < 0.5*(fsize+pitch)) * fct_edge(-x+fsize) +\
        (0.5*(fsize+pitch) <= x) * (x < pitch)             * fct_edge(x-pitch)
    return f

def fct_edge(sigma, x):
    if sigma == 0:
        return 1.*(x>0)
    else:
        return 0.5 * (1+special.erf(x/sigma))

def fct_feat(pitch, fsize, sigma, x):
    return\
    (0. <= x)                * (x < 0.5*fsize)         * fct_edge(sigma, x) +\
    (0.5*fsize <= x)         * (x < 0.5*(fsize+pitch)) * fct_edge(sigma, -x+fsize) +\
    (0.5*(fsize+pitch) <= x) * (x < pitch)             * fct_edge(sigma, x-pitch)

def fct_grat(pitch, fsize, sigma, xx, yy):
    return fct_feat(pitch, fsize, sigma, xx%pitch)