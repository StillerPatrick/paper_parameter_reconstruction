import numpy

def func2areamap(x, y, f):
    a = numpy.zeros((x.size, y.size))
    for i in range(x.size):
        for j in range(y.size):
            if f[i] >= y[j]:
                a[i, j] = 1
            else:
                a[i, j] = 0
    return a

import edges
from auxiliary import theta

def func2areamap2(x, y, f):
    xx, yy = numpy.meshgrid(x, y)
    fvals = f(xx)
    e1 = theta
    e2 = edges.make_skewed_edge(numpy.mean(numpy.diff(x)), 1)
    a = e1(fvals-yy)[:, 0:-1] - e1(fvals-yy)[:, 1:yy.shape[1]]
    return a.T
    #return (e2(fvals-yy)).T