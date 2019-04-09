import numpy as np
from scipy import special

def pnfwunorm(q, con=5):
    y = q*con
    return np.log(1.0 + y)-y/(1.0 + y)

def dnfw(x, con=5):
    d = (x*con**2)/(((x*con)+1.0)**2*(1.0/(con+1.0)+np.log(con+1.0)-1.0))
    d[x>1] = 0
    d[x<=0] = 0
    return d

def pnfw(q, con=5, logp=False):
    p = pnfwunorm(q, con=con)/pnfwunorm(1, con=con)
    p[q>1] = 1
    p[q<=0] = 0
    return p

def qnfw(p, con=5, logp=False):
    p[p>1] = 1
    p[p<=0] = 0
    p *= pnfwunorm(1, con=con)
    return (-(1.0/np.real(special.lambertw(-np.exp(-p-1))))-1)/con

def rnfw(con):
    con = np.atleast_1d(con)
    n = int(con.size)
    return qnfw(np.random.rand(n), con=con)
    