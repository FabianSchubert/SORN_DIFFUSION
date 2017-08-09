import numpy as np
from scipy.special import kn

def psi_old(x,D,lambd):
    
    with np.errstate(divide='ignore'):
        return np.nan_to_num(np.divide(kn(0,x*np.sqrt(np.divide(lambd,D))),(2.*np.pi*D))) + np.nan_to_num((x==0.)*np.inf)


def psi_0(D,lambd,h):
    
    if D > 0.:
        return (1.-h*np.sqrt(lambd/(np.pi*D))*kn(1,h*np.sqrt(lambd/(np.pi*D)))) / (h**2 * lambd)
    else:
        return 1./(h**2 * lambd)

def psi(x,D,lambd,h,expon):
    
    return 1./(psi_0(D,lambd,h)**-expon + psi_old(x,D,lambd)**-expon)**(1./expon)
