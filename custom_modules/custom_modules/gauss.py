import numpy as np

def gauss1d(x,mu,sigm):
    return np.exp(-(x-mu)**2/(2.*sigm**2))/(np.sqrt(2.*np.pi)*sigm)


def gauss2d(x,mu,sigm):
    return np.exp(-(x-mu)**2/(2.*sigm**2))/(2.*np.pi*sigm**2)