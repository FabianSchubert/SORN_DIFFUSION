import numpy as np
from scipy.special import erfcx

def phi_noise(I,V_t,V_r,E_l,tau_m,sigm,nint):
	
	range_grid = np.meshgrid(np.linspace(0,1,nint),xrange(len(I)))[0]
	
	dint = (V_t-V_r)/(sigm*nint)
	#int_mesh = np.ndarray((480,nint))
	diff_grid = np.meshgrid(xrange(nint),(V_t-V_r)/sigm)[1]
	low_value_grid = np.meshgrid(xrange(nint),(V_r-E_l-I*tau_m)/sigm)[1]
	int_mesh = erfcx(-(low_value_grid + range_grid*diff_grid))
	return 1./(np.sqrt(np.pi)*tau_m*dint*int_mesh.sum(axis=1))
