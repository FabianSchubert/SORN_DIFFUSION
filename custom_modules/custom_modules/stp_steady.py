import numpy as np

def stp_steady(f,tau_d,tau_f,U):
	
	u = U/(1.-(1.-U)*np.exp(-1./(f*tau_f)))
	r = (1.-np.exp(-1./(f*tau_d)))/(1.-(1.-u)*np.exp(-1./(f*tau_d)))
		
	return u*r
