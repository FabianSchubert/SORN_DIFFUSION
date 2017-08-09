from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from custom_modules.plot_setting import *

from custom_modules.stp_steady import *
from custom_modules.lif_act_funct import *

from tqdm import tqdm

# This script makes a prediction about the steady state population mean firing rate
# activities of the network based on the standard settings for the connection matrices
# and the parameters of the LIF-neuron. The simulation implements a theoretical approach
# of describing the activity in an I-E recurrent network, which can be found in 
# "On the Distribution of Firing Rates in Networks of Cortical Neurons" (Roxin et al., 2011).

## Simulation scripts required to run before this script:
# None


# where to save figure
savefolder = plots_base_folder
plot_filename = "approx_rates_EI"

# time step for integration
dt = 0.001

# membrane time constant , seconds	
tau = 0.02	

# inhibitory threshold, mV
theta_i = -58.

tau_th = 2.5 # time constant of threshold adaptation, seconds

Vr_e = -70. # excitatory reset value, mV
Vr_i = -60. # inhibitory reset value, mV
El = -60. # resting value, mV


## STP parameters
tau_d= 0.5 #short-term depression time constant, seconds
tau_f=2.0 #short-term facilitation time constant, seconds 
U_stp=.04 #Maximum conuctance

### Synaptic normalization parameters
total_in_eTOe = 40. #total e->e synaptic input, mV
total_in_iTOe = 12. #total i->e synaptic input, mV 
total_in_eTOi = 60. #total e->i synaptic input, mV
total_in_iTOi = 60. # total i->i synaptic input, mV

# connection fractions
cf_eTOe = 0.1
cf_iTOe = 0.1
cf_eTOi = 0.1
cf_iTOi = 0.5

# number of neurons
N_e = 400
N_i = 80

# Mean number of incoming connections
N_in_eTOe = N_e*cf_eTOe
N_in_iTOe = N_i*cf_iTOe
N_in_eTOi = N_e*cf_eTOi
N_in_iTOi = N_i*cf_iTOi

# Mean weight per incoming connection
J_eTOe = total_in_eTOe/N_in_eTOe
J_iTOe = -total_in_iTOe/N_in_iTOe
J_eTOi = total_in_eTOi/N_in_eTOi
J_iTOi = -total_in_iTOi/N_in_iTOi

sigm_sqrt_mem_noise = 5. # variance of intrinsic membrane noise, mV^2

T=1. # simulation runtime

# Runge-Kutta integration step
def step(phi,func,dt):
	phi1=func(phi)
	phi2=func(phi+phi1*dt/2)
	phi3=func(phi+phi2*dt/2)
	phi4=func(phi+phi3*dt)
	return phi+(phi1+2*phi2+2*phi3+phi4)*dt/6
# function to be integrated
def F(phi):
	
	# standard deviation of excitatory input fluctuations
	sigma_sqrt_J_e = phi[0]*N_in_eTOe*(J_eTOe*stp_steady(phi[0],tau_d,tau_f,U_stp))**2 + phi[1]*N_in_iTOe*J_iTOe**2
	
	# total excitatory membrane fluctuations, including intrinsic noise
	sigma_e = np.sqrt(sigm_sqrt_mem_noise+tau*sigma_sqrt_J_e)
	
	# the same variables for inhibitory neurons
	
	sigma_sqrt_J_i = phi[0]*N_in_eTOi*J_eTOi**2 + phi[1]*N_in_iTOi*J_iTOi**2
		
	sigma_i = np.sqrt(sigm_sqrt_mem_noise+tau*sigma_sqrt_J_i)
	
	# dynamic state variable
	r = np.zeros(4)
	
	# index order:
	# 0: exc. pop. firing rate
	# 1: inh. pop. firing rate
	# 2: exc. pop. threshold
	# 3: inh. pop. threshold

	r[0] = (1./tau)*(-phi[0]+phi_noise(np.array([phi[0]*total_in_eTOe*stp_steady(phi[0],tau_d,tau_f,U_stp) - total_in_iTOe*phi[1]]),phi[2],Vr_e,El,tau,sigma_e,3000)[0])
	r[1] = (1./tau)*(-phi[1]+phi_noise(np.array([phi[0]*total_in_eTOi - total_in_iTOi*phi[1]]),phi[3],Vr_i,El,tau,sigma_i,3000)[0])
	r[2] = 0.
	r[3] = 0.
	
	return r


def fixed_th(theta_e,theta_i,file_loc):
	
	# initialize list for recording firing rates
	rec = []
	
	# initialize dynamic vector
	phi = np.zeros(4)
	
	# initialize rates and thresholds (although thresholds do not change)
	phi[0] = 1.
	phi[1] = 1.
	phi[2] = theta_e
	phi[3] = theta_i
	
	# main simulation loop
	for k in tqdm(xrange(int(T/dt))):
		
		#integration step
		phi = step(phi,F,dt)
		
		# record rates
		rec.append(phi[:2])
		
	#convert list to 2d-array
	rec = np.array(rec)
	
	plt.figure(figsize=(default_fig_width*0.7,3.*0.7))
	
	# plot time course of population rates
	plt.plot(np.linspace(0.,T,int(T/dt)),rec[:,0])
	plt.plot(np.linspace(0.,T,int(T/dt)),rec[:,1])
	# plot steady state population rates of spiking network for comparison
	plt.plot([0,T],[3.,3.],'--',c=mpl.rcParams['axes.color_cycle'][0])
	plt.plot([0,T],[6.768,6.768],'--',c=mpl.rcParams['axes.color_cycle'][1])
	plt.ylim([0.,8.])
	plt.xlabel("T [s]")
	plt.ylabel("Frequency [Hz]")
	
	#save figure
	for f_f in file_format:
		plt.savefig(file_loc + f_f)

		
	plt.show()
	
	print (phi[0],phi[1])
	
	
# run a simulation with the mean thresholds found in the full spiking network
fixed_th(-56.963,-58.,savefolder + plot_filename)
