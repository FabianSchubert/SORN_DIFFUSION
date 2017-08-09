import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
from custom_modules.lif_act_funct import *
from custom_modules.plot_setting import *
from tqdm import tqdm

mpl.rcParams['lines.linewidth'] = 1.

# Script for simulating a rate based network using the mean field diffusive model described by Sweeney et al.

## Simulation scripts required to run before this script:
# -all_diffusive_slow_hom_adaptation

# where to save figure
savefolder = plots_base_folder + "rate_threshold_simpl_hip_alpha/"
plot_filename = "rate_threshold_simpl_hip_alpha_comb"

# neuron parameters
tau_m=0.02 # membrane time constant, secons
E_l=-60. # resting potential, mV
V_r=-70. # reset voltage, mV
sigm_noise_mem = np.sqrt(5.) # standard dev. of intrinsic membrane noise, mV

## STP parameters
tau_d = 0.5 # short term depression time constant, seconds
tau_f = 2. # short term facilitation time constant, seconds
U=.04 # maximum conductance

r_goal = 3. # excitatory target firing rate

alpha = [1.,.9,.8,.4] # list of alpha values, setting the mixture between individual activity and population activity for homeostatic adaptation

tau_hip = 1. # time constant of homeostatic intrinsic plasticity

T=20. # simulation run time, seconds
dt = .01 # integration step, seconds
n_timesteps = int(T/dt) # number of simulation steps

t_arr = np.linspace(0.,T,n_timesteps) # time axis

n_int = 200 # parameter defining the accuracy of an integration process used during thw calculation of the f-I function. Larger values -> more accurate

# load weight matrices from a full network simulation
W_ee=np.load(sim_data_base_folder + "complete_diff_long/W_eTOe.npy").T
W_ie=np.load(sim_data_base_folder + "complete_diff_long/W_eTOi.npy").T
W_ei=np.load(sim_data_base_folder + "complete_diff_long/W_iTOe.npy").T
W_ii=np.load(sim_data_base_folder + "complete_diff_long/W_iTOi.npy").T

# combine into one large weight matrix
W=np.ndarray((480,480))

W[:400,:400]=W_ee
W[400:480,:400]=W_ie
W[:400,400:480]=W_ei
W[400:480,400:480]=W_ii

# binary connectivity matrix
W_bin = 1.*(W!=0)

# number of incoming connections
n_e_in = W_bin[:,:400].sum(axis=1)
n_i_in = W_bin[:,400:].sum(axis=1)

# copy of the initial weight matrix
W_comb = np.copy(W)

fig, ax = plt.subplots(2,1,figsize=(default_fig_width,default_fig_width*0.7))

# simulate for different alpha values
for ind_alpha in xrange(len(alpha)):
	
	# initialize random firing rates
	r = np.random.rand(480)*r_goal*2.

	# initialize thresholds
	theta = np.ndarray(480)

	theta[:400]=-57. #initial excitatory neuron threshold
	theta[400:]=-58. #inhibitory neuron threshold

	# recorders for excitatory thresholds and firing rates
	theta_e_rec = np.ndarray((n_timesteps,400))
	r_rec = np.ndarray((n_timesteps,480))

	# main simulation loop
	for k in tqdm(xrange(n_timesteps)):

		# continuous time approximation of STP
		stp_vec = np.dot(W_bin[:400,:400],r[:400])

		stp_vec = (1+stp_vec*tau_f)/(1./U+stp_vec*(tau_d+tau_f)+stp_vec**2*tau_d*tau_f)

		# calculate weights modulated by STP
		W_comb[:400,:400] = np.dot(np.diag(stp_vec),W[:400,:400])
		
		# ignore division by zero
		with np.errstate(divide='ignore', invalid='ignore'):
			# calc mean inc. weights
			mean_e = np.true_divide(W_comb[:,:400].sum(axis=1)*1000,n_e_in)
			mean_e[mean_e == np.inf] = 0
			mean_e = np.nan_to_num(mean_e)
	
			mean_i = np.true_divide(W_comb[:,400:].sum(axis=1)*1000,n_i_in)
			mean_i[mean_i == np.inf] = 0
			mean_i = np.nan_to_num(mean_i)
	    
	    	# calculate total membrane noise (intrinsic + input)
		sigm_noise = np.sqrt(tau_m)*np.sqrt(sigm_noise_mem**2 + mean_e**2*np.dot(W_bin[:,:400],r[:400]) + mean_i**2*np.dot(W_bin[:,400:],r[400:]))
		
		# update firing rates
		r += dt*(phi_noise(np.dot(W_comb*1000,r),theta,V_r,E_l,tau_m,sigm_noise,n_int)-r)/tau_m
		
		# update thresholds
		theta[:400] += dt * (alpha[ind_alpha]*(r[:400].mean()-r_goal)+(1.-alpha[ind_alpha])*(r[:400]-r_goal))/tau_hip
		
		# write to recorders
		r_rec[k,:] = r
		theta_e_rec[k,:] = theta[:400]
	
	# plot time course of random subset of excitatory firing rates
	for k in xrange(30):
		if k==0:
			ax[0].plot(t_arr,r_rec[:,k],alpha=0.9,c=mpl.rcParams['axes.color_cycle'][ind_alpha],label='$\\mathrm{\\alpha=}$'+str(alpha[ind_alpha]))
			ax[1].plot(t_arr,theta_e_rec[:,k],alpha=0.9,zorder=len(alpha)-ind_alpha,c=mpl.rcParams['axes.color_cycle'][ind_alpha],label='$\\mathrm{\\alpha=}$'+str(alpha[ind_alpha]))
		else:
			ax[0].plot(t_arr,r_rec[:,k],alpha=0.9,c=mpl.rcParams['axes.color_cycle'][ind_alpha])
			ax[1].plot(t_arr,theta_e_rec[:,k],alpha=0.9,zorder=len(alpha)-ind_alpha,c=mpl.rcParams['axes.color_cycle'][ind_alpha])



ax[0].set_xlabel("t [sec]")
ax[0].set_ylabel("f [Hz]")
ax[0].legend()
ax[0].set_title("A",loc="left")

ax[1].set_xlabel("t [sec]")
ax[1].set_ylabel("$\\mathrm{V_t}$ [mV]")
ax[1].legend()
ax[1].set_title("B",loc="left")

# save figure
for f_f in file_format:
	plt.savefig(savefolder + plot_filename + f_f)

plt.show()

#pdb.set_trace()
