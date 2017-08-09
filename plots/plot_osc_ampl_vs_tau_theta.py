import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

from custom_modules.plot_setting import *

# Script that compares the prediction of maximum and the associated frequency of the power spectrum of the excitatory neurons' thresholds' population mean to simulation
# data, as a function of the time constant of the threshold adaptation.

## Simulation scripts required to run before this script:
# -tau_theta_sweep

#analytic expression of the shape of the thresholds power spectrum, values were taken from a curve fit to the simulation data
def pow_analytic(f,sigm_noise = 2.675122*10**-4,alpha=-989.873137,lambd=0.591615852,tau_th=2.5,NO_0 = 2.803*10.**-5.,**kwargs):
	
	if 'tau_th' in kwargs:
		tau_theta = kwargs['tau_th']
	
	w=np.pi*2.*f
		
	gamma = 0.01 * np.log(2.)/3.
	L = 1000.
	N_e = 400.
	
	p = sigm_noise**2/(w**2*NO_0**2*tau_th**2*lambd**2+(w**2*NO_0*tau_th+gamma*alpha*N_e/L**2)**2)
	
	return p


# simulation data folder and folder for saving figure
folder = sim_data_base_folder + "tau_theta_sweep/"
savefolder = plots_base_folder + "diff_hom/"
plot_filename = "osc_ampl_vs_tau_th"

# time resolution of threshold recording
dt = 0.1

#load thresholds and calculate population mean
th = np.load(folder+"thresholds_e.npy").T *1000.
th_m = th.mean(axis=1)

# size of recording
n_t = th_m.shape[0]

# load recorded values for the time constant of threshold adaptation
tau_switch_times = np.load(folder+"tau_hip_switch_times.npy")
tau_values = np.load(folder+"tau_hip_value_list.npy")/1000.

# size of tau_hip recording
n_steps = tau_values.shape[0]

snippet_list = [] # list that will contain snippets of the threshold recording corresponding to a particular tau_hip value
pow_list = [] # the power spectra of these snippets
pow_max_list = [] # the maxima of these power spectra

pow_max_analytic_list = [] #  corresponding analytic predictions

pow_max_w_list = [] # frequency associated with the maxima of the spectra
pow_max_w_analytic_list = [] # frequency predicted by analytic expression

t = np.linspace(0.,(th_m.shape[0]-1)*dt,n_t) # time axis

f = np.linspace(0.,1.5,10000) # frequency axis for analytic predictions

f_sim = np.linspace(0.,1./dt,2000) # frequency axis for simulation

smooth_wind_width = 5 # number of indices used for smoothing power spectrum

# cut threshold recording into snippets
for k in xrange(n_steps):
	# append snippets
	snippet_list.append(th_m[np.where((t>=tau_switch_times[k])*(t<tau_switch_times[k+1])==1)[0]])
	# fourier transform
	ft = np.fft.fft(snippet_list[-1])*dt
	# discard offset
	ft[0] = 0.
	# calculate power spectrum from fourier transform
	power = np.abs(ft)**2/(tau_switch_times[k+1]-tau_switch_times[k])
	# append to list
	pow_list.append(power)
	
	# find maximal value in power spectrum and smooth value with sourrounding values
	max_mean = power[np.argmax(power[:int(power.shape[0]/2)])-smooth_wind_width:np.argmax(power[:int(power.shape[0]/2)])+smooth_wind_width+1].mean()
	pow_max_list.append(max_mean)
	
	# append maximum of analytic curve
	pow_max_analytic_list.append(pow_analytic(f,tau_th=tau_values[k]).max())
	
	# append frequencies associated with simulation/analytic maximum
	pow_max_w_list.append(f_sim[np.argmax(power[:int(power.shape[0]/2)])])
	pow_max_w_analytic_list.append(f[np.argmax(pow_analytic(f,tau_th=tau_values[k])[:int(f.shape[0]/2)])])

# convert to numpy array
pow_max_list = np.array(pow_max_list)
#pow_max_list = pow_max_list/pow_max_list.max()
pow_max_analytic_list = np.array(pow_max_analytic_list)
#pow_max_analytic_list = pow_max_analytic_list/pow_max_analytic_list.max()

fig,ax = plt.subplots(1,2,figsize=(default_fig_width,default_fig_width*0.6))

# plot results
ax[0].plot(tau_values,pow_max_list,label="Simulation")
ax[0].plot(tau_values,pow_max_analytic_list,label="Analytic")

ax[0].set_xlim([tau_values[0],tau_values[-1]])
ax[0].set_xlabel(r"$\mathrm{\tau_{V_t}}$")
ax[0].set_ylabel(r"$\mathrm{P_{max}}$")
ax[0].legend()

ax[1].plot(tau_values,pow_max_w_list)
ax[1].plot(tau_values,pow_max_w_analytic_list)

ax[1].set_xlim([tau_values[0],tau_values[-1]])
ax[1].set_xlabel(r"$\mathrm{\tau_{V_t}}$")
ax[1].set_ylabel(r"$\mathrm{f_{max}\,[Hz]}$")

ax[0].set_title("A",loc="left")
ax[1].set_title("B",loc="left")

# save figure
for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)

plt.show()

