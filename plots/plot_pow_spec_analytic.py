import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from custom_modules.frequ_from_spikes import *
from custom_modules.plot_setting import *

from scipy.optimize import curve_fit as curve_fit


mpl.rcParams['lines.linewidth'] = 1.

# This script fits an analytic expression for the power spectrum of excitatory thresholds to simulation data

## Simulation scripts required to run before this script:
# -all_diffusive_fast_hom_adaptation

# simulation data and plot folder
folder = sim_data_base_folder + "complete_diff_long_small_tau/"
savefolder = plots_base_folder +  "power_spec/"
plot_filename = "power_spectrum_analytic"

# time resolution of threshold recording
dt = 0.1

# use thresholds for t>1000s for analysis
ind_start = 10000

# load thresholds
th = (np.array(pickle.load(open(folder+"thresholds_e.p"))).T)[ind_start:,:]*1000.
# calculate population mean
th_mean = th.mean(axis=1)

# calculate fourier transforms
fft_th_mean = np.fft.fft(th_mean)
# discard offset
fft_th_mean[0] = 0.

# calculate power spectrum
pow_th_mean = np.abs(fft_th_mean)**2*dt/fft_th_mean.shape[0]
# subsample power spectrum
pow_th_mean_smooth = np.reshape(pow_th_mean,(pow_th_mean.shape[0]/5,5)).mean(axis=1)

# linearly estimate the relation between excitatory firing rate and threshold
spt = pickle.load(open(folder + "spiketimes_e.p"))

f_arr = frequ_bin_time(spt,0,1500,15000)

f_mean = f_arr.mean(axis=1)[ind_start:]

fit = np.polyfit(th_mean,f_mean,deg=1)

# analytic expression for fitting
def pow_analytic(f, sigm_noise = 1., alpha = fit[0]*1000., NO_0 = 2.803*10.**-5.,lambd=0.1,**kwargs):
    
	if 'alpha' in kwargs:
		alpha = kwargs['alpha']
	if 'sigm_noise' in kwargs:
		sigm_noise = kwargs['sigm_noise']
	if 'NO_0' in kwargs:
		NO_0 = kwargs['NO_0']
	if 'lambd' in kwargs:
		lambd = kwargs['lambd']
	
	w = np.pi * 2. * f

	r_0 = 3. # target rate
	tau_ca = 0.01 # time constant of calcium dynamics, seconds
	tau_nos = 0.1 # time constant of nNOS dynamics, seconds
	L = 1000. # side length of tissue, micrometer
	tau_th = 2.5 # time constant of threshold adaptation, seconds
	N=400 # number of neurons
	gamma = tau_ca*np.log(2.)/(3.) # approximate proportionality factor between nNOS and firing rate
	
	p = sigm_noise**2/(w**2*NO_0**2*tau_th**2*lambd**2+(w**2*NO_0*tau_th+gamma*alpha*N/L**2)**2) # analytic expression for the power spectrum
	
	#p = sigm_noise**2/(w**2*(lambd*NO_0*tau_th + alpha*N*gamma*tau_nos/(L**2*(w**2*tau_nos**2 + 1.)))**2 + (w**2*NO_0*tau_th + alpha*N*gamma/(L**2*(w**2*tau_nos**2 + 1.)))**2)
		
	# return in units of mV^2
	return 1000**2 * p

## different fitting functions with different numbers of free parameters

def pow_fit_sigm(x,sigm):
	
	return pow_analytic(x,sigm_noise=sigm)

def pow_fit_sigm_lambd(x,sigm,l):
	
	return pow_analytic(x,sigm_noise=sigm,lambd=l)

def pow_fit_sigm_alpha_lambd(x,sigm,a,l):
	
	return pow_analytic(x,sigm_noise=sigm,alpha=a,lambd=l)



f_ax = np.linspace(0,1./dt,15000-ind_start)

# only fit by means of sigma
par_fits_sigm = curve_fit(pow_fit_sigm,f_ax[:int(f_ax.shape[0]/2.)],pow_th_mean[:int(f_ax.shape[0]/2.)],p0=[.4*10**-6.])

# fit with sigma and lambda
par_fits_sigm_lambd = curve_fit(pow_fit_sigm_lambd,f_ax[:int(f_ax.shape[0]/2.)],pow_th_mean[:int(f_ax.shape[0]/2.)],p0=[.4*10**-6.,1.])

# fit with sigma, lambda and alpha
par_fits_sigm_alpha_lambd = curve_fit(pow_fit_sigm_alpha_lambd,f_ax[:int(f_ax.shape[0]/2.)],pow_th_mean[:int(f_ax.shape[0]/2.)],p0=[.4*10**-6.,-1000.,1.])


plt.figure(figsize=(default_fig_width,default_fig_width*0.5))

# plot power spectrum and the respective fits

plt.plot(f_ax[::5],pow_th_mean_smooth,label=r"Power Spectrum of $\mathrm{\langle V_t \rangle}$")

plt.plot(f_ax,pow_analytic(f_ax,sigm_noise=par_fits_sigm[0][0]),label=r"Noise Amplitude Fit")

plt.plot(f_ax,pow_analytic(f_ax,sigm_noise=par_fits_sigm_lambd[0][0],lambd=par_fits_sigm_lambd[0][1]),label=r"Noise Amplitude and $\mathrm{\lambda}$ Fit")

plt.plot(f_ax,pow_analytic(f_ax,sigm_noise=par_fits_sigm_alpha_lambd[0][0],alpha=par_fits_sigm_alpha_lambd[0][1],lambd=par_fits_sigm_alpha_lambd[0][2]),label=r"Noise Amplitude, $\mathrm{\lambda}$ and $\mathrm{\alpha}$ Fit",c='k')



#par_fits = curve_fit(pow_fit,f_ax[:int(f_ax.shape[0]/2.)],pow_th_mean[:int(f_ax.shape[0]/2.)],p0=[0.04,-1000.,1.])

#plt.plot(f_ax,par_fits[0][2]*pow_analytic(f_ax,tau_m=par_fits[0][0],c=par_fits[0][1]),'--',label="Best fit",c='r')

plt.xlim([0.,1.5])

plt.xlabel(r"$\mathrm{f\,[Hz]}$")
plt.ylabel(r"$\mathrm{P(f)\,[mV^2]}$")

plt.legend()

# save figure
for f_f in file_format:
	plt.savefig(savefolder + plot_filename + f_f)

plt.show()
