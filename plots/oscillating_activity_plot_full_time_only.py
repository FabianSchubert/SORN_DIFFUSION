import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

from custom_modules.plot_setting import *
mpl.rcParams['lines.linewidth'] = .8


from custom_modules.frequ_from_spikes import *

# This script plots the unintended oscillations of exc. firing rates, NO concentrations and exc. thresholds that we found (similar to oscillating_activity_plot.py, but without closeups and without inh. population)

## Simulation scripts required to run before this script:
# -all_diffusive_slow_hom_adaptation

# where to find sim data and where to save figure
folder = sim_data_base_folder + "complete_diff_long_small_tau/"
savefolder = plots_base_folder + "diff_hom/"
plot_filename = "rate_NO_th_compl_large_tau"

#load spiketimes
spt_e = pickle.load(open(folder+"spiketimes_e.p","rb"))
spt_i = pickle.load(open(folder+"spiketimes_i.p","rb"))

# unify spiketimes for calculation of population rates
spt_pop_e = []
spt_pop_i = []

for k in xrange(len(spt_e)):
	
	spt_pop_e.extend(spt_e[k])

for k in xrange(len(spt_i)):
	
	spt_pop_i.extend(spt_i[k])

# calculate population rates for a bin width of 0.1s
r_pop_e = frequ_bin_time([spt_pop_e],0,1500,15000)/len(spt_e)
r_pop_i = frequ_bin_time([spt_pop_i],0,1500,15000)/len(spt_i)

# load NO concentration (mean over exc. population)
rho_mean = np.array(pickle.load(open(folder+"rho_rec.p","rb")))

# load thresholds and calculate population mean
th = np.array(pickle.load(open(folder+"thresholds_e.p","rb"))).T

th_mean = th.mean(axis=1)


fig, ax = plt.subplots(3,1,figsize=(default_fig_width,default_fig_width*0.8))


## plot the previously described data
ax[0].plot(np.linspace(0,1500,15000),r_pop_e)
ax[0].plot(np.linspace(0,1500,15000),r_pop_i)
ax[0].set_ylabel(r"$\mathrm{\langle f \rangle \,[Hz]}$")
ax[0].set_xlim([0,1500])
#ax[0].set_ylim([0,8])
ax[0].locator_params(axis='y',nbins=5)
ax[0].set_title("A",loc='left')

ax[1].plot(np.linspace(0,1500,15000),rho_mean*10**5)
ax[1].set_ylabel(r"$\mathrm{\langle NO \rangle\,[10^{-5}]}$")
ax[1].set_xlim([0,1500])
#ax[1].set_ylim([0.0000267*10**5,0.0000295*10**5])
ax[1].locator_params(axis='y',nbins=5)
ax[1].set_title("B",loc='left')

ax[2].plot(np.linspace(0,1500,15000),th_mean*1000)
ax[2].set_ylabel(r"$\mathrm{\langle V_t  \rangle \,[mV]}$")
ax[2].set_xlim([0,1500])
ax[2].locator_params(axis='y',nbins=5)
ax[2].set_title("C",loc='left')

ax[2].set_xlabel("t [s]")



# save figure
for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)


plt.show()
