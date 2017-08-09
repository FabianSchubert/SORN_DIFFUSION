import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

from custom_modules.plot_setting import *
mpl.rcParams['lines.linewidth'] = .8

from custom_modules.frequ_from_spikes import *

# This script plots the unintended oscillations of exc./inh. firing rates, NO concentrations and exc. thresholds that we found.

## Simulation scripts required to run before this script:
# -all_diffusive_fast_hom_adaptation

# where to find sim data and where to save figure
folder = sim_data_base_folder + "complete_diff_long_small_tau/"
savefolder = plots_base_folder + "diff_hom/"
filename = "rate_NO_th_compl"

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


fig, ax = plt.subplots(6,1,figsize=(default_fig_width,8))

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
ax[1].set_ylim([0.0000267*10**5,0.0000295*10**5])
ax[1].locator_params(axis='y',nbins=5)
ax[1].set_title("B",loc='left')

ax[2].plot(np.linspace(0,1500,15000),th_mean*1000)
ax[2].set_ylabel(r"$\mathrm{\langle V_t  \rangle \,[mV]}$")
ax[2].set_xlim([0,1500])
ax[2].locator_params(axis='y',nbins=5)
ax[2].set_title("C",loc='left')

ax[2].set_xlabel("t [s]")

t_range = [680,740]

e_max = r_pop_e[t_range[0]*10:t_range[1]*10].max()
e_min = r_pop_e[t_range[0]*10:t_range[1]*10].min()

e_plt_max = e_max + 0.1*(e_max-e_min)
e_plt_min = e_min - 0.1*(e_max-e_min)

i_max = r_pop_i[t_range[0]*10:t_range[1]*10].max()
i_min = r_pop_i[t_range[0]*10:t_range[1]*10].min()

i_plt_max = i_max + 0.1*(i_max-i_min)
i_plt_min = i_min - 0.1*(i_max-i_min)

ax[3].plot(np.linspace(0,1500,15000),r_pop_e)
#ax[3].plot(np.linspace(0,1500,15000),r_pop_i)
ax[3].set_ylabel(r"$\mathrm{\langle f \rangle \,[Hz]}$")
for tl in ax[3].get_yticklabels():
    tl.set_color(mpl.rcParams['axes.color_cycle'][0])
ax[3].set_xlim([t_range[0],t_range[1]])
ax[3].set_ylim([e_plt_min,e_plt_max])
ax[3].locator_params(axis='y',nbins=5)
ax[3].set_title("A*",loc='left')

ax_inh = ax[3].twinx()
ax_inh.plot(np.linspace(0,1500,15000),r_pop_i,c=mpl.rcParams['axes.color_cycle'][1])
for tl in ax_inh.get_yticklabels():
    tl.set_color(mpl.rcParams['axes.color_cycle'][1])
ax_inh.set_xlim([t_range[0],t_range[1]])
ax_inh.set_ylim([i_plt_min,i_plt_max])

ax[4].plot(np.linspace(0,1500,15000),rho_mean*10**5)
ax[4].set_ylabel(r"$\mathrm{\langle NO \rangle\,[10^{-5}]}$")
ax[4].set_xlim([t_range[0],t_range[1]])
ax[4].set_ylim([0.0000267*10**5,0.0000295*10**5])
ax[4].locator_params(axis='y',nbins=5)
ax[4].set_title("B*",loc='left')

ax[5].plot(np.linspace(0,1500,15000),th_mean*1000)
ax[5].set_ylabel(r"$\mathrm{\langle V_t  \rangle \,[mV]}$")
ax[5].set_xlim([t_range[0],t_range[1]])
ax[5].set_ylim([-60,-52])
ax[5].locator_params(axis='y',nbins=5)
ax[5].set_title("C*",loc='left')

ax[5].set_xlabel("t [s]")


# save figure
for fileformat in [".png",".svg",".eps"]:
	plt.savefig(savefolder+filename+fileformat)


plt.show()
