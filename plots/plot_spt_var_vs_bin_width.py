import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import json

from custom_modules.plot_setting import *
from custom_modules.frequ_from_spikes import *

# Script plotting firing rate variance against inverse time bin widths

## Simulation scripts required to run before this script:
# -spiketime_dataset_diff

# sim data and plot folder
folder = sim_data_base_folder + "spiketime_dataset_diff/"
savefolder = plots_base_folder + "Spike_Stats/"
plot_filename = "sim_f_var_vs_hom_poisson"

# load data
with open(folder + "data_list.dat","rb") as datafile:
	data = json.loads(next(datafile))
	#spiketimes
	spt = data["spt_e"]

# function plotting firing rate variance against number of bins (corresponding to inverse bind width)
def plot_spt_var_vs_bin(spt,t_range,n_bin_iterations,bin_range,lab,ax):
	# time span to consider in total
	delta_t = t_range[1]-t_range[0]
	# bin numbers
	range_bins = np.linspace(bin_range[0],bin_range[1],n_bin_iterations).astype('int')
	# calculate overall mean firing rate
	f_mean = frequ_vec([spt],t_range[0],t_range[1])[0]
	
	
	var_f = np.ndarray((n_bin_iterations)) # array for calculating firing rate variance
	var_f_p = f_mean*range_bins/delta_t # theoretical prediction for a poisson process with same overall mean rate
	
	# go through different bin numbers
	for k in xrange(n_bin_iterations):
		bins = range_bins[k]
		# calc. binned firing rates
		f = frequ_bin_time([spt],t_range[0],t_range[1],bins)

		var_f[k] = f.var() #firing rate variance
	
	# plot result and prediction
	p, = ax.plot(range_bins,var_f,label=lab,alpha=0.8)
	ax.plot(range_bins,var_f_p,'--',c=p.get_color())


t_range = [1000.,1500.] # use spiketimes from 1000-1500s

# amount of bin numbers to test
n_bin_iterations = 300

# range of bin numbers
range_bins = [5,1500]

fig,ax = plt.subplots(1,2,figsize=(default_fig_width,default_fig_width*0.5))

# test for 3 random exemplary neurons
for i in xrange(3):
	plot_spt_var_vs_bin(spt[int(np.random.rand()*len(spt))],t_range,n_bin_iterations,range_bins,None,ax[0])

ax[0].set_xlabel(r'Number of Bins')
ax[0].set_ylabel(r'Variance of Signal $\mathrm{[Hz^2]}$')
ax[0].set_xlim(range_bins)
ax[0].set_title("A",loc="left")

# counterexample, using all spiketimes from 0-1500s
plot_spt_var_vs_bin(spt[int(np.random.rand()*len(spt))],[0.,1500.],n_bin_iterations,[2,1500],None,ax[1])

ax[1].set_xlabel(r'Number of Bins')
ax[1].set_ylabel(r'Variance of Signal $\mathrm{[Hz^2]}$')
ax[1].set_xlim(range_bins)
ax[1].set_title("B",loc="left")

# save figure
for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)

plt.show()

