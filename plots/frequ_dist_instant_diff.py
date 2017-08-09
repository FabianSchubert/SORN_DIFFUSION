import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from custom_modules.frequ_from_spikes import *
from custom_modules.plot_setting import *
import json
import sys

#This script plots the distribution of excitatory/inhibitory firing rates for the case
#of instantaneous diffusion in regular and log-space.

## Simulation scripts required to run before this script:
# -spiketime_dataset_instant_diff

# where to find data
folder = sim_data_base_folder + "spiketime_dataset_instant_diffusion/"

# where to save figure
savefold = plots_base_folder + "firing_rate_dists/"

# number of histogram bins
n_bins = 40

# command-line argument whether to analyze excitatory or inhibitory population. Keywords are "exc" and "inh".
plot_pop = sys.argv[1]

# different settings for these cases
if plot_pop == "exc":
	#### excitatory
	regular_bin_range = [0.,10.] # frequency range for histogram in regular space
	log_bin_range = [-.3,1.1] # frequ. range for hist. in log space
	y_lim_reg = [0.,.4] # y- limits of plot in regular...
	y_lim_log = [0.,2.5] # ... and log space
	title_reg = "A" # titles of the panels
	title_log = "B" # "
	plot_filename = "instant_diff_fir_dist_e" # filename of the figure
	dict_key = "spt_e" # dictionary key for accessing spiketime data
elif plot_pop == "inh":
	#### inhibitory - see above for descriptions
	regular_bin_range = [0.,25.]
	log_bin_range = [0.,1.5]
	y_lim_reg = [0.,.2]
	y_lim_log = [0.,2.7]
	title_reg = "A*"
	title_log = "B*"
	plot_filename = "instant_diff_fir_dist_i"
	dict_key = "spt_i"
else:
	print "Wrong Population Argument"
	sys.exit()



# function for plotting the firing rate distribution
def plot_dist(folder,dict_key,t_range,bins_hist,plot_type,label_hist,ax):
	
	# initialize spiketime list
	spt_list = []
	
	# load spiketimes from file
	with open(folder + "data_list.dat","r") as datafile:
		for row in datafile:
			dat = json.loads(row)
			spt_list.extend(dat[dict_key])

	# total number of neurons to be used in statistics
	n_total = len(spt_list)
	
	# calculate firing rate vector
	if plot_type == "log":
		f_arr = np.log10(frequ_vec(spt_list,t_range[0],t_range[1]))
	elif plot_type == "regular":
		f_arr = frequ_vec(spt_list,t_range[0],t_range[1])
	else:
		print("Error - Wrong plot type argument")
	# calcuate skewness of firing rate vector
	skew = (((f_arr-f_arr.mean())/f_arr.std())**3.).mean()
	
	# print skewness and std. dev.
	print(label_hist + " " + plot_type + " plot, skewness: " + str(skew))
	print(label_hist + " " + plot_type + " plot, standard deviation: " + str(f_arr.std()))
	
	# plot histogram
	ax.hist(f_arr,bins=bins_hist,normed=True,histtype="step",label=label_hist)

fig, ax = plt.subplots(1,2,figsize=(default_fig_width,default_fig_width*0.4))

# define bins for regular and logarithmic case
bins_h_regular = np.linspace(regular_bin_range[0],regular_bin_range[1],n_bins+1)
bins_h_log = np.linspace(log_bin_range[0],log_bin_range[1],n_bins+1)


# plot dists in both spaces, with spiketimes averaged over 1000-1500s

plot_dist(folder,dict_key,[1000.,1500.],bins_h_regular,"regular","Instant Diff.",ax[0])
if plot_pop == "exc":
	ax[0].legend()
ax[0].set_title(title_reg,loc="left")
ax[0].set_ylim(y_lim_reg)
ax[0].set_xlim(regular_bin_range[0],regular_bin_range[1])
ax[0].set_xlabel("f [Hz]")
ax[0].set_ylabel("Prob. Dens.")


plot_dist(folder,dict_key,[1000.,1500.],bins_h_log,"log","Instant Diff.",ax[1])
#ax[1].legend()
ax[1].set_title(title_log,loc="left")
ax[1].set_ylim(y_lim_log)
ax[1].set_xlim(log_bin_range[0],log_bin_range[1])
ax[1].set_xlabel(r"$\mathrm{log_{10}(f\, [Hz])}$")
ax[1].set_ylabel("Prob. Dens.")


# save figure
for format in [".png",".svg",".eps"]:
	plt.savefig(savefold+plot_filename+format)

plt.show()
