import numpy as np
import matplotlib as mpls
import matplotlib.pyplot as plt
import cPickle as pickle
from custom_modules.plot_setting import *
from custom_modules.frequ_from_spikes import *

# This script plots the mean of outgoing excitatory weights among the excitatory population against their mean firing rate.

## Simulation scripts required to run before this script:
# -all_diffusive_slow_hom_adaptation
# -all_instant_diffusion
# -all_non_diffusive

# where to find data
folders = [["complete_diff_long/"],
	["complete_instant_diff_long/"],
	["complete_non_diff_long/"]]

# list of labels for plotting
labels = ["Diffusive",
"Instant Diffusion",
	"Non-Diffusive"]

# where to save figure		
savefolder = plots_base_folder + "syn_topology/"
plot_filename = "mean_out_weights_vs_rate"

fig,ax = plt.subplots(2,1,figsize=(default_fig_width,default_fig_width*0.74))

# analyze different cases of homeostasis
for k in xrange(len(folders)):
	for l in xrange(len(folders[k])):
	
		# load weights
		W = np.load(sim_data_base_folder + folders[k][l] + "W_eTOe.npy").T
		
		# calculate total number of outgoing weights
		out_weights = (W.sum(axis=0)/(W!=0.).sum(axis=0))*1000.
		
		# load spiketimes
		spt = pickle.load(open(sim_data_base_folder + folders[k][l]+"spiketimes_e.p")).values()
		
		# calculate firing rates (averaged over 700-1000s)
		f = frequ_vec(spt,700,1000)
		
		# scatter data points
		if l==0:
			ax[0].plot(f,out_weights,'.',label=labels[k],c=mpl.rcParams['axes.color_cycle'][k])
			ax[1].plot(f,out_weights,'.',label=labels[k],c=mpl.rcParams['axes.color_cycle'][k])
		else:
			ax[0].plot(f,out_weights,'.',c=mpl.rcParams['axes.color_cycle'][k])
			ax[1].plot(f,out_weights,'.',c=mpl.rcParams['axes.color_cycle'][k])
		

ax[0].set_xlabel("f [Hz]")
ax[1].set_xlabel("f [Hz]")

ax[0].set_ylabel("Mean of Outg. W. [mV]")
ax[1].set_ylabel("Mean of Outg. W. [mV]")



ax[1].set_yscale("log")

ax[1].set_ylim([10**-2.,50.])

ax[0].set_title("A",loc="left")
ax[1].set_title("B",loc="left")

ax[0].legend(loc=0)

#save figure
for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)

plt.show()
