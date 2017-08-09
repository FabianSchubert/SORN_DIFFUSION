import numpy as np
import matplotlib as mpls
import matplotlib.pyplot as plt
import cPickle as pickle
from custom_modules.plot_setting import *
from custom_modules.frequ_from_spikes import *
import sys
import pdb

# This script plots the number of outgoing excitatory connections among the excitatory population against their mean firing rate.

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
plot_filename = "out_degree_vs_rate"

fig,ax = plt.subplots(1,1,figsize=(default_fig_width,default_fig_width*0.37))

# analyze different cases of homeostasis
for k in xrange(len(folders)):
	for l in xrange(len(folders[k])):
		# load weights
		W = np.load(sim_data_base_folder + folders[k][l] + "W_eTOe.npy").T
		
		# calculate total number of outgoing weights
		out_weights = (W!=0).sum(axis=0)
		
		# load spiketimes
		spt = pickle.load(open(sim_data_base_folder + folders[k][l]+"spiketimes_e.p")).values()
		
		# calculate firing rates (averaged over 700-1000s)
		f = frequ_vec(spt,700,1000)
		
		# scatter data points
		if l==0:
			ax.plot(f,out_weights,'.',label=labels[k],c=mpl.rcParams['axes.color_cycle'][k])
		else:
			ax.plot(f,out_weights,'.',c=mpl.rcParams['axes.color_cycle'][k])
			
		

ax.set_xlabel("f [Hz]")

ax.set_ylabel("Out Degree")

ax.set_title("D",loc="left")

#save figure
for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)

plt.show()
	
