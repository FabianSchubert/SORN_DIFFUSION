import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from custom_modules.plot_setting import *
from scipy.stats import ks_2samp as kolmogorov_test

# Script that plots the mean of outgoing excitatory weights, ordered by their quantile. Additionally, it is compared to shuffled versions of the weight matrices. Furthermore, it performs kolmogorov-smirnov tests between all sample sets.

## Simulation scripts required to run before this script:
# -all sub-folders of diff_topology_variants

# where to find sim. data
folders = ["complete_non_diff_long/",
	"complete_diff_long/",
	"complete_instant_diff_long/"]
# plot labels
labels = ["Non-Diffusive Homeostasis",
	"Diffusive Homeostasis",
	"Instant Diffusion"]
# where to save figure
savefolder = plots_base_folder + "syn_topology/"
plot_filename = "outgoing_weights_distribution_new"

fig = plt.figure(figsize=(default_fig_width,default_fig_width*0.6))

w_mean_out_arr = [None]*4
w_shuffle_mean_out_arr = [None]*4

# analyze different simulation protocols
for k in xrange(3):
	
	# load weights
	W=np.load(sim_data_base_folder + folders[k] + "W_eTOe.npy")*1000.
	# generate shuffled version of weight matrix
	W_shuffle = np.reshape(np.random.permutation(np.reshape(W,(W.shape[0]*W.shape[1]))),W.shape)
	
	# calculate mean of outgoing weights
	w_mean_out = W.mean(axis=1)
	w_shuffle_mean_out = W_shuffle.mean(axis=1)
	
	# reshape 
	#w_mean_out = np.reshape(w_mean_out,(w_mean_out.shape[0]*w_mean_out.shape[1]))
	#w_shuffle_mean_out = np.reshape(w_shuffle_mean_out,(w_shuffle_mean_out.shape[0]*w_shuffle_mean_out.shape[1]))
	
	# sort mean outgoing weights for plotting by quantile
	w_mean_out_sort = np.sort(w_mean_out)
	w_shuffle_mean_out_sort = np.sort(w_shuffle_mean_out)
	
	w_mean_out_arr[k] = w_mean_out_sort
	w_shuffle_mean_out_arr[k] = w_shuffle_mean_out_sort
	
	color = mpl.rcParams['axes.color_cycle'][k]
	
	# x-axis
	quantile = np.linspace(0.,1.,w_mean_out_sort.shape[0])
	
	# plot weight vectors
	plt.plot(quantile,w_mean_out_sort,c=color,label=labels[k])
	plt.plot(quantile,w_shuffle_mean_out_sort,'--',c=color)
		
for k in xrange(3):
	for l in xrange(3):
		D,p = kolmogorov_test(w_mean_out_arr[k],w_mean_out_arr[l])
		print (labels[k] + " non-shuffled vs. " + labels[l] + " non-shuffled:" + str(D) + ", " + str(p))
		
		D,p = kolmogorov_test(w_mean_out_arr[k],w_shuffle_mean_out_arr[l])
		print (labels[k] + " non-shuffled vs. " + labels[l] + " shuffled:" + str(D) + ", " + str(p))
		
		D,p = kolmogorov_test(w_shuffle_mean_out_arr[k],w_shuffle_mean_out_arr[l])
		print (labels[k] + " shuffled vs. " + labels[l] + " shuffled:" + str(D) + ", " + str(p))

plt.legend(loc=2)
plt.ylim([.01,5.])
plt.xlabel("Quantile")
plt.ylabel("Mean outgoing Weight [mV]")
plt.yscale("log")

# save figure
for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)
plt.show()
	
