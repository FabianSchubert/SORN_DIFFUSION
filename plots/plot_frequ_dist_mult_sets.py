import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from custom_modules.frequ_from_spikes import *
from custom_modules.plot_setting import *
from scipy.optimize import curve_fit
import matplotlib.colors as colors
import json
import sys

# This script plots histograms of the firing rates of the exc./inh. population in regular and log space.

## Simulation scripts required to run before this script:
# -spiketime_dataset_diff
# -spiketime_dataset_non_diff

# initialize data lists
files_diff = []
files_non_diff = []

# where to find data
folder_diff = sim_data_base_folder + "spiketime_dataset_diff/"
folder_non_diff = sim_data_base_folder + "spiketime_dataset_non_diff/"
#folder_instant_diff = sim_data_base_folder + "spiketime_dataset_instant_diff/"

#where to save data
savefold = plots_base_folder + "firing_rate_dists/"

# number of bins for histogram
n_bins = 40

# command-line argument, indicating whether to analyze "exc" or "inh" population
plot_pop = sys.argv[1]

# Gaussian function for curve fitting
def gauss(x,A,mu,sigm):
	
	return A*np.exp(-(x-mu)**2/(2.*sigm**2))/(np.sqrt(2.*np.pi)*sigm)

# settings depending on command-line argument
if plot_pop == "exc":
	#### excitatory
	regular_bin_range = [0.,8.]
	log_bin_range = [-.3,1.1]
	y_lim_reg = [0.,.7]
	y_lim_log = [0.,4.]
	title_reg = "A - excitatory"
	title_log = "B - excitatory"
	plot_filename = "fir_rate_dist_e_compare"
	
elif plot_pop == "inh":
	#### inhibitory
	regular_bin_range = [0.,25.]
	log_bin_range = [0.,1.5]
	y_lim_reg = [0.,.3]
	y_lim_log = [0.,3.]
	title_reg = "A* - inhibitory"
	title_log = "B* - inhibitory"
	plot_filename = "fir_rate_dist_i_compare"
	
else:
	print "Wrong Population Argument"
	sys.exit()



# analysis and plotting function
def plot_dist(folder,population,t_range,bins_hist,plot_type,label_hist,plot_col,ax):

	spt_sets = []
	spt_sets_joint = []
	
	# open datafile
	with open(folder + "data_list.dat","rb") as datafile:
		# each row in file corresponds to one simulation dataset
		for row in datafile:
			
			data = json.loads(row)
			if population == "exc":
				spt_sets.append(data["spt_e"])
			elif population == "inh":
				spt_sets.append(data["spt_i"])
			else:
				sys.exit()
			# add spiketime sets to an overall list
			spt_sets_joint.extend(spt_sets[-1])
	
	# total number of neurons contributing to statistics
	n_total = len(spt_sets_joint)
	
	# calculate firing rates
	if plot_type == "log":
		f_arr = np.log10(frequ_vec(spt_sets_joint,t_range[0],t_range[1]))
	elif plot_type == "regular":
		f_arr = frequ_vec(spt_sets_joint,t_range[0],t_range[1])
	else:
		print("Error - Wrong plot type argument")
	# calculate skewness
	skew = (((f_arr-f_arr.mean())/f_arr.std())**3.).mean()
	print(label_hist + " " + plot_type + " plot, skewness: " + str(skew))
	
	# initialize histogram array
	h = np.histogram(f_arr,bins=bins_hist)
	
	# normalize histogram
	y = h[0]/(h[0].sum()*(h[1][1]-h[1][0]))
	
	# x-axis, use midpoints of bins
	x_step = (h[1][1:] + h[1][:-1])*0.5
	
	#plot histogram
	p, = ax.step(x_step,y,where="mid",label=label_hist,c=plot_col)
	
	# fit gaussian to excitatory population in log space
	if plot_type == "log" and label_hist != "Non-Diff. H." and plot_pop == "exc":
		
		# set line color
		col = colors.colorConverter.to_rgb(plot_col)
		
		# fit gaussian curve to histogram
		fit,err = curve_fit(gauss,x_step,y,[1.,.45,.1])
		x = np.linspace(bins_hist[0],bins_hist[-1],1000)
		# plot fitted curve
		plt.plot(x,gauss(x,fit[0],fit[1],fit[2]),'--',c=(col[0]*0.5,col[1]*0.5,col[2]*0.5))
		print("Gaussian fit " + label_hist + ":")
		print("mu = " + str(fit[1]) + ", std-err = " + str(np.sqrt(np.diag(err))[1]))
		print("sigm = " + str(fit[2]) + ", std-err = " + str(np.sqrt(np.diag(err))[2]))
	

fig, ax = plt.subplots(1,2,figsize=(default_fig_width,default_fig_width*0.5))

# bins for regular and log space
bins_h_regular = np.linspace(regular_bin_range[0],regular_bin_range[1],n_bins+1)
bins_h_log = np.linspace(log_bin_range[0],log_bin_range[1],n_bins+1)

# analyze diff. and non-diff. case in regular space
plot_dist(folder_non_diff,plot_pop,[1000.,1500.],bins_h_regular,"regular","Non-Diff. H.",mpl.rcParams['axes.color_cycle'][0],ax[0])
plot_dist(folder_diff,plot_pop,[1000.,1500.],bins_h_regular,"regular","Diff. H.",mpl.rcParams['axes.color_cycle'][1],ax[0])
#plot_dist(folder_instant_diff,plot_pop,[1000.,1500.],bins_h_regular,"regular","Instant. Diff. H.",mpl.rcParams['axes.color_cycle'][2],ax[0])
if plot_pop == "exc":
	ax[0].legend()
ax[0].set_title(title_reg,loc="left")
ax[0].set_ylim(y_lim_reg)
ax[0].set_xlim(regular_bin_range[0],regular_bin_range[1])
ax[0].set_xlabel("f [Hz]")
ax[0].set_ylabel("Prob. Dens.")

# analyze diff. and non-diff. case in regular space
plot_dist(folder_non_diff,plot_pop,[1000.,1500.],bins_h_log,"log","Non-Diff. H.",mpl.rcParams['axes.color_cycle'][0],ax[1])
plot_dist(folder_diff,plot_pop,[1000.,1500.],bins_h_log,"log","Diff. H.",mpl.rcParams['axes.color_cycle'][1],ax[1])
#plot_dist(folder_instant_diff,plot_pop,[1000.,1500.],bins_h_log,"log","Instant. Diff. H.",mpl.rcParams['axes.color_cycle'][2],ax[1])
#ax[1].legend()
ax[1].set_title(title_log,loc="left")
ax[1].set_ylim(y_lim_log)
ax[1].set_xlim(log_bin_range[0],log_bin_range[1])
ax[1].set_xlabel(r"$\mathrm{log_{10}(f) \, [log_{10}(Hz)]}$")
ax[1].set_ylabel("Prob. Dens.")

# save figure
for format in [".png",".svg",".eps"]:
	plt.savefig(savefold+plot_filename+format)

plt.show()



#import pdb
#pdb.set_trace()






