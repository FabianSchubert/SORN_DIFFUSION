import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from custom_modules.plot_setting import *
import powerlaw
import pdb
# Script to plot the statistics of excitatory recurrent synaptic lifetimes. A powerlaw fit is done with the powerlaw-package

## Simulation scripts required to run before this script:
# -all_diffusive_slow_hom_adaptation
# run convert_weight_to_bin_conn.py

import matplotlib.colors as colors

from scipy.stats import kstest as kolmogorov

# where to save figure
savefolder = plots_base_folder + "syn_topology/"
plot_filename = "syn_lifetimes_new_new"

# -alpha: power law exponent
# -estimated standard error of the fit
# -D: kolmogorov-smirnov distance
alpha_list = []
error_list = []
D_list = []

# settings for finding the maximum lifetime to include into the fit in order to minimize the kolmogorov-smirnov distance
n_range_max = 30
x_max_range = np.linspace(3.,750.,n_range_max)
x_min = 1.0


def analyze_and_plot(filename,pltlabel):
	
	global alpha_list
	global error_list
	global D_list
	
	alpha_list.append(np.ndarray(n_range_max))
	error_list.append(np.ndarray(n_range_max))
	D_list.append(np.ndarray(n_range_max))
	
	## sliding windows for identifying growth, persistence and pruning of a synapse over time
	## 1: synapse exists , 0: synapse does not exist
	
	## case 1: ...[0,0]... -> no synapse present
	case1=np.zeros((2,400,400))
	## case 2: ...[0,1]... -> synapse is created -> start counting lifetime
	case2=np.zeros((2,400,400))
	case2[1,:,:]=1
	## case 3: ...[1,1]... -> synapse persists -> keep counting the lifetime
	case3=np.ones((2,400,400))
	## case 4: ...[1,0]... -> synapse is pruned -> terminate counting of the lifetime
	case4=np.zeros((2,400,400))
	case4[0,:,:]=1

	# load recorded weight matrices
	W=np.load(filename)
	
	# generate binary connectivity array
	W_bin = 1.*(W!=0)
	W_bin=np.append(np.zeros((1,400,400)),W_bin,axis=0)
	
	# initialize lifetimes vector
	lifetimes=np.array([])
	
	# initialize "lifetime counter"
	count = np.zeros((400,400))
	
		
	for k in range(750,1501):
		
		# increase count in case 2 or 3 (see above)
		count+=  (W_bin[[k-1,k],:,:]==case2).prod(axis=0) + (W_bin[[k-1,k],:,:]==case3).prod(axis=0)
		# write lifetimes into array in case 4 
		lifetimes = np.append(lifetimes,count[np.where((W_bin[[k-1,k],:,:]==case4).prod(axis=0)==1)])
		# reset counter in case 4
		count[np.where((W_bin[[k-1,k],:,:]==case4).prod(axis=0)==1)] = 0
	
	# make sure no "zero lifetimes" are included
	lifetimes = lifetimes[np.where(lifetimes!=0)[0]]
	
	# sweep through x_max to find optimal upper fitting bound
	for k in xrange(n_range_max):
		print("Searching optimal x_max... " + str(100.*k/n_range_max))
		# fit exponent via powerlaw-package
		Fit = powerlaw.Fit(lifetimes,xmin=x_min,xmax=x_max_range[k],discrete=True)
		alpha_list[-1][k] = Fit.power_law.alpha
		error_list[-1][k] = Fit.power_law.sigma
		
		# calculate kolmogorov-smirnov distance, given the upper limit
		lifetimes_cut = lifetimes[np.where((lifetimes>=x_min)*(lifetimes<=x_max_range[k])==1)[0]]
		D_list[-1][k] = kolmogorov(lifetimes_cut,Fit.power_law.cdf)[0]
	
	# find optimal upper limit
	ind_min_D = D_list[-1].argmin()
	
	print("Min. d " + pltlabel + ": " + str(D_list[-1].min()))
	
	# fit again for optimal choice
	Fit = powerlaw.Fit(lifetimes,xmin=x_min,xmax=x_max_range[ind_min_D],discrete=True)
	
	# get distribution of lifetimes
	y = powerlaw.pdf(lifetimes)
	
	# logarithmic x-axis
	x = np.exp((np.log(y[0][1:])+np.log(y[0][:-1]))*0.5)
	
	pdb.set_trace()


	print("alpha fit: " + str(alpha_list[-1][ind_min_D]))
	print("standard error: " + str(error_list[-1][ind_min_D]))
	
	# plot dist of lifetimes
	p, = plt.plot(x,y[1],'.',label=pltlabel)
	col = colors.colorConverter.to_rgb(p.get_color())
	Fit.power_law.plot_pdf(c=(col[0]*0.5,col[1]*0.5,col[2]*0.5))
	



fig = plt.figure(figsize=(default_fig_width,default_fig_width*0.5))

print("Analyzing non-diff. case")
analyze_and_plot(sim_data_base_folder + "complete_non_diff_long/W_eTOe_record_bin.npy","Non-Diffusive")
print("Analyzing diff. case")
analyze_and_plot(sim_data_base_folder + "complete_diff_long/W_eTOe_record_bin.npy","Diffusive")
print("Analyzing instant-diff. case")
analyze_and_plot(sim_data_base_folder + "complete_instant_diff_long/W_eTOe_record_bin.npy","Instant Diffusion")

# set x- and y- axis to log scale
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Syn. Lifetime [s]')
plt.ylabel('Count in Bins')
plt.legend()
plt.xlim([4.,800.])

# save figure
for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)
plt.show()
	
