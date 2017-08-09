import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from custom_modules.plot_setting import *
from tqdm import tqdm

# This script calculates the population mean of the cross correlation function among the exc./inh. population for diff./non-diff. homeostasis

## Simulation scripts required to run before this script:
# -all_diffusive_slow_hom_adaptation
# -all_non_diffusive

# where to find data
filename_exc_diff = sim_data_base_folder + "complete_diff_long/spiketimes_e.p"
filename_inh_diff = sim_data_base_folder + "complete_diff_long/spiketimes_i.p"
filename_exc_non_diff = sim_data_base_folder + "complete_non_diff_long/spiketimes_e.p"
filename_inh_non_diff = sim_data_base_folder + "complete_non_diff_long/spiketimes_i.p"

# where to save figure
savefolder = plots_base_folder + "Spike_Stats/"
plot_filename = "pop_mean_cross_corr"

# load spiketimes
spt_exc_diff = pickle.load(open(filename_exc_diff))
spt_inh_diff = pickle.load(open(filename_inh_diff))
spt_exc_non_diff = pickle.load(open(filename_exc_non_diff))
spt_inh_non_diff = pickle.load(open(filename_inh_non_diff))

# analysis function
def plot_cross_corr(spt1,spt2,autocorr,t_span,dt_span,n_bins,lab,ax):

	# linspace for binning spiketime differences
	dt_arr = np.linspace(dt_span[0],dt_span[1],n_bins+1)
	# centers of bins
	dt_arr_mid = (dt_arr[1:]+dt_arr[:-1])/2.
	
	# initialize array for cross-corr. function
	total_cross_corr = np.zeros(n_bins)
	
	# number of neurons in pop. one and two
	N1 = len(spt1)
	N2 = len(spt2)
	
	# "autocorr" in this context means that both spiketime lists are the same. Still, cross-correlations are calculated between the neurons within the 
	# corresponding population (probably replace by better name). In this case, half of the spiketime differences are redundant (inverted sign) and
	# the calculation is faster. 
	
	if autocorr:
	
		for k in tqdm(xrange(N1)):
			for l in xrange(N2):
				if k!=l:
					times1 = np.array(spt1[k])
					times2 = np.array(spt2[l])
					
					# discard spiketimes outside of given time span
					times1 = times1[np.where((times1>=t_span[0])*(times1<t_span[1]))[0]]
					times2 = times2[np.where((times2>=t_span[0])*(times2<t_span[1]))[0]]
					
					# calculate meshgrid of spiketime differences
					T1,T2 = np.meshgrid(times1,times2,indexing='ij')
					
					diff = T2-T1
					
					# calculate histogram of these differences					
					total_cross_corr += np.histogram(diff,bins=dt_arr)[0]

	else:
		
		# same as in the other case, except that calculations are carried out over the full set of combinations.
		for k in tqdm(xrange(N1)):
			for l in xrange(N2):
		
				times1 = np.array(spt1[k])
				times2 = np.array(spt2[l])
		
				times1 = times1[np.where((times1>=t_span[0])*(times1<t_span[1]))[0]]
				times2 = times2[np.where((times2>=t_span[0])*(times2<t_span[1]))[0]]
		
				T1,T2 = np.meshgrid(times1,times2,indexing='ij')
		
				diff = T2-T1
		
				total_cross_corr += np.histogram(diff,bins=dt_arr)[0]

	
	# count instances in units of 10^3
	total_cross_corr /= 1000.#(total_cross_corr.sum()*(dt_arr[1]-dt_arr[0]))
	# plot histogram
	ax.step(dt_arr_mid,total_cross_corr,where='mid',label=lab)
	
	
	
fig,ax = plt.subplots(figsize=(default_fig_width,default_fig_width*0.5))

# set time span to analyze (100s already take quite long!)
t_span = [1400.,1500.]
# time span to bin
dt_span = [-.15,.15]
# number of histogram bins
n_bins = 60

# analyze different datasets
plot_cross_corr(spt_exc_diff,spt_exc_diff,True,t_span,dt_span,n_bins,"Diff. H., exc. pop.",ax)
plot_cross_corr(spt_inh_diff,spt_inh_diff,True,t_span,dt_span,n_bins,"Diff. H., inh. pop.",ax)
plot_cross_corr(spt_exc_non_diff,spt_exc_non_diff,True,t_span,dt_span,n_bins,"Non-Diff. H., exc. pop.",ax)
plot_cross_corr(spt_inh_non_diff,spt_inh_non_diff,True,t_span,dt_span,n_bins,"Non-Diff. H., inh. pop.",ax)


ax.set_xlim(dt_span)
ax.set_ylim([0.,9.])

plt.xlabel(r"$\mathrm{\Delta t}$ [s]")
plt.ylabel("Prob. Density")

plt.legend()

#save figure
for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)


plt.show()

		
		




