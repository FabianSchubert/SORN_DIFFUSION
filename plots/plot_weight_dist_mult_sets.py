import numpy as np
import matplotlib.pyplot as plt
import json
from custom_modules.plot_setting import *
from scipy.optimize import curve_fit

# Plot distribution of excitatory recurrent weights in log space, using multiple simulation trials

## Simulation scripts required to run before this script:
# -weight_dataset_diff
# -weight_dataset_non_diff

# sim data folder
folders = ["weights_spiketimes_dataset_non_diff/",
		"weights_spiketimes_dataset_diff/",
		"weights_spiketimes_dataset_instant_diff/"]
# plot labels		
labels = ["Non-Diffusive H.",
		"Diffusive H.",
		"Instant Diff."]

# where to save figure
savefolder = plots_base_folder + "syn_topology/"
plot_filename = "syn_weight_dist_mult_sets"

# Gaussion function for fitting
def gauss(x,A,mu,sigm):
	
	return A*np.exp(-(x-mu)**2/(2.*sigm**2))/(np.sqrt(2.*np.pi)*sigm)


def plot_syn_weight_dist(filename,bin_range,n_bins,label_w,color,ax):
	
	data = []
	# load data
	with open(filename,"rb") as datafile:
		# each row contains data from on simulation run
		for row in datafile:
			data.append(json.loads(row))
			data[-1]["W_eTOe"] = np.array(data[-1]["W_eTOe"])*1000.
	
	# number of exc. neurons
	N = data[-1]["W_eTOe"].shape[0]
	# number of datasets
	n_sets = len(data)
	
	# bins for histogram
	bins_w = np.linspace(bin_range[0],bin_range[1],n_bins+1)
	
	x = (bins_w[1:] + bins_w[:-1])/2. # mid-binning
	dx = x[1]-x[0]
	
	hist = np.ndarray((n_sets,n_bins))
	
	w = np.ndarray((0))
	
	for k in xrange(n_sets):
		
		# discard very small weights
		ind = np.where(data[k]["W_eTOe"]>=10.**-3.)
		# append decadic log. of weights
		w=np.append(w,np.log10(data[k]["W_eTOe"][ind[0],ind[1]]))
		
		# calculate histogram
		h = np.histogram(np.log10(data[k]["W_eTOe"][ind[0],ind[1]]),bins=bins_w)
		
		# write to histogram list
		hist[k,:] = h[0]
	
	# check for data sanity
	if w.std() > 0:
		#calculate skewness
		skew = (((w-w.mean())/w.std())**3).mean()
	else:
		skew = None
	
	# mean over simulation trials
	hist_mean = hist.mean(axis=0)
	# standard error of the mean
	hist_std_err = hist.std(axis=0)/(1.*N)**0.5
	
	# fit a gaussian function to histogram
	fit,err = curve_fit(gauss,x,hist_mean,[1.,0.,1.])
	print("A: "+ str(fit[0]) + "\n" + "mu: " + str(fit[1]) + "\n" + "sigm: " + str(fit[2]))
	print("Err: " + str(np.sqrt(np.diag(err))))
	print("Skew "+label_w+": " + str(skew))
	x_fine = np.linspace(bin_range[0],bin_range[1],1000)
	
	# plot histogram with errorbars	
	h_pl,h_cl,h_bl = ax.errorbar(x,hist_mean,yerr=hist_std_err,fmt='.',label=label_w,c=color)
	# plot Gaussian fit
	fit_p, = ax.plot(x_fine,gauss(x_fine,fit[0],fit[1],fit[2]),c=color)


fig, ax = plt.subplots(figsize=(default_fig_width,default_fig_width*0.5))

bin_range = [-3.,1.5]
n_bins = 30

# plot histogram for both diff. and non-diff. hom.
for k in xrange(3):
	plot_syn_weight_dist(sim_data_base_folder + folders[k] + "data_list.dat",bin_range,n_bins,labels[k],mpl.rcParams['axes.color_cycle'][k],ax)

ax.set_xlabel(r"$\mathrm{log_{10}(Weight) \, [log_{10}(mV)]}$")
ax.set_ylabel("Bin Count")
ax.legend(loc=2)

# save figure
for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)

plt.show()
		
		
			 
		
