import numpy as np
import matplotlib.pyplot as plt
from custom_modules.plot_setting import *
from tqdm import tqdm

mpl.rcParams['font.size'] = 7.
mpl.rcParams['lines.linewidth'] = 1.

# This script plots the mean and std. deviation of weight changes induced by stdp between normalization steps, averaged over the exc. population

## Simulation scripts required to run before this script:
# -all_diffusive_record_pre_and_post_weights
# -all_non_diffusive_record_pre_and_post_weights


# where to find data and where to save figure
folder_diff = sim_data_base_folder + "complete_diff_prepostnorm/"
folder_non_diff = sim_data_base_folder + "complete_non_diff_prepostnorm/"
savefolder = plots_base_folder + "syn_topology/"
plot_filename = "av_stdp_change_time"

# analysis function
def mean_std_stdp_change(W_pre_file,W_post_file,n_t,n_neur):
	
	# load weight recordings (sparse)
	W_pre = np.load(W_pre_file)[()]
	W_post = np.load(W_post_file)[()]

	W_bin = np.zeros((n_neur,n_neur)).astype("bool")
	W_diff = np.zeros((n_neur,n_neur))

	v_mean = np.zeros(n_t)
	v_std = np.zeros(n_t)
	
	# analyze each time step
	for k in tqdm(xrange(n_t)):
		if k==0: # set mean and std. dev to zero for the origin (nothing could have changed yet)
			v_mean[k] = 0.
			v_std[k] = 0.
		else:
			# only consider connections that existed before and after the stdp induced changes
			W_bin = (W_pre[k*n_neur:(k+1)*n_neur,:].toarray()>0)*(W_post[(k-1)*n_neur:(k)*n_neur,:].toarray()>0)
			# pre-post pairs being connected
			ind_syn = np.where(W_bin==1)
			# calculate difference in weights before and after stdp events
			W_diff = W_pre[k*n_neur:(k+1)*n_neur,:].toarray() - W_post[(k-1)*n_neur:(k)*n_neur,:].toarray()
			# calculate mean and std. dev. of these changes
			v_mean[k] = W_diff[ind_syn[0],ind_syn[1]].mean()
			v_std[k] = W_diff[ind_syn[0],ind_syn[1]].std()

	return v_mean,v_std

# 1000 time steps, one second in between
n_t = 1000

# analyze data for diffusive hom.
v_mean_diff,v_std_diff = mean_std_stdp_change(folder_diff+"W_eTOe_prenorm.npy",folder_diff+"W_eTOe_postnorm.npy",n_t,400)
# analyze data for non-diffusive hom.
v_mean_non_diff,v_std_non_diff = mean_std_stdp_change(folder_non_diff+"W_eTOe_prenorm.npy",folder_non_diff+"W_eTOe_postnorm.npy",n_t,400)

fig,ax = plt.subplots(2,1,figsize=(default_fig_width,default_fig_width*0.7))

# plot results
ax[0].plot(v_mean_diff*1000,label="Diffusive H.")
ax[0].plot(v_mean_non_diff*1000,label="Non Diffusive H.")
ax[0].set_xlabel("t [s]")
ax[0].set_ylabel(r"$\mathrm{\mu_{\,STDP}\, [mV/s]}$")

ax[1].plot(v_std_diff*1000,label="Diffusive H.")
ax[1].plot(v_std_non_diff*1000,label="Non Diffusive H.")
ax[1].set_xlabel("t [s]")
ax[1].set_ylabel(r"$\mathrm{\sigma_{\,STDP}\, [mV/s]}$")

# save figure
for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)

plt.show()
