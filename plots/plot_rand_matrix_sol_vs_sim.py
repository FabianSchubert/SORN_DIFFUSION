import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from custom_modules.plot_setting import *
from custom_modules.dist_mat import *
from custom_modules.psi_interact import *
from custom_modules.frequ_from_spikes import *
import cPickle as pickle
import json

# script that compares analytic predictions for individual excitatory neurons' firing rates with data from network simulations

## Simulation scripts required to run before this script:
# -spiketime_dataset_instant_diff
# -spiketime_dataset_diff
# -spiketime_dataset_non_diff

# location of simulation data
data_folders = [sim_data_base_folder + "spiketime_dataset_non_diff/",
		sim_data_base_folder + "spiketime_dataset_diff/",
		sim_data_base_folder + "spiketime_dataset_instant_diffusion/"]
# list of plot labels		
labels = ["D = 0 $\\mathrm{\\mu m^2 /ms}$, corr. coeff. $\\mathrm{\\rho}=$",
	"D = 10 $\\mathrm{\\mu m^2 /ms}$, corr. coeff. $\\mathrm{\\rho}=$",
	"D = $\\mathrm{\\infty}$, corr. coeff. $\\mathrm{\\rho}=$"]

# where to save the figure
savefolder = plots_base_folder
plot_filename = "rates_matrix_prediction_vs_sim_new"

# initialize vectors holding firing rates
r_total = np.ndarray((0))
r_sim_total = np.ndarray((0))

# proportionality factor between firing rate and nNOS
gamma = .01*np.log(2.)/(3.)

# target NO concentrations corresponding to different simulation protocols
NO_0 = [3.*gamma/(10.**2*0.1),2.73933 * 10**-5,2.73933 * 10**-5]

lambd = 0.0001
h = 10.

ax_x = 2.737
ax_y = 1.963

margin_x = .1
margin_y = .1

fig = plt.figure(figsize=(ax_x/(1.-2.*margin_x),ax_y/(1.-2.*margin_y)))
#fig = plt.figure(figsize=(default_fig_width,default_fig_width*0.6))

ax = plt.axes([margin_x,margin_y,(1.-2.*margin_x),(1.-2.*margin_y)])

# analyze different simulation protocols
for set_ind in xrange(len(data_folders)):
	print("Analyzing " + labels[set_ind])
	
	r_total = np.ndarray((0))
	r_sim_total = np.ndarray((0))
	#import pdb
	
	# load data
	with open(data_folders[set_ind] + "data_list.dat") as datafile:
		# each row holds data from one simulation run
		for row in datafile:
			#pdb.set_trace()
			data = json.loads(row)
			if set_ind == 0:
				Diff_c = 0.
			elif set_ind == 1:
				Diff_c = data["D"] # used diffusion constant
			else:
				Diff_c = 100.
			spt = data["spt_e"] # spiketimes
			X = np.array(data["Pos_e"]) #neurons' positions
			D = dist_mat_from_pos(X,4,4,1000.,"neumann") # distance matrix
			Psi = psi(D,Diff_c*1000.,lambd*1000.,h,10.).sum(axis=0) #diff. interaction matrix
			r = np.linalg.solve(Psi,np.ones(400)*NO_0[set_ind]/gamma) # firing rate prediction
			r_sim = frequ_vec(spt,1000.,1500.) # firing rates from spikes (averaged over 1200-1500s)
			r_total = np.append(r_total,r) #append firing rates to list
			r_sim_total = np.append(r_sim_total,r_sim) # " 
	
	# calculate pearson correlation coefficient
	corr_coef = np.corrcoef(r_total,r_sim_total)
	corr_coef = corr_coef[0,1]
	
	# check if value is valid
	if corr_coef == corr_coef:
		corr_coef = str(round(corr_coef,3))
	else:
		corr_coef = "undef."
	
	# scatter sim. firing rates against predicted
	ax.plot(r_total,r_sim_total,'.',alpha=0.8,label=labels[set_ind] + corr_coef,zorder=len(data_folders)-1-set_ind)
	

ax.plot([0.,9.],[0.,9.],'--',c='k')
ax.set_xlabel("$\mathrm{f_{matrix \\, predict}\\;[Hz]}$")
ax.set_ylabel("$\mathrm{f_{sim} \\;[Hz]}$")
ax.set_xlim([0.,8.])
ax.set_ylim([0.,9.])
ax.legend()

#save figure
for f_f in file_format:
	plt.savefig(savefolder + plot_filename + f_f)

plt.show()
	
