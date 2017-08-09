import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from custom_modules.dist_mat import *
from custom_modules.psi_interact import *
from custom_modules.frequ_from_spikes import *
from custom_modules.plot_setting import *
import json

#This script compares the distribution of estimated firing rates based on the spatial configuration with the distribution of actual firing rates found in a full simulation

## Simulation scripts required to run before this script:
# -spiketime_dataset_diff

# where to save figure
savefolder = plots_base_folder + "firing_rate_dists/"
plot_filename = "fir_rate_dist_sim_vs_mat"

# paramters for histogram
n_bins = 50
f_space = np.linspace(0.,10.,n_bins+1)
df = f_space[1]-f_space[0]

h_sim = np.zeros(n_bins)
h_mat = np.zeros(n_bins)

# load and analyse simulation data
with open(sim_data_base_folder + "spiketime_dataset_diff/data_list_new.dat","rb") as datafile:
	# each row in the datafile corresponds to a simulation run
	for row in datafile:
		# load data saved with json
		data = json.loads(row)
		spt = data["spt_e"]
		# calculate firing rates averaged over 1200-1500s
		f = frequ_vec(spt,1200.,1500.)
		X = np.array(data["Pos_e"])
		# calculate distance matrix
		D = dist_mat_from_pos(X,2,2,1000.,"neumann")
		# calculate interaction matrix
		Psi = psi(D,10000.,0.1,10.,10.).sum(axis=0)
		# calculate firing rate prediction based on interaction matrix
		target = np.dot(Psi,3.*np.ones(400)).mean()
		f_mat = np.linalg.solve(Psi,np.ones(400)*target)
	
		# add sample histogram to "mean" histograms
		h_sim += np.histogram(f,bins=f_space)[0]
		h_mat += np.histogram(f_mat,bins=f_space)[0]
		




fig = plt.figure(figsize=(default_fig_width*0.7,default_fig_width*0.7*0.6))

#plot normalized histograms
plt.step(f_space[:-1],h_sim/(df*h_sim.sum()),label="Full simulation")
plt.step(f_space[:-1],h_mat/(df*h_mat.sum()),label="Analytic solution")
plt.xlabel("f [Hz]")
plt.ylabel("Prob. Dens.")
plt.legend()

# save figure
for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)

plt.show()
