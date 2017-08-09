import numpy as np
import matplotlib.pyplot as plt
import json
from custom_modules.plot_setting import *
from custom_modules.frequ_from_spikes import *
from custom_modules.dist_mat import *
from custom_modules.psi_interact import *

# This script plots the distribution of excitatory firing rates of a network simulation that had all excitatory neurons placed on a square grid structure.
#It is compared to the prediction being made by solving the theoretical steady state based on the neurons' positions.

## Simulation scripts required to run before this script:
# -regular_grid_neumann

# where to find sim. data and where to save figure
folder = sim_data_base_folder + "regular_grid_neumann/"
savefolder = plots_base_folder + "firing_rate_dists/"
plot_filename = "regular_grid_firing_rate_dist"

# initialize data list
data = []

# open data file
with open(folder + "data_list.dat","r") as file:
	# each row corresponds to one simulation run
	for row in file:
		data.append(json.loads(row))
		data[-1]["f"] = frequ_vec(data[-1]["spt_e"],1200.,1500.)
		
		# load neurons' positions and solve linear system given by interaction matrix
		X = np.array(data[-1]["Pos_e"])
		D = dist_mat_from_pos(X,2,2,1000.,"neumann")
		Psi = psi(D,10000.,0.1,10.,10.).sum(axis=0)
		target = np.dot(Psi,3.*np.ones(400)).mean()
		data[-1]["f_mat"] = np.linalg.solve(Psi,np.ones(400)*target)

# number of bins for histogram
n_bins = 50
# linspace for binning
f_space = np.linspace(0.,10.,n_bins+1)
# bin width
df = f_space[1] - f_space[0]

# initialize arrays for histogram
h = np.zeros((n_bins))
h_mat = np.zeros((n_bins))

# sum up histograms of all simulation run
for k in xrange(len(data)):
	h += np.histogram(data[k]["f"],bins=f_space)[0]
	h_mat += np.histogram(data[k]["f_mat"],bins=f_space)[0]
# normalize
h = h/(df*h.sum())
h_mat = h_mat/(df*h_mat.sum())

fig = plt.figure(figsize=(default_fig_width*0.7,default_fig_width*0.7*0.6))

# plot hist.
plt.step(f_space[:-1],h,label="Full Simulation",zorder=2)
plt.step(f_space[:-1],h_mat,label="Analytic Solution",zorder=1)
plt.xlabel("f [Hz]")
plt.ylabel("Prob. Dens.")
plt.ylim([0.,h.max()*1.2])
plt.legend()

#save figure
for f_f in file_format:
	plt.savefig(savefolder + plot_filename + f_f)

plt.show()

