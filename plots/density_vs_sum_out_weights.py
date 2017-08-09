import numpy as np
import matplotlib.pyplot as plt
from custom_modules.plot_setting import *
import cPickle as pickle
from custom_modules.frequ_from_spikes import *
from custom_modules.dist_mat import *
from custom_modules.gauss import *

#This script plots the estimated reciprocal neuronal density and the sum of outgoing excitatory weights after a simulation run by means of two color-mapped scatter plots.

## Simulation scripts required to run before this script:
# -Diff_distance_topology


# folder of simulation data and for saving figure
folder = sim_data_base_folder + "Diff_topology_versions/Diff_distance_topology/"
savefolder = plots_base_folder
plot_filename = "scatter_density_out_weights"

#Load neurons' positions
X = np.array(pickle.load(open(folder+"X_e.p")))

# Calculate distance matrix of neurons' positions, including "copies" for specific boundary conditions
D = dist_mat_from_pos(X,2,2,1000.,"neumann")

# estimate neuronal density by means of a gaussian kernel of 70 micrometer FWHM
dens = gauss2d(D,0.,35.).sum(axis=(0,1))

# load recurrent excitatory weights
W = np.load(folder + "W_eTOe.npy").T

# calculate vector of sum of outgoing weights
out = W.sum(axis=0)

# calculate pearson corr. coeff. between 1/dens and the dec. logarithm of the sum of outgoing weights
print(np.corrcoef(1./dens,np.log10(out))[0,1])

fig,ax = plt.subplots(1,2,figsize=(default_fig_width,default_fig_width*0.6))

# plot results spatially (color coded outgoing weights)

dens_scatt = ax[0].scatter(X[:,0],X[:,1],linewidth=0.,c=1./(dens*1000**2))
ax[0].set_xlim([0.,1000.])
ax[0].set_ylim([0.,1000.])
ax[0].set_xlabel("$\\mathrm{\\mu m}$")
ax[0].set_ylabel("$\\mathrm{\\mu m}$")
ax[0].set_title("A",loc="left")
cb_dens = plt.colorbar(dens_scatt,ax=ax[0],label="1/Neuron Density $\\mathrm{[mm^{2}]}$",orientation="horizontal")
cb_dens.set_ticks([0.001,0.0045])

weight_scatt = ax[1].scatter(X[:,0],X[:,1],linewidth=0.,c=np.log10(out))
ax[1].set_xlim([0.,1000.])
ax[1].set_ylim([0.,1000.])
ax[1].set_xlabel("$\\mathrm{\\mu m}$")
ax[1].set_ylabel("$\\mathrm{\\mu m}$")
ax[1].set_title("B",loc="left")
cb_weight = plt.colorbar(weight_scatt,ax=ax[1],label="$\\mathrm{log_{10}\,(W_{sum,out})}$",orientation="horizontal")
cb_weight.set_ticks([-2.7,-.3])

# save figure
for f_f in file_format:
	plt.savefig(savefolder + plot_filename + f_f)

plt.show()
