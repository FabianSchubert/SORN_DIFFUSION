import numpy as np
import matplotlib.pyplot as plt
from custom_modules.plot_setting import *
import cPickle as pickle
from custom_modules.frequ_from_spikes import *
from custom_modules.dist_mat import *
from custom_modules.gauss import *

#This script plots excitatory firing rates and sum of outgoing weights against the estimated reciprocal neuronal density

## Simulation scripts required to run before this script:
# -Diff_distance_topology

# folders of simulation data and for saving the figure
folder = sim_data_base_folder + "Diff_topology_versions/Diff_distance_topology/"
savefolder = plots_base_folder
plot_filename = "scatter_density_out_weights_points"

# load spiketime data and neurons' positions
spt = pickle.load(open(folder+"spiketimes_e.p")).values()
X = np.array(pickle.load(open(folder+"X_e.p")))

# calculate vector of excitatory firing rates, averaged between 1200-1500s 
f = frequ_vec(spt,1200.,1500.)

# calculate distance matrix from positions
D = dist_mat_from_pos(X,2,2,1000.,"neumann")

# estimate neuronal density by means of a gaussian kernel of 70 micrometer FWHM
dens = gauss2d(D,0.,50.).sum(axis=(0,1))

# load excitatory recurrent weights
W = np.load(folder + "W_eTOe.npy").T

# calculate vector of sum of outgoing weights
out = W.sum(axis=0)

# calculate pearson corr. coeff. between 1/dens and the dec. logarithm of the sum of outgoing weights
print("Coeff. of Corr. = " + str(np.corrcoef(1./dens,np.log10(out))[0,1]))

fig,ax  = plt.subplots(1,2,figsize=(default_fig_width,default_fig_width*0.5))

#plot 1/dens against firing rates

ax[0].plot(1./(dens),f,'.')
ax[0].set_title("C",loc="left")
ax[0].set_xlabel("1/Neuron Density $\\mathrm{[\\mu m^{2}]}$")
ax[0].set_ylabel("$\\mathrm{f\\; [Hz]}$")

# plot 1/dens against sum of outgoing weights
ax[1].plot(1./(dens),out,'.')
ax[1].set_title("D",loc="left")
ax[1].set_xlabel("1/Neuron Density $\\mathrm{[\\mu m^{2}]}$")
ax[1].set_ylabel("$\\mathrm{log_{10}\,(W_{sum,out})}$")
ax[1].set_yscale("log")

ax[0].locator_params(axis='x',nbins=4)
ax[1].locator_params(axis='x',nbins=4)

# save figure
for f_f in file_format:
	plt.savefig(savefolder + plot_filename + f_f)

plt.show()
