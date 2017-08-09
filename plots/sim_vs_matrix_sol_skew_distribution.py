import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from custom_modules.plot_setting import *
from custom_modules.dist_mat import *
from custom_modules.psi_interact import *
from custom_modules.frequ_from_spikes import *
from custom_modules.rand_pos_min_dist import *
from tqdm import tqdm

import json

# This script plots the relation between diffusion constant and the skewness of the resulting firing rate distribution, including its
# variation among subsequent simulations. It compares it to the distribution/relation predicted by solving the steady state of activity for
# randomly generated cell positions.
### caution !! This script takes long to run!

## Simulation scripts required to run before this script:
# -diff_constant_test_neumann

# where to find sim. data and where to save figure
sim_data_file = sim_data_base_folder + "Diffusion_constant_test_neumann/data_list.dat"
savefolder = plots_base_folder + "firing_rate_dists/"
plot_filename = "mat_sim_skew_dist"

# resolution of the histogram for the 2d heatmap showing the (semi)analytic prediction in D- and "skew"-direction
n_D = 50
n_skew = 50

D_space = np.linspace(0.,20.,n_D) # space of diff. constants
skew_space = np.linspace(0.,1.5,n_skew+1) # space of skewness

# initialize array to be displayed as heatmap
H = np.zeros((n_D,n_skew))

# Number of Neurons
N_n = 400
# side length of tissue, micrometer
L = 1000.
# minimum distance between neurons
min_dist = 10.

# generate 200 random sets of positions and solve the diffusive interaction matrix for all values in D_space
for k in xrange(200):
	
	X = rand_pos_min_dist(N_n,L,min_dist) # random positions
	
	D = dist_mat_from_pos(X,3,3,L,"neumann") # distance matrix
	
	for l in tqdm(xrange(n_D)):
		
		Psi = psi(D,D_space[l]*1000.,0.1,min_dist,10.).sum(axis=0) # interaction matrix for given diff. constant
		
		target = np.dot(Psi,np.ones(N_n)*3.).mean() # calculate target conc. given 3 Hz target rate
		
		r = np.linalg.solve(Psi,np.ones(N_n)*target) # solve for activity
		
		if r.std() != 0: # calculate skewness if std. dev. is nonzero
			skew = (((r-r.mean())/r.std())**3).mean()
		else: #  else, set skewn. to zero
			skew = 0.
		
		h = np.histogram([skew],bins=skew_space) # generate histogram for given diff. constant
		
		H[l,:] += h[0] # add it to overall statistics


for k in xrange(n_D):
	H[k,:] = H[k,:]/H[k,:].max() # normalize maximum of each slice in vertical direction to 1 for better visualization/usage of the color mapping


D_space = np.linspace(0.,20. + 20./(n_D-1),n_D+1) # update D_space for plotting


data_list = []

# load simulation data
with open(sim_data_file) as f:
	for line in f:
		try:
			data_list.append(json.loads(line))
		except:
			print "Error reading line"

# number of simulations recorded
n=len(data_list)

D_arr = range(n)
CF_arr = range(n)
spt_arr = range(n)

# copy data from dictionary
for k in xrange(n):
	
	D_arr[k] = data_list[k]['D']
	
	CF_arr[k] = data_list[k]['CF']
	
	spt_arr[k] = data_list[k]['spt']
	


# function for plotting skewness taken from sim. data
def plot_skew_D(spt_arr,D_arr,t_begin,t_end,ax):
	
	# number of sim. trials
	n = len(spt_arr)
	
	# number of neurons
	n_cells = len(spt_arr[0])
	
	skew_arr = []
	
	D_arr_filt = []
	
	for j in xrange(n):

		frequs = np.zeros(n_cells)
	
		for k in xrange(n_cells):
			
			# calc. mean firing rate for each cell
			times=np.array(spt_arr[j][k])
			frequs[k]=((times>=t_begin) * (times <= t_end)*1.).sum()/(t_end-t_begin)
		
		# see above
		if frequs.std() != 0:
			skew = (((frequs-frequs.mean())/frequs.std())**3).mean()
		else:
			skew = 0.

		# only depict positive skew (negative skew only appears for very small diffusion constants,
		# where the width of the distr. approaches very small values anyway!)
		if skew > 0:
			skew_arr.append(skew)
			D_arr_filt.append(D_arr[j])

	skew_arr = np.array(skew_arr)
	ax.plot(D_arr_filt,skew_arr,'.')

fig,ax = plt.subplots(figsize=(default_fig_width*0.8,default_fig_width*0.8*0.6))

# plot results
ax.pcolormesh(D_space,skew_space,H.T,cmap="Greys")
plot_skew_D(spt_arr,D_arr,1000.,1500.,ax)
ax.set_xlabel("D $\\mathrm{[\\mu m^2 /ms]}$")
ax.set_ylabel("$\\mathrm{Skewness}$")
ax.set_xlim([D_space[0],D_space[-1]])
ax.set_ylim([skew_space[0],skew_space[-1]])
ax.set_title("B", loc="left")

# save figure
for f_f in file_format:
	plt.savefig(savefolder + plot_filename + f_f)

plt.show()
