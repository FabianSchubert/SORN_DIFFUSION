import numpy as np
import cPickle as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
import sys
import json

from custom_modules.plot_setting import *

# This script plots the standard deviation of the steady state excitatory firing rate distribution
# against the diffusion constant D.

## Simulation scripts required to run before this script:
# -diff_constant_test_neumann

# locations of sim. data and where to save figure
sim_data_file = sim_data_base_folder + "Diffusion_constant_test_neumann/data_list.dat"
savefolder = plots_base_folder + "firing_rate_dists/"
plot_filename = "std_dev_vs_D_neumann"

# prepare loading data
data_list = []

# open sim. data
with open(sim_data_file) as f:
	for line in f:
		try:
			data_list.append(json.loads(line))
		except:
			print "Error reading line"

# number of successfully read simulation datasets
n=len(data_list)

# Lists of Diffusion constants and spiketimes
D_arr = range(n)
spt_arr = range(n)

#write data to these lists
for k in xrange(n):
	
	D_arr[k] = data_list[k]['D']
	
	# nomenclature may vary
	try:
		spt_arr[k] = data_list[k]['spt']
	except:
		spt_arr[k] = data_list[k]['spt_e']
	


# function that plots the previously described property
def plot_std_D(spt_arr,D_arr,t_begin,t_end,ax):
	
	# number of datasets
	n = len(spt_arr)
	
	# number of Neurons
	n_cells = len(spt_arr[0])
	
	# initialize std. dev. list
	std_arr = []
	
	# Some datasets could not be properly analyzed. These were discarded before plotting to avoid artifacts
	D_arr_filt = []
	
	for j in xrange(n):

		frequs = np.zeros(n_cells)
	
		for k in xrange(n_cells):
			
			# calculate mean frequencies within the given time window
			times=np.array(spt_arr[j][k])
			frequs[k]=((times>=t_begin) * (times <= t_end)*1.).sum()/(t_end-t_begin)
		
		# calculate std. deviations
		std = np.std(frequs)
		
		# check for the previously mentioned artifacts
		if std > 0:
			std_arr.append(std)
			D_arr_filt.append(D_arr[j])
	
	#plot std. dev. against D
	std_arr = np.array(std_arr)
	ax.plot(D_arr_filt,std_arr,'.')

fig , ax = plt.subplots(figsize=(default_fig_width*0.5,default_fig_width*0.5))

# call plot_std_D with the loaded data and a time window of 1000-1500s
plot_std_D(spt_arr,D_arr,1000.,1500.,ax)

ax.set_xlabel(r"$\mathrm{Diffusion\, Constant\, D\, [\mu m^2/ms]}$")
ax.set_ylabel(r"$\mathrm{Standard\, Deviation\, of\, Firing\, Rate\, Distribution\, [Hz]}$")
ax.set_xlim([0.,22.])
ax.set_ylim([0.,1.3])
ax.set_title("A",loc="left")

#save figure
for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)

plt.show()

#pdb.set_trace()
