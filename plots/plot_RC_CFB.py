import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import numpy as np
import cPickle as pickle
from custom_modules.plot_setting import *

# Script plotting the time course of the ratio-over-chance value for bidirectional recurrent excitatory connections for different simulation protocols

## Simulation scripts required to run before this script:
# -all sub-folders of connection_fraction

# where to save figure
savefolder = plots_base_folder + "syn_topology/"
plot_filename = "RC_CFB_plot_new"

#0:diff + topol
#1:non-diff + topol
#2:diff + no topol
#3:non-diff + no topol

# location of sim. data
source = ["CF_dataset_non_diff/RC_CFB_eTOe.csv",
	"CF_dataset/RC_CFB_eTOe.csv",
	"CF_dataset_instant_diff/RC_CFB_eTOe.csv"]
# plot labels
labels = ["Non-Diffusive Homeostasis",
	"Diffusive Homeostasis",			
	"Instant Diffusion"]

RC_CFB = []
for k in xrange(3):
	rc_cfb = []
	# open sim. data
	with open(sim_data_base_folder+source[k],"rb") as csvfile:
		reader = csv.reader(csvfile)
		# each row contains data from one simulation run
		for row in reader:
			rc_cfb.append(np.array(row).astype("float"))
	rc_cfb = np.array(rc_cfb).T
	
	# calculate mean and standard error over simulation runs
	rc_cfb_m = rc_cfb.mean(axis=1)
	rc_cfb_err = rc_cfb.std(axis=1)/np.sqrt(rc_cfb.shape[1])
	
	# append to data list
	RC_CFB.append([rc_cfb_m,rc_cfb_err])

# time axis
t = range(RC_CFB[0][0].shape[0])

fig = plt.figure(figsize=(default_fig_width,default_fig_width*0.6))

# plot time course, line width representing standard error
for k in xrange(3):
	plt.fill_between(t,RC_CFB[k][0]-RC_CFB[k][1],RC_CFB[k][0]+RC_CFB[k][1],color=mpl.rcParams['axes.color_cycle'][k])
	plt.plot(t,RC_CFB[k][0],c=mpl.rcParams['axes.color_cycle'][k],label=labels[k],lw=0.5)



plt.xlabel("t [s]")
plt.ylabel("Ratio/Chance of Bidirectional Connections")
plt.xlim([t[0],t[-1]])
plt.ylim([0.,4.])
plt.legend()

#save figure
for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)


plt.show()

#import pdb
#pdb.set_trace()
