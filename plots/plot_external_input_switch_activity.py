import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cPickle as pickle
from custom_modules.frequ_from_spikes import *
from custom_modules.plot_setting import *

mpl.rcParams['lines.linewidth'] = 1.

# This script plots the time course of the population firing rate of 3 excitatory subpopulations upon external Poisson input.

## Simulation scripts required to run before this script:
# -ext_input

# where to find data and where to save figure
folder = sim_data_base_folder + "ext_input_switch/"
savefolder = plots_base_folder
plot_filename = "external_input_switch"

# load spiketimes of subpopulations:
# spt_full: Entire population (400 neurons)
# spt_e_inp_0: 5 Hz Poisson input (100 neurons)
# spt_e_inp_1: 2.5 Hz Poisson input (100 neurons)
spt_full = pickle.load(open(folder+"spiketimes_e.p")).values()
spt_e_inp_0 = pickle.load(open(folder + "spiketimes_e_inp_0.p")).values()
spt_e_inp_1 = pickle.load(open(folder + "spiketimes_e_inp_1.p")).values()

# calculate population firing rates, using time bins of 1s
f_full = frequ_bin_time(spt_full,0.,1500.,1500).mean(axis=1)
f_e_inp_0 = frequ_bin_time(spt_e_inp_0,0.,1500.,1500).mean(axis=1)
f_e_inp_1 = frequ_bin_time(spt_e_inp_1,0.,1500.,1500).mean(axis=1)

# calculate population firing rate of neurons not belonging to either one of the populations with external input
f_e_rest = (f_full*400. - (f_e_inp_0*100. + f_e_inp_1*100.))/200.

# x-axis for plotting
t = np.linspace(0.,1499.,1500)

fig = plt.figure(figsize=(default_fig_width,default_fig_width*0.6))

# plot firing rates
plt.plot(t,f_e_rest,label="0 Hz")
plt.plot(t,f_e_inp_0,label="5 Hz")
plt.plot(t,f_e_inp_1,label="2.5 Hz")
plt.plot(t,f_full,c="k",label="Total Population Mean")
plt.xlim([0.,1500.])
plt.ylim([0.,11.])
plt.xlabel("t [s]")
plt.ylabel("f [Hz]")
plt.legend()

# save figure
for f_f in file_format:
	plt.savefig(savefolder + plot_filename + f_f)

plt.show()
