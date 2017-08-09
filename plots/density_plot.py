import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from custom_modules.plot_setting import *

#This script plots an exemplary snapshot of the NO concentration across the tissue
#during a simulation

## Simulation scripts required to run before this script:
# None

# folders of simulation data and for saving figure
folder = sim_data_base_folder + "complete_diff_long/"
savefolder = plots_base_folder + "density_plots/"
plot_filename = "density_plot"

# load NO concentration snapshot
rho = np.array(pickle.load(open(folder+"rho_rec_tot.p","rb")))

plt.figure(figsize=(4.7*0.7,4.7*0.7))

# plot NO distribution
plt.pcolormesh(np.linspace(0,1000,101),np.linspace(0,1000,101),rho[-1,:,:])
plt.xlabel(r'$\mu$m')
plt.ylabel(r'$\mu$m')
plt.axes().set_aspect('equal', 'box')

plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

# save figure
for f_f in file_format:
	plt.savefig(savefolder + plot_filename + f_f)

plt.show()
