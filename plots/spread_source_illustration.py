import numpy as np
import matplotlib.pyplot as plt
import pdb
from custom_modules.plot_setting import *

# This script plots an illustration of the circular nNOS kernel used in one variant of diffusive homeostasis in the thesis

## Simulation scripts required to run before this script:
# None

# where to save figure
savefolder = plots_base_folder
plot_filename = "spread_source_illustration"

# numeric grid resolution
N_grid = 100
# Number of sources/neurons
N_e = 400
# sheet size
L=1000.
# grid resolution
h=L/N_grid
hsqr = h**2

# radius of the circular source
R_source = 50.

# "radius" measured in grid points
n_max_source = int(R_source/h)

kernel_ind_source = [[],[]]

# collect grid coordinates included into circular source
for k in xrange(-n_max_source,n_max_source+1):
	for l in xrange(-n_max_source,n_max_source+1):
		if k**2 + l**2 <= n_max_source**2:
			kernel_ind_source[0].append(k)
			kernel_ind_source[1].append(l)

kernel_ind_source = np.array(kernel_ind_source)
# calculate total area of kernel
area_kernel = 1.*kernel_ind_source.shape[1]*hsqr

ind_e = np.ndarray((N_e,2))

##avoid placing two exc. neurons at the same position on the grid
for k in xrange(N_e):
	
	while True:
		
		rand_ind = (np.random.rand(2)*N_grid).astype("int")
		
		passed = True
		
		for l in xrange(k):
			
			if ind_e[l,0] == rand_ind[0] and ind_e[l,1] == rand_ind[1]:
				passed = False
		if passed:
			ind_e[k,:] = rand_ind
			break

ind_e = ind_e.astype("int")

# one illustration showing a superposition of all sources...
rho = np.zeros((N_grid,N_grid))
# and one with only a single source
rho_single = np.zeros((N_grid,N_grid))

# shift indices of kernel to the desired location on the grid for the single source illustration
kernel_temp = np.array([kernel_ind_source[0,:]+ind_e[0,0],kernel_ind_source[1,:]+ind_e[0,1]])
kernel_temp = np.mod(kernel_temp,N_grid)
rho_single[kernel_temp[0,:],kernel_temp[1,:]]+=1./area_kernel

# the same for all sources
for k in xrange(N_e):
		kernel_temp = np.array([kernel_ind_source[0,:]+ind_e[k,0],kernel_ind_source[1,:]+ind_e[k,1]])
		kernel_temp = np.mod(kernel_temp,N_grid)
		rho[kernel_temp[0,:],kernel_temp[1,:]]+=1./area_kernel

axis = np.linspace(0.,L,N_grid+1)

fig,ax = plt.subplots(1,2,figsize=(default_fig_width,default_fig_width*0.5))

# plot heatmaps of the resulting grids
ax[0].pcolormesh(axis,axis,rho)
ax[1].pcolormesh(axis,axis,rho_single)

ax[0].set_xlabel("$\\mathrm{\\mu m}$")
ax[0].set_ylabel("$\\mathrm{\\mu m}$")
ax[1].set_xlabel("$\\mathrm{\\mu m}$")
ax[1].set_ylabel("$\\mathrm{\\mu m}$")

ax[0].set_title("A",loc="left")
ax[1].set_title("B",loc="left")

# save figure
for f_f in file_format:
	plt.savefig(savefolder + plot_filename + f_f)

plt.show()
