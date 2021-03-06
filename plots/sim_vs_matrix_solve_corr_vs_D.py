import numpy as np
import matplotlib.pyplot as plt
import json
from custom_modules.plot_setting import *
from custom_modules.frequ_from_spikes import *
from custom_modules.dist_mat import *
from custom_modules.psi_interact import *

# This plots the pearson correlation coefficient between predicted and firing rates and those taken from a full simulation against the diffusion constant used
# in the particular simulation

## Simulation scripts required to run before this script:
# -diff_constant_test_neumann_matrix_solve_correlation


# where to save figure
savefolder = plots_base_folder
plot_filename = "sim_vs_matrix_solve_corr_vs_D"

data = []
diff_const = []
corr_list = []

# open simulation dataset
with open(sim_data_base_folder + "Diff_constant_test_matrix_solve_correlation/data_list.dat","r") as file:
	print("Loading Data...")
	for row in file:
		data.append(json.loads(row))
		# calculate mean frequencies, for 1200-1500s
		try:
			spt = data[-1]['spt']
		except:
			spt = data[-1]['spt_e']
		data[-1]["f_e"] = frequ_vec(spt,1200,1500)
		# calculate dist. matrix
		D = dist_mat_from_pos(np.array(data[-1]["Pos_e"]),2,2,1000.,"neumann")
		# calculate diff. interaction matrix
		Psi = psi(D,data[-1]["D"]*1000.,0.1,10.,10.).sum(axis=0)
		# calculate target concentration for given targe rate of 3 Hz
		target = np.dot(Psi,3.*np.ones(400)).mean()
		# calculate predicted firing rates
		data[-1]["f_e_mat"] = np.linalg.solve(Psi,np.ones(400)*target)
		# calculate corr. coeff. between simulation and prediction
		data[-1]["corr"] = np.corrcoef(data[-1]["f_e_mat"],data[-1]["f_e"])[0,1]
		# group results into sets with the same diff. constant for further calculation of mean and std. error
		if data[-1]["D"] in diff_const:
			corr_list[diff_const.index(data[-1]["D"])].append(data[-1]["corr"])
		else:
			diff_const.append(data[-1]["D"])
			corr_list.append([data[-1]["corr"]])

# number of diffusion constants present in data
n_data = len(diff_const)

corr_mean = np.ndarray((n_data))
corr_err = np.ndarray((n_data))

# sort diff. constants for plotting
ind_sort = np.argsort(diff_const)

# calc. mean and std. error of corr. coeff. for these diff. constants
for k in xrange(n_data):
	corr_mean[k] = np.array(corr_list[k]).mean()
	corr_err[k] = np.array(corr_list[k]).std()/len(corr_list[k])

# rearrange data according to ind_sort
diff_const = np.array(diff_const)[ind_sort]
corr_mean = corr_mean[ind_sort]
corr_err = corr_err[ind_sort]


fig = plt.figure(figsize=(default_fig_width*0.7,default_fig_width*0.7*0.6))

# plot results, line width representing std. error
plt.fill_between(diff_const,corr_mean-corr_err,corr_mean+corr_err,linewidth=0.)
plt.xlabel("D $\\mathrm{[\\mu m^2 /ms]}$")
plt.ylabel("Corr. Coeff.")

# save figure
for f_f in file_format:
	plt.savefig(savefolder + plot_filename + f_f)

plt.show()

