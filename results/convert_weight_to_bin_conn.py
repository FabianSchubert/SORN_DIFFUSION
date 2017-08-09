import numpy as np
from custom_modules.plot_setting import *

# Convert float array of weights into binary connectivity matrix to reduce space

fold_diff = "complete_diff_long/"
fold_non_diff = "complete_non_diff_long/"
fold_instant_diff = "complete_instant_diff_long/"

W = np.load(sim_data_base_folder + fold_diff + "W_eTOe_record.npy")

np.save(sim_data_base_folder + fold_diff + "W_eTOe_record_bin.npy",W!=0)

W = np.load(sim_data_base_folder + fold_non_diff + "W_eTOe_record.npy")

np.save(sim_data_base_folder + fold_non_diff + "W_eTOe_record_bin.npy",W!=0)

W = np.load(sim_data_base_folder + fold_instant_diff + "W_eTOe_record.npy")

np.save(sim_data_base_folder + fold_instant_diff + "W_eTOe_record_bin.npy",W!=0)
