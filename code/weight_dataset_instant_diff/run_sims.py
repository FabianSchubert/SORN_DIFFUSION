import os
from parameters import N_runs

for k in xrange(N_runs):
	os.system("python SORN_weight_dataset_instant_diff.py")
