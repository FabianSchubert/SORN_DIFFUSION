import os
from parameters import N_runs

for k in xrange(N_runs):
	os.system("python SORN_cf_non_diff_no_topology.py")
