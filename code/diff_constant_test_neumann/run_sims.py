import os
from parameters import N_runs
from parameters import D_list

for D in D_list:
	if D > 20.:
		ip_time = 0.1
	else:
		ip_time = 1.0
	
	for k in xrange(N_runs):
		os.system("python SORN_diff_constant_test.py " + str(D) + " " + str(ip_time))
