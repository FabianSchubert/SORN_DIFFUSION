import numpy as np

def frequ_vec(spiketimes,t_start,t_end):
	
	t_start = float(t_start)
	t_end = float(t_end)
	
	n = len(spiketimes)
	
	frequs = np.ndarray(n)
	
	for k in xrange(n):
		
		spkvec = np.array(spiketimes[k])
		
		frequs[k] = 1.*((spkvec >= 1.*t_start)*(spkvec < 1.*t_end)).sum()/(t_end-t_start)
	
	return frequs

def frequ_bin_time(spiketimes,t_start,t_end,nbins):
	
	t_start = float(t_start)
	t_end = float(t_end)
	nbins = int(nbins)
	
	n = len(spiketimes)
	
	frequs = np.ndarray((nbins,n))
	
	for k in xrange(n):
		
		spkvec = np.array(spiketimes[k])
		
		h = np.histogram(spkvec,bins=np.linspace(t_start-(t_end-t_start)/(nbins-1),t_end,nbins+1))
		
		frequs[:,k] = h[0]*nbins/(t_end-t_start+(t_end-t_start)/(nbins-1))
	
	return frequs
