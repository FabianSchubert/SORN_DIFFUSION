from __future__ import division
from parameters import *   # import parameter file

from brian import * # brian module
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.mlab as mlab
import scipy.optimize as opt
import scipy.cluster.hierarchy as ch
import numpy.random as npr
import random
import time
import pdb
import sys
import os
import cPickle as pickle
from brian.globalprefs import *
set_global_preferences(usecodegen=True)
import json

from custom_modules.plot_setting import sim_data_base_folder

# set diffusion constant according to command line argument
D = float(sys.argv[1])

# set ip time constant according to command line argument
ip_time = float(sys.argv[2]) * ms


### Helper objects

### postsynaptic normalization
def synnorm(connection_target,total_in):
	for i in xrange(shape(connection_target.W)[1]):
		sum_in = connection_target.W[:,i].sum()
		if not sum_in == 0:
			connection_target.W[:,i] = total_in * connection_target.W.todense()[:,i] / sum_in

### Gaussian function for distance dependent connection probabilities			
def gaussian(x,u,s):
	g = (2/sqrt(2*pi*s*s))*exp(-(x-u)*(x-u)/(2*s*s))
	return g

### Initialize numeric diffusion grid
rho = NO_0*np.ones((N_grid,N_grid))
### Dynamic variables for IP and the NO synthesis pathway
NOS_Th = np.zeros((N_e,3))


### Runge-Kutta integration step
def step(phi,func,dt):
	phi1=func(phi)
	phi2=func(phi+phi1*dt/2)
	phi3=func(phi+phi2*dt/2)
	phi4=func(phi+phi3*dt)
	return phi+(phi1+2*phi2+2*phi3+phi4)*dt/6
	

	
### Diffusion Diff equ. operator	
def F_diff(Phi):
	return D*lapl(Phi,hsqr,N_grid)-lbd*Phi


### NOS and Threshold dynamics adaption Diff. equation operator
### index: 0: Ca , 1: nNOS , 2: LIF-Thresh
def F_nos_th(Phi):
	result = np.zeros((N_e,3))
	result[:,0]=-Phi[:,0]/tau_ca
	result[:,1]=(1/tau_n)*(Phi[:,0]**n_nos/(Phi[:,0]**n_nos+K_nos**n_nos)-Phi[:,1])
	result[:,2]=(1/tau_hip)*(rho[ind_e[:,0],ind_e[:,1]]-NO_0)/NO_0#rho[ind_e[:,0],ind_e[:,1]]
	return result


if bound_cond == "periodic":
	## Laplace operator, periodic bounds
	
	def lapl(Phi,hsqr,N):
		l=np.zeros((N,N))


		l[1:-1,1:-1]=(Phi[0:-2,1:-1]+Phi[2:,1:-1]+Phi[1:-1,0:-2]+Phi[1:-1,2:])/hsqr

		l[0,0]=(Phi[-1,0]+Phi[1,0]+Phi[0,-1]+Phi[0,1])/hsqr
		l[-1,0]=(Phi[-2,0]+Phi[0,0]+Phi[-1,-1]+Phi[-1,1])/hsqr
		l[-1,-1]=(Phi[-2,-1]+Phi[0,-1]+Phi[-1,-2]+Phi[-1,0])/hsqr
		l[0,-1]=(Phi[-1,-1]+Phi[1,-1]+Phi[0,-2]+Phi[0,0])/hsqr

		l[1:-1,0]=(Phi[0:-2,0]+Phi[2:,0]+Phi[1:-1,-1]+Phi[1:-1,1])/hsqr
		l[1:-1,-1]=(Phi[0:-2,-1]+Phi[2:,-1]+Phi[1:-1,-2]+Phi[1:-1,0])/hsqr
		l[0,1:-1]=(Phi[-1,1:-1]+Phi[1,1:-1]+Phi[0,0:-2]+Phi[0,2:])/hsqr
		l[-1,1:-1]=(Phi[-2,1:-1]+Phi[0,1:-1]+Phi[-1,0:-2]+Phi[-1,2:])/hsqr
		l=l-4*Phi/hsqr

		return l
		
elif bound_cond == "neumann":
	## Laplace operator, von-neumann bounds
	def lapl(Phi,hsqr,N):
		l=np.zeros((N,N))


		l[1:-1,1:-1]=(Phi[0:-2,1:-1]+Phi[2:,1:-1]+Phi[1:-1,0:-2]+Phi[1:-1,2:])/hsqr

		l[0,0]=(2*Phi[1,0]+2*Phi[0,1])/hsqr
		l[-1,0]=(2*Phi[-2,0]+2*Phi[-1,1])/hsqr
		l[-1,-1]=(2*Phi[-2,-1]+2*Phi[-1,-2])/hsqr
		l[0,-1]=(2*Phi[1,-1]+2*Phi[0,-2])/hsqr

		l[1:-1,0]=(Phi[0:-2,0]+Phi[2:,0]+2*Phi[1:-1,1])/hsqr
		l[1:-1,-1]=(Phi[0:-2,-1]+Phi[2:,-1]+2*Phi[1:-1,-2])/hsqr
		l[0,1:-1]=(2*Phi[1,1:-1]+Phi[0,0:-2]+Phi[0,2:])/hsqr
		l[-1,1:-1]=(2*Phi[-2,1:-1]+Phi[-1,0:-2]+Phi[-1,2:])/hsqr
		l=l-4*Phi/hsqr

		return l
		
		
elif bound_cond == "dirichlet":	
	## Laplace operator, dirichlet bounds to NO_bound

	NO_bound=NO_0

	def lapl(Phi,hsqr,N):
		l=np.zeros((N,N))


		l[1:-1,1:-1]=(Phi[0:-2,1:-1]+Phi[2:,1:-1]+Phi[1:-1,0:-2]+Phi[1:-1,2:])/hsqr

		l[0,0]=(Phi[1,0]+Phi[0,1]+2*NO_bound)/hsqr
		l[-1,0]=(Phi[-2,0]+Phi[-1,1]+2*NO_bound)/hsqr
		l[-1,-1]=(Phi[-2,-1]+Phi[-1,-2]+2*NO_bound)/hsqr
		l[0,-1]=(Phi[1,-1]+Phi[0,-2]+2*NO_bound)/hsqr

		l[1:-1,0]=(Phi[0:-2,0]+Phi[2:,0]+Phi[1:-1,1]+NO_bound)/hsqr
		l[1:-1,-1]=(Phi[0:-2,-1]+Phi[2:,-1]+Phi[1:-1,-2]+NO_bound)/hsqr
		l[0,1:-1]=(Phi[1,1:-1]+Phi[0,0:-2]+Phi[0,2:]+NO_bound)/hsqr
		l[-1,1:-1]=(Phi[-2,1:-1]+Phi[-1,0:-2]+Phi[-1,2:]+NO_bound)/hsqr
		l=l-4*Phi/hsqr

		return l
else:
	print("wrong bound. cond. argument!")
	sys.exit(0)
	

### LIF neuron with adjustable threshold and O-U noise     
noisylif = Equations('''
	  dV / dt =  - (V - El) / tau + sigma_noise * xi / (tau **.5): volt
      Vt : volt
      ''')

### Neuron Groups
G_e = NeuronGroup(N = N_e, model = noisylif, threshold ='V > Vt', reset=Vr_e)  # excitatory group
G_i = NeuronGroup(N = N_i, model = noisylif, threshold ='V > Vt', reset=Vr_i)  # inhibitory group

### Topology

## Random exc. positions

ind_e = np.ndarray((N_e,2))


##avoid placing two exc. neurons at the same position on the grid
for k in xrange(N_e):
	
	while True:
		
		rand_ind = (rand(2)*N_grid).astype("int")
		
		passed = True
		
		for l in xrange(k):
			
			if ind_e[l,0] == rand_ind[0] and ind_e[l,1] == rand_ind[1]:
				passed = False
		if passed:
			ind_e[k,:] = rand_ind
			break

ind_e = ind_e.astype("int")

ind_i = (rand(N_i,2)*N_grid).astype("int")
X_e = h * (ind_e.astype("float")+.5) # excitatory neuron positions
X_i = h * (ind_i.astype("float")+.5) # inhibitory neuron positions


### Separation and probability generation
D_eTOe = ndarray(shape=(N_e,N_e)) # e->e separation
for i in xrange(N_e):
	for j in xrange(N_e):
		Dx = X_e[i,0]-X_e[j,0]
		Dy = X_e[i,1]-X_e[j,1]
		D_eTOe[i,j] = sqrt(Dx**2 + Dy**2)

P_eTOe = ndarray(shape=(N_e,N_e)) # e->e probability
for i in xrange(N_e):
	for j in xrange(N_e):
		P_eTOe[i,j] = gaussian(D_eTOe[i,j],0,width_T)
	P_eTOe[i,i] = 0  # prevent self-connections
	
D_eTOi = ndarray(shape=(N_e,N_i)) # e->i separation
for i in xrange(N_e):
	for j in xrange(N_i):
		Dx = X_e[i,0]-X_i[j,0]
		Dy = X_e[i,1]-X_i[j,1]
		D_eTOi[i,j] = sqrt(Dx**2 + Dy**2)
		
P_eTOi = ndarray(shape=(N_e,N_i)) # e->i probability
for i in xrange(N_e):
	for j in xrange(N_i):
		P_eTOi[i,j] = gaussian(D_eTOi[i,j],0,width_T)
			
D_iTOe = ndarray(shape=(N_i,N_e)) #i->e separation
for i in xrange(N_i):
	for j in xrange(N_e):
		Dx = X_i[i,0]-X_e[j,0]
		Dy = X_i[i,1]-X_e[j,1]
		D_iTOe[i,j] = sqrt(Dx**2 + Dy**2)

P_iTOe = ndarray(shape=(N_i,N_e)) #i->e probability
for i in xrange(N_i):
	for j in xrange(N_e):
		P_iTOe[i,j] = gaussian(D_iTOe[i,j],0,width_T)
	
D_iTOi = ndarray(shape=(N_i,N_i)) # i->i separation
for i in xrange(N_i):
	for j in xrange(N_i):
		Dx = X_i[i,0]-X_i[j,0]
		Dy = X_i[i,1]-X_i[j,1]
		D_iTOi[i,j] = sqrt(Dx**2 + Dy**2)

P_iTOi = ndarray(shape=(N_i,N_i)) # i->i probability
for i in xrange(N_i):
	for j in xrange(N_i):
		P_iTOi[i,j] = gaussian(D_iTOi[i,j],0,width_T)
	P_iTOi[i,i] = 0 # prevent self-connections

print "Separation matrices initialized."			


### Randomize initial voltages
G_e.V = -(Vvi + rand(N_e) * Vvar) # starting membrane potential
G_i.V = -(Vvi + rand(N_i) * Vvar) # starting membrane potential 

### Randomize initial thresholds
G_e.Vt = -(Vti + rand(N_e) * Vtvar)  # starting threshold

### Set dynamic excitatory threshold variable to the value of G_e.Vt
NOS_Th[:,2] = G_e.Vt

### Set inhibitory threshold
G_i.Vt = -58 * mV

### Connections
C_eTOe = Connection(G_e, G_e, 'V', delay=delay_eTOe, structure='dynamic') # recurrent excitatory connection
C_iTOi = Connection(G_i, G_i, 'V', delay=delay_iTOi, structure='dynamic') # inhibitory -> inhibitory connection
C_iTOe = Connection(G_i, G_e, 'V', delay=delay_iTOe, structure='dynamic') # inhibitory -> excitatory connection
C_eTOi = Connection(G_e, G_i, 'V', delay=delay_eTOi, structure='dynamic') # excitatory -> inhibitory connection



n_new = int(round(N_e*N_e*sparse_eTOe)) # determine number of connections to generate
for i in xrange(n_new): # throw randoms to populate connections; better method to come
	addition_allowed = False
	new_i = randint(0, high=N_e)
	new_j = randint(0, high=N_e)
	while not addition_allowed:
		new_i = randint(0, high=N_e)
		new_j = randint(0, high=N_e)
		if rand() < P_eTOe[new_i,new_j]:
			if not new_i == new_j and C_eTOe[new_i,new_j] == 0:
				addition_allowed = True
	C_eTOe[new_i, new_j] = wi_eTOe
synnorm(C_eTOe, total_in_eTOe) # normalize synapses


n_new = int(round(N_e*N_i*sparse_eTOi)) # determine number of connections to generate
for i in xrange(n_new): # throw randoms to populate connections; better method to come
	addition_allowed = False
	new_i = randint(0, high=N_e)
	new_j = randint(0, high=N_i)
	while not addition_allowed:
		new_i = randint(0, high=N_e)
		new_j = randint(0, high=N_i)
		if rand() < P_eTOi[new_i,new_j]:
			if C_eTOi[new_i,new_j] == 0:
				addition_allowed = True
	C_eTOi[new_i, new_j] = wi_eTOi
synnorm(C_eTOi, total_in_eTOi) # normalize synapses


n_new = int(round(N_i*N_e*sparse_iTOe)) # determine number of connections to generate
for i in xrange(n_new): # throw randoms to populate connections; better method to come
	addition_allowed = False
	new_i = randint(0, high=N_i)
	new_j = randint(0, high=N_e)
	while not addition_allowed:
		new_i = randint(0, high=N_i)
		new_j = randint(0, high=N_e)
		if rand() < P_iTOe[new_i,new_j]:
			if C_iTOe[new_i,new_j] == 0:
				addition_allowed = True
	C_iTOe[new_i, new_j] = wi_iTOe
synnorm(C_iTOe, -1 * total_in_iTOe)


n_new = int(round(N_i*N_i*sparse_iTOi)) # determine number of connections to generate
for i in xrange(n_new): # throw randoms to populate connections; better method to come
	addition_allowed = False
	new_i = randint(0, high=N_i)
	new_j = randint(0, high=N_i)
	while not addition_allowed:
		new_i = randint(0, high=N_i)
		new_j = randint(0, high=N_i)
		if rand() < P_iTOi[new_i,new_j]:
			if not new_i == new_j and C_iTOi[new_i,new_j] == 0:
				addition_allowed = True
	C_iTOi[new_i, new_j] = wi_iTOi
synnorm(C_iTOi, -1 * total_in_iTOi)



### Ext. Poisson Input Group to G_e

G_ext = PoissonInput(G_e,N_ext, 0.*hertz, ext_weight, 'V') ### initialize with zero frequency



print "Connections initialized."			


### Counters
Counter_e = SpikeCounter(G_e)  # excitatory spike counter, for IP
Counter_i = SpikeCounter(G_i)  # inhibitory spike counter

### Exponential STDP
stdp_eTOe=ExponentialSTDP(C_eTOe, taupre, taupost, Ap, Ad, interactions='nearest',
						wmin=0, wmax=total_in_eTOe, update='additive')

### STP
stp_eTOe = STP(C_eTOe,taud=tau_d,tauf=tau_f,U=U_stp)

### Extra clocks
slow_clock = Clock(slow_time, makedefaultclock=False)

w_rec_clock = Clock(w_rec_time, makedefaultclock=False)

interm_clock = Clock(interm_time, makedefaultclock=False)

ip_clock = Clock(ip_time, makedefaultclock=False)
dt_ip=float(ip_clock.dt/ms)



### Synaptic normalization - requires different optimizations online
@network_operation(when='end', clock=slow_clock)
#@network_operation(when='end', clock=defaultclock)
def synnorm_eTOe():
	for i in xrange(N_e):
		sum_in = C_eTOe.W[:,i].sum()
		if not sum_in == 0:
			C_eTOe.W[:,i] = total_in_eTOe * C_eTOe.W[:,i] / sum_in
			

### Initialize lists for recording
Ca_rec = [] # Intracellular Calcium - population mean
Ca_rec_tot = [] # Individual intracellular Calcium
nNOS_rec = [] # NO synthesis rate - population mean
nNOS_rec_tot = [] # Individual NO synthesis rate
rho_rec = [] # Mean NO concentration at excitatory neuronal sites
rho_rec_N_e = [] #  NO concentration at individual neuronal sites

### Misc. Recorders
@network_operation(when='end', clock = interm_clock)
def glob_recorders():
	global rho_rec
	global Ca_rec
	global Ca_rec_tot
	global nNOS_rec
	global nNOS_rec_tot
	global tau_hip_rec
	global rho_rec_N_e
	
	Ca_rec.append(NOS_Th[:,0].sum()/N_e)
	Ca_rec_tot.append(NOS_Th[:,0])
	nNOS_rec.append(NOS_Th[:,1].sum()/N_e)
	nNOS_rec_tot.append(NOS_Th[:,1])
	rho_rec.append(rho[ind_e[:,0],ind_e[:,1]].sum()/N_e)
	rho_rec_N_e.append(rho[ind_e[:,0],ind_e[:,1]])
	
	
### Initialize List for recording of complete NO field - caution, large file sizes
rho_rec_tot=[]

## Record NO field
@network_operation(when='end', clock = w_rec_clock)
def NO_rec():
	
	global rho_rec_tot
	rho_rec_tot.append(rho)
	

### IP parameters

#h_ip_hz = clip(npr.lognormal(3.0,size=N_e),0,20)
h_ip = h_ip_hz / (1 * second / ip_clock.dt) # target rate, mean excitatory spikes per timestep

# Flags for switching of IP mechanisms and external input
passed_t_switch = False
passed_t_switch_ext = False


### Intrinsic Plasticity / Diff. Hom.
@network_operation(when='end', clock=ip_clock)
def IP():
	### Diffusive homeostasis variables
	global rho
	global NOS_Th
	global NO_0
	global tau_hip
	global passed_t_switch
	global passed_t_switch_ext
	
	
	### Simple forward Euler for NO inflow
	rho[ind_e[:,0],ind_e[:,1]]+=(NOS_Th[:,1]/1000.)*dt_ip/hsqr # divide by 1000 to get from production/s -> production/ms
        ### Runge-Kutta for the Rest
        rho=step(rho,F_diff,dt_ip)
	
	
	### NOS and Threshold
	
	NOS_Th[:,0]+=Counter_e.count*Ca_sp
	NOS_Th=step(NOS_Th,F_nos_th,dt_ip)
	Thsmallerbool=(1*(NOS_Th[:,2] <= (Vr_e+ThBoundOffs)))
	NOS_Th[:,2] = (Vr_e+ThBoundOffs) * Thsmallerbool + NOS_Th[:,2] * (1-Thsmallerbool)
	
		
	# Switch on external input at t_switch_ext seconds
	
	if defaultclock.t >= t_switch_ext:
		
		if not(passed_t_switch_ext):
			
			passed_t_switch_ext = True
			
			G_ext.rate = f_ext
				
	# Switch between IP mechanisms at t_switch seconds
	
	if defaultclock.t >= t_switch:
				
		G_e.Vt = NOS_Th[:,2] # update excitatory thresholds according to the dynamic variable
		
			
	else:
		G_e.Vt += eta_ip * (Counter_e.count - h_ip)  # non-diffusive IP
		NOS_Th[:,2]=G_e.Vt
		
		NO_0+=dt_ip*(-NO_0+rho[ind_e[:,0],ind_e[:,1]].mean())/(2000.)  # adapt target NO concentration during non-diff. IP via a linear filter
	
	
	Counter_e.count[:] = 0 # Reset spike counter used for IP

### Structural Pruning
#@network_operation(when='end', clock=defaultclock)
@network_operation(slow_clock)
def struct_prune():
	global count_pruned
	for i in xrange(N_e):
		for j in C_eTOe.W.rowj[i]:
			if C_eTOe[i,j] < zero_cut:
				C_eTOe.W.remove(i,j)

### Structural Growth
@network_operation(when='end', clock=slow_clock)
def struct_new_synapses():
	global count_added
	n_new_float = np.random.normal(sp_rate*slow_time,sqrt(sp_rate*slow_time))
	n_new = int(round(n_new_float)) # number to insert
	for i in xrange(n_new): # throw randoms to populate connections; better method to come
		addition_allowed = False
		new_i = randint(0, high=N_e)
		new_j = randint(0, high=N_e)
		while not addition_allowed:
			new_i = randint(0, high=N_e)
			new_j = randint(0, high=N_e)
			if not(use_topol) or (rand() < P_eTOe[new_i,new_j]): 
				if not new_i == new_j and C_eTOe[new_i,new_j] == 0:
					addition_allowed = True
		C_eTOe.W.insert(new_i, new_j, sp_initial)
	synnorm_eTOe() # normalize synapses


### Weight recorder
total_time = run_time
W_eTOe = np.zeros((int(floor((total_time)/w_rec_time)),N_e,N_e))
times_W = []
@network_operation(when='end', clock=w_rec_clock)
def record_W_eTOe():
	W_eTOe[int(floor(w_rec_clock.t/w_rec_time))] = C_eTOe.W.todense()
	times_W.append(w_rec_clock.t)
		
	
### Connection fraction recorders
CF_eTOe = []  # connection fraction
CFB_eTOe = []  # bidirectional connection fraction
RC_CFB_eTOe = []  # ratio over chance of bidirectional connection fraction
tot_con_eTOe = N_e * N_e
@network_operation(when='end', clock=slow_clock)
def record_CF_eTOe():
	count_con = sum(C_eTOe.W.todense() > 0)
	count_con_bi = sum((C_eTOe.W.todense() * C_eTOe.W.todense().transpose()) > 0)
	CF_eTOe.append(count_con/tot_con_eTOe)
	CFB_eTOe.append(count_con_bi/tot_con_eTOe)
	RC_CFB_eTOe.append((count_con_bi/tot_con_eTOe)/((count_con/tot_con_eTOe)**2))


### not setting clock to defaultclock for the Monitors will use slow_clock for some reason.
MS_e = SpikeMonitor(G_e)
MS_i = SpikeMonitor(G_i)

### The network object
SORN = Network(G_e, G_i,
		G_ext,
              C_eTOe, C_eTOi, C_iTOe, C_iTOi,
              Counter_e, Counter_i,
              stdp_eTOe, 
              synnorm_eTOe,
	      IP,
              stp_eTOe,
              struct_prune, 
              struct_new_synapses,
              MS_e,
              MS_i
              )


print "Running."
SORN.run(run_time, report='text')
print "Done."


print "Saving sim. data ..."

spt_e = []
spt_i = []

for k in xrange(N_e):
	spt_e.append(MS_e.spiketimes[k].tolist())
	
for k in xrange(N_i):
	spt_i.append(MS_i.spiketimes[k].tolist())

## Dicitionary for data to be saved
save_data = {"D":D,
	"spt_e":spt_e,
	"spt_i":spt_i,
	"Pos_e":X_e.tolist(),
	"Pos_i":X_i.tolist()}

path = sim_data_base_folder + "Diff_constant_test_matrix_solve_correlation/"  #set path for saving sim.

if not os.path.isdir(path):   # create if not existent
	os.makedirs(path)

# save data with json
with open(path + "data_list.dat","a") as datafile:
	out = json.dumps(save_data)
	datafile.write(out + "\n")


