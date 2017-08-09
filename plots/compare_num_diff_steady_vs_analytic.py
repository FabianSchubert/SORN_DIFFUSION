import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from custom_modules.plot_setting import *
from scipy.special import kn
from custom_modules.psi_interact import *
import pdb
from tqdm import tqdm

mpl.rcParams['lines.linewidth'] = 1.

#This script compares the numeric steady state solution of a single point source + 2d
#diffusion with the analytic expression described in the thesis

## Simulation scripts required to run before this script:
# None


# where to save figures
savefolder = plots_base_folder
plot_filename = "Psi_Interact_Approximation"

# Diffusion parameters and params. for numeric solution
D= 10000.
lbd = .1
N_grid=201
h=10.
hsqr=h**2
rho = np.zeros((N_grid,N_grid))

# index of center of grid
center = int((N_grid-1.)/2.)

# Kernel for point source influx
I = np.zeros((N_grid,N_grid))
I[center,center] = 1./hsqr

# iteration steps for numeric solution
n_steps = 50000

# time step
dt = 0.001

# recorder to optionally control convergence of peak of distribution
rho_peak_rec = np.ndarray(n_steps)


# Runge-Kutta integration
def step(phi,func,dt):
	phi1=func(phi)
	phi2=func(phi+phi1*dt/2)
	phi3=func(phi+phi2*dt/2)
	phi4=func(phi+phi3*dt)
	return phi+(phi1+2*phi2+2*phi3+phi4)*dt/6
# function to be integrated
def F_diff(Phi):
	return D*lapl(Phi,hsqr,N_grid)-lbd*Phi +  I
# neumann laplace operator
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
# main simulation loop
for k in tqdm(xrange(n_steps)):
	
	#integration step
	rho = step(rho,F_diff,dt)
	# record conc. at source
	rho_peak_rec[k] = rho[center,center]

# x-axis for plotting
x = np.linspace(-h*(N_grid-1.)/2.,h*(N_grid-1.)/2.,N_grid)


fig = plt.figure(figsize=(default_fig_width*0.7,default_fig_width*0.5))

# plot numeric simulation and analytic solution
plt.plot(x,rho[:,center]/rho[:,center].max(),label="Numeric Simulation, $\mathrm{D=\\,10\\, \\mu m^2 / ms}$")
plt.plot(x,psi(np.abs(x),D,lbd,h,10.)/rho[:,center].max(),'--',label="Approximation")
plt.xlabel("x $\mathrm{\\mu m}$")
plt.ylabel("$\mathrm{\\psi/\\psi_{max}}$")
plt.ylim([0.,1.5])
plt.legend()

# save figure
for f_f in file_format:
	plt.savefig(savefolder + plot_filename + f_f)

plt.show()

