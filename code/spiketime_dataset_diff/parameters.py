from brian import *

### Number of simulation runs

N_runs = 10


### Base parameters
run_time = 1500. * second # run time, default 1500. * second
t_switch = 500. * second # time to switch between hom. mechanisms non diff -> diff
t_switch_ext = 3000. * second # time to switch on external input
t_syn_plast_off = 1500. * second # optional removal of synaptic plasticity mechanisms
syn_plast_switch = False # activate or deactivate switching

size_T = 1000 # sheet length, microns
width_T = 200 # growth radius, microns
use_topol = True #Use distance-dependent connection profile?
sparse_eTOe = 0.0 # recurrent excitatory sparseness -> change to .1 , default 0.0
sparse_iTOe = 0.10 # inhibitory to excitatory sparseness
sparse_eTOi = 0.10 # excitatory to inhibitory  sparseness
sparse_iTOi = 0.5 # inhibitory to inhibitory sparseness
N_e = 400 # excitatory population size -> 200 , default 400
N_i = int(0.2*N_e) # inhibitory population size
wi_eTOe = 1.5 * mV # initial e->e weight
wi_eTOi = 1.5 * mV # initial e->i weight
wi_iTOe = -1.5 * mV # initial i->e weight
wi_iTOi = -1.5 * mV # initial i->i weight
delay_eTOe = 1.5 * ms # e->e latency
delay_eTOi = 0.5 * ms # e->i latency
delay_iTOe = 1.0 * ms # i->e latency
delay_iTOi = 1.0 * ms # i->i latency

sigma_noise = sqrt(5.0) * mV # noise amplitude
tau = 20 * ms # membrane time constant
Vr_e = -70 * mV # excitatory reset value
Vr_i = -60 * mV # inhibitory reset value
El = -60 * mV # resting value
Vti = 60 * mV # minus maximum initial threshold voltage
Vtvar = 0 * mV # maximum initial threshold voltage swing
Vvi = 50 * mV # minus maximum initial voltage
Vvar = 20 * mV # maximum initial voltage swing
Vti_i = -58 * mV # inhibitory firing threshold

### STDP parameters
taupre = 15 * ms # pre-before-post STDP time constant
taupost = taupre * 2.0 # post-before-pre STDP time constant
Ap = 15.0 * mV # potentiating STDP learning rate 
Ad = -Ap * 0.5 # depressing STDP learning rate
interaction_stdp = 'nearest'

### Synaptic normalization parameters
total_in_eTOe = 40 * mV # total e->e synaptic input
total_in_iTOe = 120 * mV # total i->e synaptic input #default: 12mV
total_in_eTOi = 60 * mV # total e->i synaptic input
total_in_iTOi = 60 * mV # total i->i synaptic input

### Structural plasticity parameters
sp_initial = 0.0001 * mV # initial weight for newly created synapses
zero_cut = 1e-9 # zero pruning cutoff, volts
sp_rate = 2.3 * N_e # stochastic rate of new synapse production


### Threshold adaption parameters

tau_hip = 2500000. # ms, time constant of threshold adaption
NO_0 = 0.0000274 # NO / (micrometer^2) initial target concentration
ThBoundOffs = 0*mV # optional lower bound for threshold to avoid "bursts" (actual bound is V_r + ThBoundOffs)


### Diffusion parameters and objects

N_grid = 100
h = size_T/N_grid
hsqr = h**2
D = 10. # (micrometer^2)/ms, Sweeney 2015 paper had 12.5 times the amount of neurons -> adapt for scale
lbd = 0.0001 # NO / (ms * micrometer^2) 

### NO synth parameters

Ca_sp = 1. 
tau_ca = 10. # ms, time constant of Calcium concentration
tau_n = 100. # ms, time constant of nNO-Synthesis
n_nos = 3. # Exponent of the Hill Equation in nNOS ("Hill coefficient")
K_nos = 1. # "ligand concentration occupying half of the binding sites"


### Non-Diff IP parameters

h_ip_hz = 3 # target rate, Hz per unit !!ORIGINAL 3!!
eta_ip = 0.1 * mV # IP learning rate

### Ext. input
f_ext = 0. *hertz # external poisson input rate
N_ext = N_e # Size of external input group
ext_weight = total_in_eTOe/N_ext # weight per connection

### STP

tau_d=500*ms #default:500ms
tau_f=2000*ms #default:2000ms
U_stp=.04


### Extra Clocks

slow_time = 1000 * ms # slow clock used for syn. norm., struct. growth & pruning and recording of connections and total NO-field
interm_time = 100 * ms # intermediate clock for recording of a number of state variables
ip_time = 1. * ms # fast clock for accurate calculation of intrinsic plasticity

w_rec_time = 5000 * ms # clock for recording of synaptic connectivity

### Diffusion Boundary Condition

bound_cond = "neumann"

