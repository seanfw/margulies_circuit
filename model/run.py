# %load run.py

from connectivity import get_surface_connectivity, get_longdistance_connectivity
from local_circuit import initialize_local, current_to_frequency, NMDA_deriv, GABA_deriv
from model import get_stim
from delays import get_delays
from initialize_parameters import initialize_pm
import brian2
import numpy as np
from numba import jit

# Set up simulation details
dt= 1 * brian2.ms  # timestep
trial_length= 2500 * brian2.ms # trial length (s)
num_iterations = int(trial_length/dt)
time = np.arange(0, trial_length, dt)
stim_on      = 0.5 * brian2.second
stim_off     = 1.0 * brian2.second

surf_file  = './fsaverage4/lh.pial'
cort_file  = './fsaverage4/lh.cortex.label'
annot_file = './fsaverage4/lh.aparc.a2009s.annot'

surface_connectivity = get_surface_connectivity(surf_file, cort_file)
num_nodes = len(surface_connectivity)

tract_connectivity = get_longdistance_connectivity(surf_file, cort_file, './average_sc.pconn.nii', 'Glasser_NetworkPartition_v9.fsa4.L.label.gii')

tract_delays, max_delay, S_tract_delay = get_delays(surf_file, cort_file)

pl = initialize_local(num_nodes)
pm = initialize_pm(pl, num_nodes, num_iterations, dt)

label_name   = 'S_calcarine'
V1_index     = get_stim(annot_file, label_name, cort_file)
I_ext        = np.zeros((num_iterations,num_nodes,pm['num_pops'])) * brian2.amp
I_ext[int(stim_on/dt):int(stim_off/dt),V1_index,0] = pl['stim_strength']

@jit(nopython=True)
def dot_np(A,B):
    C = np.dot(A,B)
    return C

for i_t in range(1,num_iterations):

    # update noise - dims = num local pops x num areas
    pm['I_noise'] = pm['I_noise'] + -pm['I_noise']*(dt/pl['tau_AMPA']) + pm['noise_rhs'][i_t-1,:,:]

    # long range NMDA (along white matter tracts) with variable delays
    pm['I_longRange_NMDA'][i_t-1,:,0] = pl['g_E_longRange'] * np.sum(tract_connectivity*S_tract_delay,axis=1)

    # mid range NMDA (along surface), with a 7 timestep delay
    pm['I_midRange_NMDA'][i_t-1,:,:] = pl['g_E_midRange'] * dot_np(surface_connectivity,pm['S_NMDA'][i_t-7,:,:])

    # local NMDA
    pm['I_local_NMDA'][i_t-1,:,:] = pm['J_NMDA'].dot(pm['S_NMDA'][i_t-1,:,:].T).T

    # sum up all the local GABA current onto E and I cell somas
    pm['I_local_GABA'][i_t-1,:,:] = pm['J_GABA'].dot(pm['S_GABA'][i_t-1,:,:].T).T

    # Define total input current as sum of local NMDA & GABA inputs, with background, external currents & noise & longRange
    pm['I_total'][i_t-1,:,:] = pm['I_local_NMDA'][i_t-1,:,:] +  pm['I_local_GABA'][i_t-1,:,:] + pm['I_0'] + I_ext[i_t-1,:,:] + pm['I_noise'] + pm['I_midRange_NMDA'][i_t-1,:,:] + pm['I_longRange_NMDA'][i_t-1,:,:]

    # Update the firing rates of the excitatory population
    pm['R'][i_t,:,0] = pm['R'][i_t-1,:,0] + dt*current_to_frequency(pm['I_total'][i_t-1,:,0],'E',pl)/pl['tau_rates']-dt*pm['R'][i_t-1,:,0]/pl['tau_rates']

    # Update the firing rates of the inhibitory population.
    pm['R'][i_t,:,1] = pm['R'][i_t-1,:,1] + dt*current_to_frequency(pm['I_total'][i_t-1,:,1],'I',pl)/pl['tau_rates']-dt*pm['R'][i_t-1,:,1]/pl['tau_rates']

    # Update the NMDA synapses
    pm['S_NMDA'][i_t,:,0] = pm['S_NMDA'][i_t-1,:,0] + dt*NMDA_deriv(pm['S_NMDA'][i_t-1,:,0],pm['R'][i_t,:,0],pl)

    S_tract_delay = pm['S_NMDA'][i_t-tract_delays,range(num_nodes),0]

    # Update the GABA synapses
    pm['S_GABA'][i_t,:,1] = pm['S_GABA'][i_t-1,:,1] + dt*GABA_deriv(pm['S_GABA'][i_t-1,:,1],pm['R'][i_t,:,1],pl)

np.save('R.npy', pm['R'])
