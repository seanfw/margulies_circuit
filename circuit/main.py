#!/bin/python3

import brian2
import numpy as np
import configparser
import argparse

from circuit import *
from connectivity import get_surface_connectivity, get_longdistance_connectivity
from local_circuit import initialize_local, current_to_frequency, NMDA_deriv, GABA_deriv
from model import get_stim
from delays import get_delays
from initialize_parameters import initialize_pm
from surfdist_functions import *
from speedups import mk_tr_delay

# Set up simulation details
parser = argparse.ArgumentParser()
parser.add_argument('-c', metavar='in-file', type=argparse.FileType('r'))
args = parser.parse_args()
config = configparser.ConfigParser()
if args.c:
    config.read(args.c.name)
else:
    config.read('./config.ini')
c = config['SETUP']

# setup connectivity
surface_connectivity = get_surface_connectivity(c.get('surf_file'), c.get('cort_file'))
tract_connectivity   = get_longdistance_connectivity(c.get('surf_file'),
                                                     c.get('cort_file'),
                                                     './circuit/data/average_sc.pconn.nii',
                                                     './circuit/data/Glasser_NetworkPartition_v9.fsa4.L.label.gii')

# setup delays
tract_delays, max_delay, S_tract_delay = get_delays(c.get('surf_file'), c.get('cort_file'))

# setup circuit and model parameters
num_nodes      = len(surface_connectivity)
dt             = float(c.get('dt')) * brian2.ms  # timestep
num_iterations = int((float(c.get('trial_length')) * brian2.ms) / dt)
pl             = initialize_local(num_nodes)
pm             = initialize_pm(pl, num_nodes, num_iterations, dt)

# setup stim:
stim_on        = float(c.get('stim_on')) * brian2.second
stim_off       = float(c.get('stim_off')) * brian2.second
V1_index       = get_stim(c.get('annot_file'), c.get('label_name'), c.get('cort_file'))
I_ext          = np.zeros((num_iterations,num_nodes,pm['num_pops'])) * brian2.amp
I_ext[int(stim_on/dt):int(stim_off/dt),V1_index,0] = pl['stim_strength']

# run model
#@profile
def run_model(i_t, pm, pl, S_tract_delay, tract_connectivity, surface_connectivity, tract_delays, I_ext, dt, num_nodes):

    # update noise - dims = num local pops x num areas
    pm['I_noise'] = pm['I_noise'] + -pm['I_noise'] * (dt/pl['tau_AMPA']) + pm['noise_rhs'][i_t-1,:,:]

    # local NMDA
    pm['I_local_NMDA'][i_t-1,:,:] = pm['J_NMDA'].dot(pm['S_NMDA'][i_t-1,:,:].T).T

    # sum up all the local GABA current onto E and I cell somas
    pm['I_local_GABA'][i_t-1,:,:] = pm['J_GABA'].dot(pm['S_GABA'][i_t-1,:,:].T).T

    # long range NMDA (along white matter tracts) with variable delays
    if c.get('long_range') == 'True':
        d_tract = np.einsum('ij,ij->i',tract_connectivity,S_tract_delay)
        pm['I_longRange_NMDA'][i_t-1,:,0] = pl['g_E_longRange'] * d_tract
        #pm['I_longRange_NMDA'][i_t-1,:,0] = pl['g_E_longRange'] * np.sum(np.multiply,tract_connectivity,S_tract_delay),axis=1)

    # mid range NMDA (along surface), with a 7 timestep delay
    if c.get('mid_range') == 'True':
        pm['I_midRange_NMDA'][i_t-1,:,:] = pl['g_E_midRange'] * surface_connectivity.dot(pm['S_NMDA'][i_t-pl['midrange_delay'],:,:])

    # Define total input current as sum of local NMDA & GABA inputs, with background, external currents & noise & longRange
    if c.get('long_range')  == 'True' and c.get('mid_range') == 'True':
        pm['I_total'][i_t-1,:,:] = pm['I_local_NMDA'][i_t-1,:,:] +  pm['I_local_GABA'][i_t-1,:,:] + pm['I_0'] + I_ext[i_t-1,:,:] + pm['I_noise'] + pm['I_midRange_NMDA'][i_t-1,:,:] + pm['I_longRange_NMDA'][i_t-1,:,:]
    elif c.get('long_range') == 'True':
        pm['I_total'][i_t-1,:,:] = pm['I_local_NMDA'][i_t-1,:,:] +  pm['I_local_GABA'][i_t-1,:,:] + pm['I_0'] + I_ext[i_t-1,:,:] + pm['I_noise'] + pm['I_longRange_NMDA'][i_t-1,:,:]
    elif c.get('mid_range') == 'True':
        pm['I_total'][i_t-1,:,:] = pm['I_local_NMDA'][i_t-1,:,:] +  pm['I_local_GABA'][i_t-1,:,:] + pm['I_0'] + I_ext[i_t-1,:,:] + pm['I_noise'] + pm['I_midRange_NMDA'][i_t-1,:,:]
    else:
        pm['I_total'][i_t-1,:,:] = pm['I_local_NMDA'][i_t-1,:,:] +  pm['I_local_GABA'][i_t-1,:,:] + pm['I_0'] + I_ext[i_t-1,:,:] + pm['I_noise']

    # Update the firing rates of the excitatory population
    pm['R'][i_t,:,0] = pm['R'][i_t-1,:,0] + dt * current_to_frequency(pm['I_total'][i_t-1,:,0],'E',pl) / pl['tau_rates'] - dt * pm['R'][i_t-1,:,0] / pl['tau_rates']

    # Update the firing rates of the inhibitory population.
    pm['R'][i_t,:,1] = pm['R'][i_t-1,:,1] + dt * current_to_frequency(pm['I_total'][i_t-1,:,1],'I',pl) / pl['tau_rates']-dt * pm['R'][i_t-1,:,1] / pl['tau_rates']

    # Update the NMDA synapses
    pm['S_NMDA'][i_t,:,0] = pm['S_NMDA'][i_t-1,:,0] + dt * NMDA_deriv(pm['S_NMDA'][i_t-1,:,0],pm['R'][i_t,:,0],pl)

    if c.get('long_range') == 'True':
        snmda = np.asarray(pm['S_NMDA'][:,:,0]).squeeze()
        S_tract_delay = mk_tr_delay(i_t-tract_delays, S_tract_delay, snmda)
        #S_tract_delay = pm['S_NMDA'][i_t-tract_delays,range(num_nodes),0]

    # Update the GABA synapses
    pm['S_GABA'][i_t,:,1] = pm['S_GABA'][i_t-1,:,1] + dt * GABA_deriv(pm['S_GABA'][i_t-1,:,1],pm['R'][i_t,:,1],pl)

    return pm, pl, S_tract_delay

for i_t1 in range(1,num_iterations):
    pm, pl, S_tract_delay = run_model(i_t1, pm, pl, S_tract_delay, tract_connectivity, surface_connectivity, tract_delays, I_ext, dt, num_nodes)

# save output
if c.get('save_output') == 'True':
    np.save('./circuit/results/R.npy', pm['R'])

if c.get('viz_ts') == 'True':
    from viz_results import plot_timeseries
    plot_timeseries(pm['R'])

if c.get('viz_surf') == 'True':
    from viz_results import plot_surfaces
    plot_surfaces(pm['R'])
