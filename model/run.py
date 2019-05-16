#!/bin/python3

from connectivity import get_surface_connectivity, get_longdistance_connectivity
from local_circuit import initialize_local, current_to_frequency, NMDA_deriv, GABA_deriv
from model import get_stim
from delays import get_delays
from initialize_parameters import initialize_pm
import brian2
import numpy as np
from numba import jit, njit, prange
import numpy.matlib
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn
from nilearn import plotting
import surfdist
from surfdist import utils, load
import configparser

# Set up simulation details
config = configparser.ConfigParser()
config.read('./config.ini')
c = config['SETUP']

# setup connectivity
surface_connectivity = get_surface_connectivity(c.get('surf_file'), c.get('cort_file'))
tract_connectivity   = get_longdistance_connectivity(c.get('surf_file'),
                                                     c.get('cort_file'),
                                                     './average_sc.pconn.nii',
                                                     'Glasser_NetworkPartition_v9.fsa4.L.label.gii')

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

@njit(parallel=True)
def mk_tr_delay(tr, sr, pr):
    for idx in prange(tr.shape[0]):
        for idy in prange(tr.shape[0]):
            sr[idx,idy] = pr[tr[idx,idy],idy]
    return sr

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
        S_tract_delay = mk_tr_delay(tract_delays-i_t, S_tract_delay, snmda)
        #S_tract_delay = pm['S_NMDA'][i_t-tract_delays,range(num_nodes),0]

    # Update the GABA synapses
    pm['S_GABA'][i_t,:,1] = pm['S_GABA'][i_t-1,:,1] + dt * GABA_deriv(pm['S_GABA'][i_t-1,:,1],pm['R'][i_t,:,1],pl)

    return pm, pl, S_tract_delay

for i_t1 in range(1,num_iterations):
    pm, pl, S_tract_delay = run_model(i_t1, pm, pl, S_tract_delay, tract_connectivity, surface_connectivity, tract_delays, I_ext, dt, num_nodes)

# save output
if c.get('save_output') == 'True':
    np.save('R.npy', pm['R'])

start_time = 0.0
end_time   = 3.0
R          = pm['R']
tm = np.arange((start_time)*brian2.second,(end_time)*brian2.second,dt)

if c.get('viz_ts') == 'True':
    # To visualize response to stimulus
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(15,10), dpi= 80, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': 20})

    # Plot E population rates
    tsE = np.mean(R[np.arange(int(start_time/dt),int(end_time/dt),1),:,0][:,V1_index], axis=1)
    plt.plot(tm,tsE,color='r')
    # Plot I population rates
    tsI = np.mean(R[np.arange(int(start_time/dt),int(end_time/dt),1),:,1][:,V1_index], axis=1)
    plt.plot(tm,tsI,color='b')

    # Plot the stimulation time
    plt.plot([stim_on,stim_off],[np.max(tsE)*1.1,np.max(tsE)*1.1],color='r',linewidth=5.0)

    # place text above the stimulation line
    axes = plt.gca()
    axes.text(0.4, 1.05,'External stimulation', transform=axes.transAxes, fontsize=20, verticalalignment='top')

    plt.legend(['E','I'])
    plt.xlabel('time (s)')
    plt.ylabel('firing rate (Hz)')
    # plt.ylim(0, 80)

    plt.savefig('./figures/stim_response.png')
    plt.close(fig)

if c.get('viz_surf') == 'True':
    # plot surfaces:
    fig, ax = plt.subplots(2,10, figsize=(60,10), subplot_kw={'projection': '3d'})
    surf   = nib.freesurfer.read_geometry(c.get('surf_file_inf'))
    cortex = np.sort(nib.freesurfer.read_label(c.get('cort_file')))
    t = np.arange(int(start_time/dt),int(end_time/dt))
    idx = np.int(np.round(len(t)/10))
    tsE = R[t,:,0][0::idx,:] / brian2.Hz
    t_split = t[0::idx]*dt
    vmax = np.max(tsE)
    for i in range(10):
        vals = surfdist.utils.recort(tsE[i,:], surf, cortex)
        # fix time title; should be in seconds
        plotting.plot_surf_stat_map(c.get('surf_file_inf'), vals, vmax = vmax,
                                    view = 'lateral',
                                    title=('%s' % t_split[i]),
                                    colorbar = False, axes=ax[0][i]
                                   )
        plotting.plot_surf_stat_map(c.get('surf_file_inf'), vals, vmax = vmax,
                                    view = 'medial',
                                    colorbar = False, axes=ax[1][i]
                                   )
    plt.savefig('./figures/stim_brains_E.png')
    plt.close(fig)
