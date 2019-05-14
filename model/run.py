# %load run.py

from connectivity import get_surface_connectivity, get_longdistance_connectivity
from local_circuit import initialize_local, current_to_frequency, NMDA_deriv, GABA_deriv
from model import get_stim
from delays import get_delays
from initialize_parameters import initialize_pm
import brian2
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn
from nilearn import plotting
import surfdist
from surfdist import utils, load

# Set up simulation details
long_range     = False
mid_range      = True
dt             = 0.5 * brian2.ms  # timestep
trial_length   = 3000 * brian2.ms # trial length (s)
num_iterations = int(trial_length/dt)
time           = np.arange(0, trial_length, dt)
stim_on        = 0.5 * brian2.second
stim_off       = 1.0 * brian2.second

surf_file      = './fsaverage4/lh.pial'
surf_file_inf  = './fsaverage4/lh.inflated'
cort_file      = './fsaverage4/lh.cortex.label'
annot_file     = './fsaverage4/lh.aparc.a2009s.annot'
label_name     = 'S_calcarine'

# begin
surface_connectivity = get_surface_connectivity(surf_file, cort_file)
num_nodes = len(surface_connectivity)

tract_connectivity = get_longdistance_connectivity(surf_file, cort_file, './average_sc.pconn.nii', 'Glasser_NetworkPartition_v9.fsa4.L.label.gii')

tract_delays, max_delay, S_tract_delay = get_delays(surf_file, cort_file)

pl = initialize_local(num_nodes, g_E_self=0.3, g_IE=0.23, g_I_self=-0.15, g_EI=-0.3)
pm = initialize_pm(pl, num_nodes, num_iterations, dt)

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

    # local NMDA
    pm['I_local_NMDA'][i_t-1,:,:] = pm['J_NMDA'].dot(pm['S_NMDA'][i_t-1,:,:].T).T

    # sum up all the local GABA current onto E and I cell somas
    pm['I_local_GABA'][i_t-1,:,:] = pm['J_GABA'].dot(pm['S_GABA'][i_t-1,:,:].T).T

    # long range NMDA (along white matter tracts) with variable delays
    if long_range:
        pm['I_longRange_NMDA'][i_t-1,:,0] = pl['g_E_longRange'] * np.sum(tract_connectivity*S_tract_delay,axis=1)

    # mid range NMDA (along surface), with a 7 timestep delay
    if mid_range:
        pm['I_midRange_NMDA'][i_t-1,:,:] = pl['g_E_midRange'] * dot_np(surface_connectivity,pm['S_NMDA'][i_t-7,:,:])

    # Define total input current as sum of local NMDA & GABA inputs, with background, external currents & noise & longRange
    if long_range and mid_range:
        pm['I_total'][i_t-1,:,:] = pm['I_local_NMDA'][i_t-1,:,:] +  pm['I_local_GABA'][i_t-1,:,:] + pm['I_0'] + I_ext[i_t-1,:,:] + pm['I_noise'] + pm['I_midRange_NMDA'][i_t-1,:,:] + pm['I_longRange_NMDA'][i_t-1,:,:]
    elif long_range:
        pm['I_total'][i_t-1,:,:] = pm['I_local_NMDA'][i_t-1,:,:] +  pm['I_local_GABA'][i_t-1,:,:] + pm['I_0'] + I_ext[i_t-1,:,:] + pm['I_noise'] + pm['I_longRange_NMDA'][i_t-1,:,:]
    elif mid_range:
        pm['I_total'][i_t-1,:,:] = pm['I_local_NMDA'][i_t-1,:,:] +  pm['I_local_GABA'][i_t-1,:,:] + pm['I_0'] + I_ext[i_t-1,:,:] + pm['I_noise'] + pm['I_midRange_NMDA'][i_t-1,:,:]
    else:
        pm['I_total'][i_t-1,:,:] = pm['I_local_NMDA'][i_t-1,:,:] +  pm['I_local_GABA'][i_t-1,:,:] + pm['I_0'] + I_ext[i_t-1,:,:] + pm['I_noise']

    # Update the firing rates of the excitatory population
    pm['R'][i_t,:,0] = pm['R'][i_t-1,:,0] + dt*current_to_frequency(pm['I_total'][i_t-1,:,0],'E',pl)/pl['tau_rates']-dt*pm['R'][i_t-1,:,0]/pl['tau_rates']

    # Update the firing rates of the inhibitory population.
    pm['R'][i_t,:,1] = pm['R'][i_t-1,:,1] + dt*current_to_frequency(pm['I_total'][i_t-1,:,1],'I',pl)/pl['tau_rates']-dt*pm['R'][i_t-1,:,1]/pl['tau_rates']

    # Update the NMDA synapses
    pm['S_NMDA'][i_t,:,0] = pm['S_NMDA'][i_t-1,:,0] + dt*NMDA_deriv(pm['S_NMDA'][i_t-1,:,0],pm['R'][i_t,:,0],pl)

    if long_range:
        S_tract_delay = pm['S_NMDA'][i_t-tract_delays,range(num_nodes),0]

    # Update the GABA synapses
    pm['S_GABA'][i_t,:,1] = pm['S_GABA'][i_t-1,:,1] + dt*GABA_deriv(pm['S_GABA'][i_t-1,:,1],pm['R'][i_t,:,1],pl)

np.save('R.npy', pm['R'])

# To check response to stimulus
plt.switch_backend('agg')
fig = plt.figure(figsize=(15,10), dpi= 80, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 20})

start_time = 0.1
end_time   = 3.0
R          = pm['R']
tm = np.arange((start_time)*brian2.second,(end_time)*brian2.second,dt)

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

# plot surfaces:
fig, ax = plt.subplots(2,10, figsize=(60,10), subplot_kw={'projection': '3d'})
surf   = nib.freesurfer.read_geometry(surf_file_inf)
cortex = np.sort(nib.freesurfer.read_label(cort_file))
t = np.arange(int(start_time/dt),int(end_time/dt))
idx = np.int(np.round(len(t)/10))
tsE = R[t,:,0][0::idx,:] / brian2.Hz
t_split = t[0::idx]*dt
vmax = np.max(tsE)
for i in range(10):
    vals = surfdist.utils.recort(tsE[i,:], surf, cortex)
    # fix time title; should be in seconds
    plotting.plot_surf_stat_map(surf_file_inf, vals, vmax = vmax,
                                view = 'lateral',
                                title=('%s' % t_split[i]),
                                colorbar = False, axes=ax[0][i]
                               )
    plotting.plot_surf_stat_map(surf_file_inf, vals, vmax = vmax,
                                view = 'medial',
                                colorbar = False, axes=ax[1][i]
                               )
plt.savefig('./figures/stim_brains_E.png')
plt.close(fig)
