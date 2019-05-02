# %load run.py


import connectivity
import local_circuit 
import delays 

surf_file = '../fsaverage4/lh.pial'
cort_file = '../fsaverage4/lh.cortex.label'

surface_connectivity = get_surface_connectivity(surf_file, cort_file)

tract_connectivity = get_longdistance_connectivity(surf_file, cort_file, '../conn_data/average_sc.pconn.nii', 'fsa4')

tract_delays, max_delay, S_tract_delay = get_delays(surf_file, cort_file)

parameters, num_iterations, I_noise, dt, noise_rhs, I_longRange_NMDA, I_midRange_NMDA, S_NMDA, I_local_NMDA, J_NMDA, J_GABA, S_GABA, I_local_GABA, I_total, I_0, R = initialize_parameters(len(surface_connectivity))

for i_t in range(1,num_iterations):

    # update noise - dims = num local pops x num areas
    I_noise = I_noise + -I_noise*(dt/parameters['tau_AMPA']) + noise_rhs[i_t-1,:,:]

    # long range NMDA (along white matter tracts) with variable delays
    I_longRange_NMDA[i_t-1,:,:] = parameters['g_E_longRange'] * np.sum(tract_connectivity*S_tract_delay,axis=1)

    # mid range NMDA (along surface), with a 6 timestep delay
    I_midRange_NMDA[i_t-1,:,:] = parameters['g_E_midRange'] * surface_connectivity.dot(S_NMDA[i_t-7,:,:])

    # local NMDA
    I_local_NMDA[i_t-1,:,:] = J_NMDA.dot(S_NMDA[i_t-1,:,:].T).T

    # sum up all the local GABA current onto E and I cell somas
    I_local_GABA[i_t-1,:,:] = J_GABA.dot(S_GABA[i_t-1,:,:].T).T

    # Define total input current as sum of local NMDA & GABA inputs, with background, external currents & noise & longRange
    I_total[i_t-1,:,:] = I_local_NMDA[i_t-1,:,:] +  I_local_GABA[i_t-1,:,:] + I_0 + I_ext[i_t-1,:,:] + I_noise + I_midRange_NMDA[i_t-1,:,:] + I_longRange_NMDA[i_t-1,:,:]

    # Update the firing rates of the excitatory population
    R[i_t,:,0] = R[i_t-1,:,0] + dt*current_to_frequency(I_total[i_t-1,:,0],'E',parameters)/parameters['tau_rates']-dt*R[i_t-1,:,0]/parameters['tau_rates']

    # Update the firing rates of the inhibitory population.
    R[i_t,:,1] = R[i_t-1,:,1] + dt*current_to_frequency(I_total[i_t-1,:,1],'I',parameters)/parameters['tau_rates']-dt*R[i_t-1,:,1]/parameters['tau_rates']

    # Update the NMDA synapses
    S_NMDA[i_t,:,0] = S_NMDA[i_t-1,:,0] + dt*NMDA_deriv(S_NMDA[i_t-1,:,0],R[i_t,:,0],parameters)

    S_tract_delay = S_NMDA[i_t-tract_delays,range(num_vertices),0]

    # Update the GABA synapses
    S_GABA[i_t,:,1] = S_GABA[i_t-1,:,1] + dt*GABA_deriv(S_GABA[i_t-1,:,1],R[i_t,:,1],parameters)
