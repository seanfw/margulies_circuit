def initialize_pm(parameters, num_nodes, num_iterations, dt):

    import numpy as np
    import brian2
    import pandas

    ######## LOCAL CONNECTIVITY MATRIX ########
    J =  np.array([
                    [parameters['g_E_self'] , parameters['g_EI']],
                    [parameters['g_IE'] , parameters['g_I_self']]
                  ]) * brian2.amp

    pops = ['E','I']
    pops_column_list  = ['from '+ mystring for mystring in pops]
    pops_row_list  = ['to '+ mystring for mystring in pops]

    J_NMDA = J*((J>0).astype(np.int))
    J_GABA = J*((J<0).astype(np.int))

    df_J = pandas.DataFrame(J, columns=pops_column_list, index=pops_row_list)

    num_pops  = J.shape[0]

    # Choose initial values for rates and synapse variables
    R0 = np.ones((num_nodes, num_pops))
    R0 = R0 * parameters['r0_E']
    S_NMDA0 = np.ones((num_nodes,num_pops)) * 0.1
    S_GABA0 = np.zeros((num_nodes,num_pops))

    #### Set up simulation details
    # Preassign rate and synapse matrices
    R           = np.zeros((num_iterations,num_nodes,num_pops)) * brian2.Hz
    R[0,:,:]      = R0
    S_NMDA      = np.zeros((num_iterations,num_nodes,num_pops))
    S_NMDA[0,:,:] = S_NMDA0
    S_GABA      = np.zeros((num_iterations,num_nodes,num_pops))
    S_GABA[0,:,:] = S_GABA0

    # Create matrices in which we can store the currents
    I_longRange_NMDA = np.zeros((num_iterations,num_nodes,num_pops)) * brian2.pA
    I_midRange_NMDA  = np.zeros((num_iterations,num_nodes,num_pops)) * brian2.pA
    I_midRange_GABA  = np.zeros((num_iterations,num_nodes,num_pops)) * brian2.pA
    I_local_NMDA     =  np.zeros((num_iterations,num_nodes,num_pops)) * brian2.pA
    I_local_GABA     =  np.zeros((num_iterations,num_nodes,num_pops)) * brian2.pA
    I_total          =  np.zeros((num_iterations,num_nodes,num_pops)) * brian2.pA
    I_noise          =  np.zeros((num_nodes,num_pops)) * brian2.pA

    # # Define background inputs
    I_0 = np.zeros((num_nodes, num_pops)) * brian2.pA
    I_0[:,pops.index('E')] = parameters['I0_E']
    I_0[:,pops.index('I')] = parameters['I0_I']

    # Let's set up the noise. We will model the noise as an Ornstein-Uhlenbeck process.
    # https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process

    # Gaussian noise. mean 0, std 1. Dims: timesteps, local populations, areas
    eta = np.random.normal(loc=0.0, scale=1.0, size=(num_iterations,num_nodes,num_pops))

    # prepare the right hand side of the above equation
    noise_rhs = eta * ((np.sqrt(parameters['tau_AMPA']*np.power(parameters['std_noise'],2))*np.sqrt(dt))/parameters['tau_AMPA'])

    initial_pm = {}
    initial_pm.update({
        'noise_rhs' :  noise_rhs,
        'I_noise' :  I_noise,
        'I_midRange_NMDA' :  I_midRange_NMDA,
        'I_midRange_GABA' :  I_midRange_GABA,
        'I_longRange_NMDA' :  I_longRange_NMDA,
        'I_local_NMDA' :  I_local_NMDA,
        'I_local_GABA' :  I_local_GABA,
        'J_NMDA' :  J_NMDA,
        'J_GABA' :  J_GABA,
        'S_NMDA' :  S_NMDA,
        'S_GABA' :  S_GABA,
        'I_total' :  I_total,
        'I_0' :  I_0,
        'R' :  R,
        'num_pops' :  num_pops
    })

    return initial_pm
