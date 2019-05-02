import numpy as np
import brian2

def current_to_frequency(input_current,population_type,parameters):
    if population_type == 'E':
        return current_to_frequency_E(input_current,parameters['a_E'],parameters['b_E'],parameters['d_E'])
    if population_type == 'I':
        return current_to_frequency_I(input_current,parameters['c_I'],parameters['r0_I'])

def current_to_frequency_E(input_current1,a,b,d):
    return np.divide((a*input_current1 - b),(1 - np.exp(-d*(a*input_current1 - b))))

def current_to_frequency_I(input_current1,c_I,r_0):
    return np.maximum(c_I*input_current1 + r_0,0)

def NMDA_deriv(S_NMDA_prev,rate_now,parameters):
    return -S_NMDA_prev/parameters['tau_NMDA'] + parameters['gam']*(1 - S_NMDA_prev)*rate_now

def GABA_deriv(S_GABA_prev,rate_now,parameters):
    return -S_GABA_prev/parameters['tau_GABA'] + rate_now

def initialize_parameters(num_nodes, g_E_self=0.39, g_IE=0.23, g_I_self=-0.05, g_EI=-0.4):

    parameters = {}

    parameters.update({ # Time constants
                    'tau_NMDA': 0.06   * brian2.second,    # s
                    'tau_AMPA': 0.002   * brian2.second,    # s
                    'tau_GABA': 0.005  * brian2.second,    # s
                    'tau_rates': 0.002 * brian2.second,    # s

                    # f-I curve parameters - E populations
                    'a_E': 270.  * brian2.Hz/brian2.nA / 2.,  # Hz/nA
                    'b_E': 108.  * brian2.Hz / 2.,            # Hz
                    'd_E': 0.154 * brian2.second * 2.,        # s
                    'gam': 0.641 * 2.,                         # unitless

                    # f-I curve parameters - I populations
                    'c_I': 330 * brian2.Hz/brian2.nA,                 # Hz/nA
                    'r0_I': -95 * brian2.Hz,

                    # Strength of connections from E cells
                    'g_E_self': g_E_self * brian2.nA ,        # nA - from E to E
                    'g_IE': g_IE * brian2.nA ,            # nA - from E to I

                    # Strength of connections from I cells
                    'g_I_self': g_I_self  * brian2.nA,     # nA  - from I to I
                    'g_EI': g_EI * brian2.nA,     # nA  - from I to E

                    # Strength of mid-range connections (along surface)
                    'g_E_midRange': 0.08 * brian2.nA,
                    'g_I_midRange': 0.04 * brian2.nA,
                    'g_E_longRange': 0.2 * brian2.nA,


                    # Background inputs
                    'I0_E': 0.20     * brian2.nA,           # nA - background onto E population
                    'I0_I': 0.18      * brian2.nA,         # nA - background onto I population

                    # Noise std dev
                    'std_noise': 0.01 * brian2.nA,         # nA  - standard deviation of noise input

                    # initial values
                    'r0_E': 5 * brian2.Hz,

                    # stimulus strength
                    'stim_strength': 0.1 * brian2.nA
                        })

    return parameters

def initiatize_local(parameters, num_nodes, num_iterations, dt):

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

    import pandas
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

    return I_noise, noise_rhs, I_longRange_NMDA, I_midRange_NMDA, I_midRange_GABA, S_NMDA, I_local_NMDA, J_NMDA, J_GABA, S_GABA, I_local_GABA, I_total, I_0, R, num_pops
