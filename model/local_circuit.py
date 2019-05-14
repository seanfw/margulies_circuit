import numpy as np
import brian2

def current_to_frequency(input_current,population_type,parameters):
    if population_type == 'E':
        return np.divide((parameters['a_E']*input_current - parameters['b_E']),(1 - np.exp(-parameters['d_E']*(parameters['a_E']*input_current - parameters['b_E']))))
    if population_type == 'I':
        return np.maximum(parameters['c_I']*input_current + parameters['r0_I'],0)

def NMDA_deriv(S_NMDA_prev,rate_now,parameters):
    return -S_NMDA_prev/parameters['tau_NMDA'] + parameters['gam']*(1 - S_NMDA_prev)*rate_now

def GABA_deriv(S_GABA_prev,rate_now,parameters):
    return -S_GABA_prev/parameters['tau_GABA'] + rate_now

def initialize_local(num_nodes, g_E_self=0.39, g_IE=0.23, g_I_self=-0.05, g_EI=-0.4):

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
                    'gam': 0.641, #* 2.,                        # unitless # CHECK!!!!

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
                    'I0_E': 0.31     * brian2.nA,           # nA - background onto E population
                    'I0_I': 0.22      * brian2.nA,         # nA - background onto I population

                    # Noise std dev
                    'std_noise': 0.01 * brian2.nA,         # nA  - standard deviation of noise input

                    # initial values
                    'r0_E': 5 * brian2.Hz,

                    # stimulus strength
                    'stim_strength': 0.2 * brian2.nA
                        })

    return parameters
