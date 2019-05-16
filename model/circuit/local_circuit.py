import numpy as np
import brian2

def initialize_local(num_nodes):
    import configparser
    config_p = configparser.ConfigParser()
    config_p.read('./config.ini')
    p = config_p['CIRCUIT_PARAMETERS']

    parameters = {}
    parameters.update({
        # Time constants
        'tau_NMDA': float(p.get('tau_NMDA'))   * brian2.second,    # s
        'tau_AMPA': float(p.get('tau_AMPA'))   * brian2.second,    # s
        'tau_GABA': float(p.get('tau_GABA'))  * brian2.second,    # s
        'tau_rates': float(p.get('tau_rates')) * brian2.second,    # s

        # f-I curve parameters - E populations
        'a_E': float(p.get('a_E')) * brian2.Hz/brian2.nA / 2.,  # Hz/nA
        'b_E': float(p.get('b_E')) * brian2.Hz / 2.,            # Hz
        'd_E': float(p.get('d_E')) * brian2.second * 2.,        # s
        'gam': float(p.get('gam')) * 2.,                        # unitless # CHECK!!!!

        # f-I curve parameters - I populations
        'c_I': int(p.get('c_I')) * brian2.Hz/brian2.nA,                 # Hz/nA
        'r0_I': int(p.get('r0_I')) * brian2.Hz,

        # Strength of connections from E cells
        'g_E_self': float(p.get('g_E_self')) * brian2.nA ,        # nA - from E to E
        'g_IE': float(p.get('g_IE')) * brian2.nA ,            # nA - from E to I

        # Strength of connections from I cells
        'g_I_self': float(p.get('g_I_self'))  * brian2.nA,     # nA  - from I to I
        'g_EI': float(p.get('g_EI')) * brian2.nA,     # nA  - from I to E

        # Strength of mid-range connections (along surface)
        'g_E_midRange': float(p.get('g_E_midRange')) * brian2.nA, #0.08
        'g_I_midRange': float(p.get('g_I_midRange')) * brian2.nA, #0.04

        'midrange_delay': int(p.get('midrange_delay')), # currently in time_steps

        # Strength of long-range connections (white matter)
        'g_E_longRange': float(p.get('g_E_longRange')) * brian2.nA,

        # Background inputs
        'I0_E': float(p.get('I0_E'))     * brian2.nA, #0.31           # nA - background onto E population
        'I0_I': float(p.get('I0_I'))      * brian2.nA,         # nA - background onto I population

        # Noise std dev
        'std_noise': float(p.get('std_noise')) * brian2.nA, # 0.01         # nA  - standard deviation of noise input

        # initial values
        'r0_E': int(p.get('r0_E')) * brian2.Hz,

        # stimulus strength
        'stim_strength': float(p.get('stim_strength')) * brian2.nA

        })

    return parameters

def current_to_frequency(input_current,population_type,parameters):
    if population_type == 'E':
        return np.divide((parameters['a_E']*input_current - parameters['b_E']),(1 - np.exp(-parameters['d_E']*(parameters['a_E']*input_current - parameters['b_E']))))
    if population_type == 'I':
        return np.maximum(parameters['c_I']*input_current + parameters['r0_I'],0)

def NMDA_deriv(S_NMDA_prev,rate_now,parameters):
    return -S_NMDA_prev/parameters['tau_NMDA'] + parameters['gam']*(1 - S_NMDA_prev)*rate_now

def GABA_deriv(S_GABA_prev,rate_now,parameters):
    return -S_GABA_prev/parameters['tau_GABA'] + rate_now
