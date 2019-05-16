import pytest
import numpy as np
import brian2

# Run tests on index_array
num_nodes       = 200
i_t             = num_nodes * 2
num_iterations  = num_nodes * 3
tract_delays    = np.random.randint(i_t-2, size=(num_nodes,num_nodes))
pm_nmda         = np.random.randn(num_iterations, num_nodes, 2)

def index_array(i_t,num_nodes,tract_delays,pm_nmda):

    S_tract_delay = pm_nmda[i_t-tract_delays,range(num_nodes),0]

    return S_tract_delay

def index_array_fast(i_t,tract_delays,pm_nmda):

    from circuit.speedups import mk_tr_delay

    #S_tract_delay = np.zeros(tract_delays.shape)
    snmda = np.asarray(pm_nmda[:,:,0]).squeeze()
    S_tract_delay = mk_tr_delay(i_t-tract_delays, snmda)

    return S_tract_delay

def test_index_array():

    S1 = index_array(i_t,num_nodes,tract_delays,pm_nmda)
    S2 = index_array_fast(i_t,tract_delays,pm_nmda)

    np.testing.assert_allclose(S1,S2,rtol=1e-06)

# Run tests on multi
tract_connectivity  = np.random.rand(num_nodes,num_nodes)
S_tract_delay       = np.random.rand(num_nodes,num_nodes)
pl                  = 0.2 * brian2.nA

def sum_mult(tract_connectivity,S_tract_delay,pl):

    pm = pl * np.sum(np.multiply(tract_connectivity,S_tract_delay),axis=1)

    return pm

def sum_mult_fast(tract_connectivity,S_tract_delay,pl):

    d_tract = np.einsum('ij,ij->i',tract_connectivity,S_tract_delay)
    pm = pl * d_tract

    return pm

def test_sum_mult():

    P1 = sum_mult(tract_connectivity,S_tract_delay,pl)
    P2 = sum_mult_fast(tract_connectivity,S_tract_delay,pl)

    np.testing.assert_allclose(P1,P2,rtol=1e-06)
