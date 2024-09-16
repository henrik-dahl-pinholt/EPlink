import sys,os
# add the path to the library and load it
sys.path.insert(0, os.path.abspath("../EPlink"))
from EPlink import MS2_HMM
import numpy as np
from jax import scipy as jsp
import jax

def test_cp_state():
    """Test that the compound state sequence propagates as expected by comparing it to the single promoter state sequence once unwrapped"""    
    nstates = 2
    window_size = 5
    k_on = 2
    k_off = 1

    T = MS2_HMM.get_transition_matrix(nstates,k_on,k_off)


    # test that propagation under both the single and compound promoter models are the same
    dt = 0.01
    nsteps = 1000

    state_0 = np.array([0,1]) # start in the on state
    # we start the cp state by making the last state on and the rest off
    smap,_ = MS2_HMM.Generate_state_map(nstates,window_size)
    state_0_cp = np.zeros(nstates**window_size)
    state_0_cp[smap["00001"]] = 1


    # `T_mat_sp` is the transition matrix for the compound promoter model. It is calculated based on
    # the transition matrix `T` for the single promoter model and the window size. The function
    # `get_cp_transition_matrix` in the `MS2_HMM` module is used to generate this transition matrix
    # for the compound promoter model by considering the effects of multiple states within the window
    # size. This matrix is then used to propagate the compound promoter state sequence in the test
    # case.
    T_mat_sp = MS2_HMM.get_cp_transition_matrix(np.eye(nstates)+dt*T,window_size)

    # propagate the single promoter model for nsteps
    states = [state_0]
    states_cp = [state_0_cp]
    for i in range(nsteps):
        states.append(states[-1] + dt*T@states[-1])
        states_cp.append(T_mat_sp@states_cp[-1])

    # unwrap the cp states to the single states
    unwrapped_cp = MS2_HMM.Unwrap_cp_probabilities(states_cp,smap,window_size)

    a_states = np.array(states)
    a_states_cp = np.array(unwrapped_cp[window_size-1:])

    assert np.allclose(a_states,a_states_cp)
    
    p_on_theory = k_on/(k_off+k_on)
    assert np.allclose(np.mean(a_states_cp[200:],axis=0),[1-p_on_theory,p_on_theory],atol=0.01)
    assert np.allclose(np.mean(a_states_cp[200:],axis=0),[1-p_on_theory,p_on_theory],atol=0.01)
    
def test_Generate_trajectory():
    nstates = 2
    k_on = 5
    k_off = 1

    T = MS2_HMM.get_transition_matrix(nstates,k_on,k_off)
    
    # test that propagation under both the single and compound promoter models are the same
    dt = 0.01
    nsteps = 100
    nsamples = 100000
    state_0 = np.array([0,1]) # start in the on stat
    T_mat = np.eye(len(T))+T*dt

    samples = MS2_HMM.Generate_sample(T_mat,nsteps,state_0,nsamples,verbose=False)

    theory_line = np.exp(-(k_off+k_on)*np.arange(nsteps+1)*dt)+(k_on/(k_off+k_on))*(1-np.exp(-(k_off+k_on)*np.arange(nsteps+1)*dt))
    assert np.allclose(np.mean(samples,axis=0),theory_line,atol=0.01)
    
def test_mean_calc():
    # check that the thing works when it takes no time to get through the MS2 array
    nstates = 3
    window_size = 10
    tau = -0.1
    loading_rates = np.array([1.5,4.,10.])

    smap,state_sequences = MS2_HMM.Generate_state_map(nstates,window_size)        
    
    state_emission_means = MS2_HMM.swap_inds_for_rates(state_sequences,loading_rates)@MS2_HMM.MS2_weighting_func(np.arange(window_size),tau)
    assert state_emission_means[0]==15.
    assert state_emission_means[1]==17.5
    assert state_emission_means[-1]==100.
    
    # check that it works when it takes a finite time to get through
    tau = 2.
    loading_rates = np.array([1.,4.,10.])

    smap,state_sequences = MS2_HMM.Generate_state_map(nstates,window_size)        
    
    state_emission_means = MS2_HMM.swap_inds_for_rates(state_sequences,loading_rates)@MS2_HMM.MS2_weighting_func(np.arange(window_size),tau)
    assert state_emission_means[0]==8.5
    assert state_emission_means[1]==11.5
    assert state_emission_means[-1]==85.