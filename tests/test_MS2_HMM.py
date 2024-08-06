import sys,os
# add the path to the library and load it
sys.path.insert(0, os.path.abspath("../EPlink"))
from EPlink import MS2_HMM
import numpy as np

def test_cp_state():
    """Test that the compound state sequence propagates as expected by comparing it to the single promoter state sequence once unwrapped"""    
    nstates = 2
    window_size = 3
    k_on = 2
    k_off = 1

    T = MS2_HMM.get_transition_matrix(nstates,k_on,k_off)


    # test that propagation under both the single and compound promoter models are the same
    dt = 0.01
    nsteps = 100

    state_0 = np.array([0,1]) # start in the on state
    # we start the cp state by making the last state on and the rest off
    smap = MS2_HMM.Generate_state_map(nstates,window_size)
    state_0_cp = np.zeros(nstates**window_size)
    state_0_cp[smap["001"]] = 1


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