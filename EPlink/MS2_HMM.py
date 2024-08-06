import jax
from jax import numpy as jnp
from jax.experimental import sparse
import numpy as np
from matplotlib import pyplot as plt
import itertools

def get_transition_matrix(nstates,kon,koff):
    """Get the single timepoint transition matrix for a promoter model with only up and down transitions allowed at the same rates

    Parameters
    ----------
    nstates : int
        Number of promoter states in the model
    kon : float
        rate of transitions up
    koff : float
        rate of transitions down

    Returns
    -------
    ndarray
        Transition rate matrix for the 
    """    
    # Initialize the transition matrix
    T = np.zeros((nstates,nstates))
    
    # Fill in the diagonal elements
    for i in range(nstates):
        if i>0 and i<nstates-1:
            T[i,i] = -1*(kon + koff)
        elif i==0:
            T[i,i] = -1*kon
        else:
            T[i,i] = -1*koff
    
    # Fill in the off diagonal elements
    for i in range(nstates-1):
        T[i,i+1] = koff
        T[i+1,i] = kon
    return T
def get_cp_transition_matrix(T,window_size):
    """Generate the transition probability matrix for the compund promoter state. Most transitions are not allowed due to the deterministic translocation of all but the last state. 
    
    Transitions from a compund state i with substates (a_1,a_2,...,a_n) to a combound state j with substates (b_1,b_2,...,b_n) is only allowed if the last n-1 states (a_2,..,a_n) equal the first n-1 states of the next state (b_1,b_2,...,b_n). If this is the case, the transition occurs at a rate given by the matrix element <b_n|T|a_n>.
    
    The state order is defined as the one output by itertools.product(range(nstates),repeat=window_size). This means that the order is the one one would obtain by using a nested that cycles like an odometer.
    
    Most elements are thus zero, and the matrix is sparse. To take advantage of this, the matrix is converted to the jax BCOO matrix format.

    Parameters
    ----------
    T : array
        Transition probability matrix for the single promoter model (note, this is not the rate matrix)
    window_size : int
        Number of timepoints it takes the polymerase to traverse the gene
    """
    nstates = T.shape[0]
    data = []
    indices = []
    if not np.all(np.sum(T,axis=0) == 1.0):
        raise ValueError("The transition matrix is not stochastic. All columns should sum to 1")
    
    for i, state_i in enumerate(itertools.product(range(nstates),repeat=window_size)):
        for j, state_j in enumerate(itertools.product(range(nstates),repeat=window_size)):
            # Check if the states are connected
            if state_j[1:] == state_i[:-1]:
                data.append(T[state_i[-1],state_j[-1]])
                indices.append((i,j))
    T_mat_sp = sparse.BCOO((jnp.array(data),jnp.array(indices)),shape=(nstates**window_size,nstates**window_size))
    
    return T_mat_sp

def Generate_state_map(nstates,window_size):
    """Generate a dictionary where the keys are strings of the compund states and the values are the number of that state.

    Parameters
    ----------
    nstates : int
        Number of states in the single promoter model
    window_size : int
        Number of timepoints it takes the polymerase to traverse the gene
    """
    state_map = {}
    for i, state_i in enumerate(itertools.product(range(nstates),repeat=window_size)):
        state_string = "".join([str(x) for x in state_i])
        state_map[state_string] = i
    return state_map
    
  
def Project_to_single_states(cp_probs,state_map,timepoints,n_single_states):
    """Project the compound promoter state probabilities to the single states. The projected probabilities at timepoint i is defined as the sum of the probabilities of all compound states that have the same single state at timepoint i.

    Parameters
    ----------
    cp_probs : ndarray
        Array with the probabilities of the compound states
    state_map : dict
        Dictionary with the mapping from the compound states to the single states
    timepoint : list of int
        Timepoints where the projected probabilities are to be computed
    n_single_states : int
        Number of single states in the model

    Returns
    -------
    ndarray
        (len(timepoints),n_single_states) array with the probabilities of the single states at each timepoint 
    """
    single_probs = np.zeros((len(timepoints),n_single_states))
    for key,value in state_map.items():
        for i,timepoint in enumerate(timepoints):
            for j in range(n_single_states):
                if key[timepoint] == str(j):
                    single_probs[i,j] += cp_probs[value]
            
    return np.array(single_probs)

def Unwrap_cp_probabilities(cp_probs,smap,window_size):
    """Compute the probability sequence of the single timepoint HMM from a combound state probability sequence. This is done by marginalizing and keeping track of only newly introduced information. For the first timepoint all marginal probabilities are computed, while for the rest only marginalization over the latest timepoint is added.

    Parameters
    ----------
    cp_probs : iterable of cp probability vectors
        Iterable with the probability vectors of the compound state sequence
    smap : dict
        Dictionary with the mapping from the string form of the compound states to the indices in the probability vectors
    window_size : int
        Number of timepoints it takes the polymerase to traverse the gene

    Returns
    -------
    list
        List with the probability vectors of the single state sequence corresponding to the compound state sequence
    """
    nstates = int(len(cp_probs[0])**(1/window_size))
    
    # to compare with the cp states we unwrap them to the single states
    cp_unwrapped = []
    
    # unwrap the initial state
    for state in Project_to_single_states(cp_probs[0],smap,list(range(window_size)),nstates):
        cp_unwrapped.append(state)
    
    # unwrap the rest of the states by only adding the newest state
    for i in range(1,len(cp_probs)):
        unwrapped_last = Project_to_single_states(cp_probs[i],smap,[window_size-1],nstates)
        cp_unwrapped.append(unwrapped_last[0])
    return cp_unwrapped

    
    
# nstates = 2
# window_size = 3
# k_on = 2
# k_off = 1

# T = get_transition_matrix(nstates,k_on,k_off)


# # test that propagation under both the single and compound promoter models are the same
# dt = 0.01
# nsteps = 100

# state_0 = np.array([0,1]) # start in the on state
# # we start the cp state by making the last state on and the rest off
# smap = Generate_state_map(nstates,window_size)
# state_0_cp = np.zeros(nstates**window_size)
# state_0_cp[smap["001"]] = 1


# T_mat_sp = get_cp_transition_matrix(np.eye(nstates)+dt*T,window_size)

# # propagate the single promoter model for nsteps
# states = [state_0]
# states_cp = [state_0_cp]
# for i in range(nsteps):
#     states.append(states[-1] + dt*T@states[-1])
#     states_cp.append(T_mat_sp@states_cp[-1])

# # unwrap the cp states to the single states
# unwrapped_cp = Unwrap_cp_probabilities(states_cp,smap,window_size)

# np.array(states).shape
# # plot the results
# a_states = np.array(states)
# a_states_cp = np.array(unwrapped_cp[window_size-1:])

# np.allclose(a_states,a_states_cp)
# plt.plot(a_states[:,0])
# plt.plot(a_states[:,1])
# plt.plot(a_states_cp[:,0],linestyle="--")
# plt.plot(a_states_cp[:,1],linestyle="--")