import jax
from jax import numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

def get_transition_matrix(nstates,kon,koff):
    
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
nstates = 4
window_size = 10
k_on = 2
k_off = 1

T = get_transition_matrix(nstates,k_on,k_off)

