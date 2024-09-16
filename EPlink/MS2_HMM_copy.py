import jax
from jax import numpy as jnp
import jax.scipy as jsp
from jax.experimental import sparse
import numpy as np
from matplotlib import pyplot as plt
import itertools
from tqdm import tqdm


def get_transition_matrix(nstates, kon, koff):
    """Get the transition matrix rate matrix for a promoter model with only up and down transitions allowed at the same rates

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
    # Initialize the transition rate matrix
    k_on_matrix = np.zeros((nstates, nstates))
    
    k_off_matrix = np.zeros((nstates, nstates))
    # Fill in the diagonal elements
    for i in range(nstates):
        if i > 0 and i < nstates - 1:
            k_on_matrix[i, i] = -1
            k_off_matrix[i, i] = -1
        elif i == 0:
            k_on_matrix[i, i] = -1
        else:
            k_off_matrix[i, i] = -1

    # Fill in the off diagonal elements
    for i in range(nstates - 1):
        k_off_matrix[i, i + 1] = 1
        k_on_matrix[i + 1, i] = 1
        
    return jnp.array(k_on_matrix) * kon + jnp.array(k_off_matrix) * koff  

# various parrallelized functions to compute transition rate matrices and transition probability matrices
v_get_transition_matrix_single = jax.vmap(
    get_transition_matrix, in_axes=(None, 0, None)
)
v_get_transition_matrix = jax.vmap(
    v_get_transition_matrix_single, in_axes=(None, 0, None)
)
v_matrix_exp_single = jax.vmap(jsp.linalg.expm)
v_matrix_exp = jax.vmap(v_matrix_exp_single)


def compute_matrix_scaffold(nstates, window_size, verbose=False):
    """Compute the scaffolds required to generate the transition probability matrix for the compound promoter state. The scaffolds are matrices in the large state space of the compound state which when multiplied with elements of the transition matrix for the single promoter model gives the transition probability matrix for the compound promoter state.

    Parameters
    ----------
    nstates : int
        Number of states in the single promoter model
    window_size : int
        Number of timepoints it takes the polymerase to traverse the gene
    verbose : bool, optional
        Wether to display a progress bar, by default False

    Returns
    -------
    matrices : list
        List with the scaffolds for the transition probability matrix in the format [[data,indices],...] suitable for creation of a sparse matrix
    ind_mat_dict : dict
        Dictionary which maps from the indices of the single state promoter transition matrix to the index in the matrices list
    """
    if verbose:
        iterator = tqdm(
            list(enumerate(itertools.product(range(nstates), repeat=window_size)))
        )
    else:
        iterator = enumerate(itertools.product(range(nstates), repeat=window_size))

    # make a matrix for each element of T_mat we do not fill them because it then allows for quick change of T_mat without having to regenerate the matrices which is useful when fitting parameters of T_mat
    matrices = []
    ind_mat_dict = {}
    count = 0
    for i in range(nstates):
        for j in range(nstates):
            matrices.append(
                [[], []]
            )  # first list is the data, second list is the indices
            ind_mat_dict[(i, j)] = count
            count += 1

    # fill the matrices
    for i, state_i in iterator:
        for j, state_j in enumerate(
            itertools.product(range(nstates), repeat=window_size)
        ):
            # Check if the states are connected
            if state_j[1:] == state_i[:-1]:
                ind_i, ind_j = state_i[-1], state_j[-1]

                # add the data and indices to the correct matrix
                mat_num = ind_mat_dict[(ind_i, ind_j)]
                matrices[mat_num][0].append(1)
                matrices[mat_num][1].append((i, j))
    return matrices, ind_mat_dict


def Assemble_matrix(matrices, ind_mat_dict, T_mat, window_size):
    """Use the output of compute_matrix_scaffold to assemble the transition probability matrix for the compound promoter state.

    Parameters
    ----------
    matrices : list
        List with the scaffolds for the transition probability matrix in the format [[data,indices],...] suitable for creation of a sparse matrix
    ind_mat_dict : dict
        Dictionary which maps from the indices of the single state promoter transition matrix to the index in the matrices list
    T_mat : array
        Transition probability matrix for the single promoter model (note, this is not the rate matrix)
    window_size : int
        Number of timepoints it takes the polymerase to traverse the gene. Only used to determine the size of the output matrix for sparse array creation.

    Returns
    -------
    jax sparce BCOO matrix
        Transition probability matrix for the compound promoter state
    """
    nstates = T_mat.shape[0]
    reverse_dict = {v: k for k, v in ind_mat_dict.items()}
    data, indices = [], []
    for i in range(len(matrices)):
        data += [
            m * T_mat[reverse_dict[i][0], reverse_dict[i][1]] for m in matrices[i][0]
        ]
        indices += matrices[i][1]
    return sparse.BCOO(
        (jnp.array(data), jnp.array(indices)),
        shape=(nstates**window_size, nstates**window_size),
    )


def get_cp_transition_matrix(T, window_size, verbose=False):
    """Generate the transition probability matrix for the compund promoter state. Most transitions are not allowed due to the deterministic translocation of all but the last state.

    Transitions from a compund state i with substates (a_1,a_2,...,a_n) to a combound state j with substates (b_1,b_2,...,b_n) is only allowed if the last n-1 states (a_2,..,a_n) equals the first n-1 states of the next state (b_1,b_2,...,b_n). If this is the case, the transition occurs at a rate given by the matrix element <b_n|T|a_n> of the single promoter model.

    The state order is defined as the one output by itertools.product(range(nstates),repeat=window_size). This means that the order is the one obtained by using a nested loop which cycles like an odometer.

    Most elements of the matrix are zero leading to a sparse output. To take advantage of this, the matrix is converted to the jax BCOO matrix format.

    Parameters
    ----------
    T : array
        Transition probability matrix for the single promoter model (note, this is not the rate matrix)
    window_size : int
        Number of timepoints it takes the polymerase to traverse the gene
    verbose : bool
        Print the progress of the matrix generation, defaults to False

    Returns
    -------
    jax sparce BCOO matrix
        Transition probability matrix for the compound promoter state
    """
    return Assemble_matrix(
        *compute_matrix_scaffold(T.shape[0], window_size, verbose), T, window_size
    )

# parallelized function to compute the compound promoter transition matrix given multiple single promoter transition matrices
v_get_cp_transition_matrix = jax.vmap(
    jax.vmap(get_cp_transition_matrix, in_axes=(0, None)), in_axes=(0, None)
)


def Generate_state_map(nstates, window_size):
    """Generate a dictionary where the keys are strings of the compund states and the values are the number of that state.

    Parameters
    ----------
    nstates : int
        Number of states in the single promoter model
    window_size : int
        Number of timepoints it takes the polymerase to traverse the gene

    Returns
    -------
    dict
        Dictionary with the mapping from the string form of the compound states to the indices in the probability vectors
    ndarray
        Array with the compound state sequences
    """
    state_map = {}
    state_sequences = np.zeros((nstates**window_size, window_size), dtype=int)
    for i, state_i in enumerate(itertools.product(range(nstates), repeat=window_size)):
        state_string = "".join([str(x) for x in state_i])
        state_map[state_string] = i
        state_sequences[i] = np.array(state_i)
    return state_map, jnp.array(state_sequences)


# def Project_to_single_states(cp_probs, state_map, timepoints, n_single_states):
#     """Project the compound promoter state probabilities to the single states. The projected probabilities at timepoint i is defined as the sum of the probabilities of all compound states that have the same single state at timepoint i.

#     Parameters
#     ----------
#     cp_probs : ndarray
#         Array with the probabilities of the compound states
#     state_map : dict
#         Dictionary with the mapping from the compound states to the single states
#     timepoint : list of int
#         Timepoints where the projected probabilities are to be computed
#     n_single_states : int
#         Number of single states in the model

#     Returns
#     -------
#     ndarray
#         (len(timepoints),n_single_states) array with the probabilities of the single states at each timepoint
#     """
#     single_probs = np.zeros((len(timepoints), n_single_states))
#     for key, value in state_map.items():
#         for i, timepoint in enumerate(timepoints):
#             for j in range(n_single_states):
#                 if key[timepoint] == str(j):
#                     single_probs[i, j] += cp_probs[value]

#     return np.array(single_probs)


# def Unwrap_cp_probabilities(cp_probs, smap, window_size):
#     """Compute the probability sequence of the single timepoint HMM from a combound state probability sequence. This is done by marginalizing and keeping track of only newly introduced information. For the first timepoint all marginal probabilities are computed, while for the rest only marginalization over the latest timepoint is added.

#     Parameters
#     ----------
#     cp_probs : iterable of cp probability vectors
#         Iterable with the probability vectors of the compound state sequence
#     smap : dict
#         Dictionary with the mapping from the string form of the compound states to the indices in the probability vectors
#     window_size : int
#         Number of timepoints it takes the polymerase to traverse the gene

#     Returns
#     -------
#     list
#         List with the probability vectors of the single state sequence corresponding to the compound state sequence
#     """
#     nstates = int(len(cp_probs[0]) ** (1 / window_size))

#     # to compare with the cp states we unwrap them to the single states
#     cp_unwrapped = []

#     # unwrap the initial state
#     for state in Project_to_single_states(
#         cp_probs[0], smap, list(range(window_size)), nstates
#     ):
#         cp_unwrapped.append(state)

#     # unwrap the rest of the states by only adding the newest state
#     for i in range(1, len(cp_probs)):
#         unwrapped_last = Project_to_single_states(
#             cp_probs[i], smap, [window_size - 1], nstates
#         )
#         cp_unwrapped.append(unwrapped_last[0])
#     return cp_unwrapped

#     # def Generate_sample(T_mat, nsteps, state_0, nsamples, verbose=False):
#     # """Generate a sample from the single timepoint HMM

#     # Parameters
#     # ----------
#     # T_mat : ndarray
#     #     Transition probability matrix for the single timepoint HMM. Note that this is not the rate matrix it is exp(dt*rate_matrix)
#     # nsteps : int
#     #     Number of steps in the sample
#     # state_0 : ndarray
#     #     Initial state probabilities
#     # nsamples : int
#     #     Number of samples to generate
#     # verbose : bool
#     #     Print the progress of the sample generation

#     # Returns
#     # -------
#     # ndarray
#     #     Samples from the single timepoint HMM
#     # """
#     # # check that the transition matrix is stochastic
#     # if not np.allclose(np.sum(T_mat, axis=0), 1.0):
#     #     raise ValueError(
#     #         "The transition matrix is not stochastic. All columns should sum to 1"
#     #     )

#     # # Initialize the random number generator
#     # seed = np.random.randint(0, 1e9)
#     # key = jax.random.PRNGKey(seed)

#     # # Initialize the sample containers
#     # samples = np.zeros((nsamples, nsteps + 1), dtype=int)
#     # samples[:, 0] = jax.random.choice(
#     #     key,
#     #     len(state_0),
#     #     replace=True,
#     #     p=state_0,
#     # )

#     # # Define a function to sample the next state
#     # def sample_state(key, probs):
#     #     return jax.random.choice(key, len(probs), replace=True, p=probs)

#     # v_sample_state = jax.vmap(sample_state)

#     # # setup the progress bar
#     # if verbose:
#     #     iterator = tqdm(list(range(nsteps)))
#     # else:
#     #     iterator = range(nsteps)

#     # # generate the samples
#     # for i in iterator:
#     #     # split the key to get a new one for each sample
#     #     key, _ = jax.random.split(key)
#     #     keys = jax.random.split(key, nsamples)

#     #     # sample the next state
#     #     transition_probs = T_mat.T[samples[:, i]]
#     #     samples[:, i + 1] = v_sample_state(keys, transition_probs)

#     # return np.array(samples)
#     # compute the propagators


def Generate_sample(k_ons, k_off, state_0, nsamples, verbose=False, seed=None):
    """Generate samples of a two-state promoter model with time-varying on rates

    Parameters
    ----------
    k_ons : (nsteps,) array
        Array with the on rates at each timepoint
    k_off : float
        Off rate of the promoter
    state_0 : (2,) array
        Initial state probabilities (must sum to 1)
    nsamples : int
        Number of samples to generate
    verbose : bool, optional
        Wether to display a progress bar for the sample generation, by default False
    seed : int, optional
        Seed for the random number generator, by default None in which case a random seed is generated.

    Returns
    -------
    (nsamples,nsteps+1) array
        Array with the samples from the two-state promoter model
    """
    nsteps = len(k_ons)

    # compute the propagators
    T_s = v_get_transition_matrix_single(nstates, k_ons, k_off)
    pmats = v_matrix_exp_single(T_s * dt)

    # Initialize the random number generator
    if seed is None:
        seed = np.random.randint(0, 1e12)
    key = jax.random.PRNGKey(seed)

    # Initialize the sample containers
    samples = np.zeros((nsamples, nsteps + 1), dtype=int)
    samples[:, 0] = jax.random.choice(
        key,
        jnp.arange(len(state_0)),
        shape=(nsamples,),
        replace=True,
        p=state_0,
    )

    # Define a function to sample the next state (to allow for vectorization with vmap)
    def sample_state(key, probs):
        return jax.random.choice(key, len(probs), replace=True, p=probs)

    v_sample_state = jax.vmap(sample_state)

    # setup the progress bar
    if verbose:
        iterator = tqdm(list(range(nsteps)))
    else:
        iterator = range(nsteps)

    # generate the samples
    for i in iterator:
        # split the key to get a new one for each sample
        key, _ = jax.random.split(key)
        keys = jax.random.split(key, nsamples)

        # sample the next state
        T_mat = pmats[i]
        transition_probs = T_mat.T[samples[:, i]]
        samples[:, i + 1] = v_sample_state(keys, transition_probs)

    return np.array(samples)


def swap_inds_for_rates(arr, rates):
    """Takes an array with integer values (0,1,2,...,len(rates)-1) and swaps them for the values in the rates array. The rate array should have the same length as the number of unique values in the array and it is assumed that only the values 0,1,2,...,n-1 are present in the array.

    Parameters
    ----------
    arr : jax array
        Array with integer values
    rates : array
        Array with the values to swap the integer values for

    Returns
    -------
    array
        Array with the integer values swapped for the values in the rates array
    """
    arr_out = jnp.zeros(arr.shape)
    for i, rate in enumerate(rates):
        arr_out = arr_out.at[arr == i].set(rate)
    return arr_out


@jax.jit
def MS2_kernel(i, tau):
    """Computes the MS2 kernel for a given timepoint i and a given delay tau

    Parameters
    ----------
    i : float
        Timepoint to compute the kernel for
    tau : float
        time for the polymerase to go through the MS2 array

    Returns
    -------
    float
        The MS2 kernel
    """
    return jnp.heaviside(i - tau, 1) + jnp.heaviside(tau - i, 0) * i / tau


def Get_emission_means(state_sequences, loading_rates, window_size, tau, dt):
    """Computes the mean of the emission distribution for each state in the model

    Parameters
    ----------
    state_sequences : (nstates**window_size,window_size) array
        Array with the compound state sequences as output by Generate_state_map
    loading_rates : (nstates,) array
        Array with the loading rates for the  different states promoter
    window_size : int
        Number of timepoints it takes the polymerase to traverse the gene
    tau : float
        Time for the polymerase to go through the MS2 array
    dt : float
        Time step of the measurements

    Returns
    -------
    (nstates**window_size,) array
        Array with the means of the emission distribution for each state in the model
    """
    return (
        swap_inds_for_rates(state_sequences, loading_rates)
        @ MS2_kernel(jnp.arange(window_size), tau)[::-1]
        * dt
    )


def vGaussian_measurement_model(observations, state_emission_means, sigma):
    """Computes the vector of probabilities of observing the measurement x given the state of the promoter model. The probabilities are computed for all states in the model.

    Parameters
    ----------
    observations : (nsamples,) array
        The measurements to compute the probabilities for
    state_emission_means : (nstates**window_size,) array
        means of the emission distribution for each state in the model
    sigma : float
        Standard deviation of the measurement


    Returns
    -------
    (nsamples,nstates**window_size,) array
        Array with the probabilities of observing the measurements given the state of the promoter model
    """

    return (1 / jnp.sqrt(2 * jnp.pi * sigma**2)) * jnp.exp(
        -0.5 * (observations[:, None] - state_emission_means[None, :]) ** 2 / sigma**2
    )


# def Gaussian_measurement_model(
#     x, state_sequences, window_size, loading_rates, sigma, tau, dt
# ):
#     """Computes the vector of probabilities of observing the measurement x given the state of the promoter model. The probabilities are computed for all states in the model.

#     Parameters
#     ----------
#     x : float
#         The measurement to compute the probabilities for
#     state_sequences : (nstates**window_size,window_size) array
#         Array with the compound state sequences as output by Generate_state_map
#     window_size : int
#         Number of timepoints it takes the polymerase to traverse the gene
#     loading_rates : (nstates,) array
#         Array with the loading rates for the promoter different states
#     sigma : float
#         Standard deviation of the measurement
#     tau : float
#         Time for the polymerase to go through the MS2 array
#     dt : float
#         Time step of the measurements

#     Returns
#     -------
#     (nstates**window_size,) array
#         Array with the probabilities of observing the measurement x given the state of the promoter model
#     """
#     state_emission_means = Get_emission_means(
#         state_sequences, loading_rates, window_size, tau, dt
#     )
#     return (1 / jnp.sqrt(2 * jnp.pi * sigma**2)) * jnp.exp(
#         -0.5 * (x - state_emission_means) ** 2 / sigma**2
#     )

# Convenience function to compute convolutions with the MS2 kernel and trajectory of loaded polymerases
conv_func = lambda x, y: jnp.convolve(x, y, mode="valid")
v_conv = jax.vmap(conv_func, in_axes=(0, None))


def Gen_MS2_measurement(pol2_rates, params, dt, random_pol2=False):
    """Generates a synthetic MS2 measurement from a given pol2 loading rate trajectory

    Parameters
    ----------
    pol2_rates : (ntrajectories,nsteps) array)
        Array with the average pol2 loading numbers at each timepoint (rate*dt)
    params : array
        array with the parameters for the MS2 measurement model (window_size,tau,noise_std)
    dt : float
        Time step of the measurements
    random_pol2 : bool, optional
        whether to sample random polymerase numbers or just use the mean, by default False

    Returns
    -------
    Loaded pol2 trajectory : (2,) tuple
        Tuple with the timepoints of the trajectory and the number of loaded polymerases at each timepoint
    MS2 measurement : (3,) tuple
        Tuple with the timepoints of the MS2 measurement, the MS2 signal (number of polymerases on the gene weighted by the MS2 kernel) and the MS2 signal with added noise.
    """
    # unpack the parameters
    w, tau, noise_std = params
    kernel_func = MS2_kernel(jnp.arange(w), tau)

    # Compute the number of polymerases loaded at each timepoint
    trajlen = pol2_rates.shape[1]
    if not random_pol2:
        pol2_nums = pol2_rates
    else:
        pol2_nums = np.random.poisson(pol2_rates)

    # Compute the MS2 signal by convolving the polymerase numbers with the MS2 kernel        
    MS2_signal = (
        v_conv(pol2_nums, kernel_func) * dt
    )  
    
    # compute the timepoints
    ts_MS2 = np.arange(w - 1, trajlen) * dt
    ts = np.arange(trajlen) * dt
    
    # add noise to the MS2 signal
    signal = MS2_signal + np.random.normal(0, noise_std, MS2_signal.shape)
    
    return (ts, pol2_nums), (ts_MS2, MS2_signal, signal)


def Run_forward_filter(
    state_sequences, k_off, k_ons, observation, measurement_error,loading_rates,compute_viterbi=False,verbose=False
):
    """Run the forward filter for the two-state promoter model with time-varying on rates. 

    Parameters
    ----------
    state_sequences : (2**window_size,window_size) array
        Array with the compound state sequences as output by Generate_state_map
    k_off : float
        Off rate of the promoter
    k_ons : (nsamples,nsteps) array
        Array with the on rates at each timepoint for each sample
    observation : (nsamples,nsteps) array
        Array with the measurements to compute the probabilities for
    measurement_error : float
        Standard deviation of the measurement noise
    loading_rates : (2,) array
        Array with the expected loading counts for the promoter states
    compute_viterbi : bool, optional
        Wether to compute the most likely state sequence (viterbi path). This takes longer to compute so may not be of interest if one seeks only the log likelihood, by default False
    verbose : bool, optional
        Wether to display a progress bar, by default False

    Returns
    -------
    posteriors : (nsamples,nsteps,nstates**window_size) array
        Array with the filter posterior probabilities for each state at each timepoint for each sample
    LLH : float
        Log likelihood of the observations given the model
    path : (nsamples,nsteps) array
        Most likely state sequence for each sample (only returned if compute_viterbi is True)
    unwrapped_paths : (nsamples,nsteps) array
        Most likely state sequence for each sample unwrapped to the single state model (only returned if compute_viterbi is True)
    """

    #check that k_ons have the same shape as the observation. This will not throw an error if not checked as the sparse array format doesn't check shapes.
    if k_ons.shape != observation.shape:
        raise ValueError("The shape of k_on and the observation should match but got {} and {}".format(k_on.shape,observation.shape))
    
    # compute the prior probabilities from the single state prior
    priors_un_norm = jnp.array(
        [
            jnp.exp(
                jnp.log(
                    swap_inds_for_rates(
                        state_sequences,
                        jnp.array([k_off / (k_off + k_on), k_on / (k_off + k_on)]),
                    )
                ).sum(axis=1)
            )
            for k_on in k_ons[:, 0]
        ]
    )
    priors = priors_un_norm / jnp.sum(priors_un_norm, axis=1)[:, None]

    # compute the emission probabilities for the input loading rates
    state_emission_means = Get_emission_means(
        state_sequences, loading_rates, window_size, tau, dt
    )

    # compute the probabilities of observing the data given the state of the promoter model
    p_data = vGaussian_measurement_model(
        observation[:, 0], state_emission_means, measurement_error
    )
    alpha = p_data * priors

    # initialize the posteriors (forward pass probabilities)
    posteriors = jnp.zeros(
        (observation.shape[0], observation.shape[1], len(state_emission_means))
    )
    posteriors = posteriors.at[:, 0].set(alpha / jnp.sum(alpha, axis=1)[:, None])

    # compute the log likelihood for the first measurement
    LLH = jnp.log(alpha.sum(axis=1)).sum()
    
    # initialize the viterbi variables if needed
    if compute_viterbi:
        pi_vals = jnp.zeros(
            (observation.shape[0], observation.shape[1], len(state_emission_means))
        )
        Q_vals = jnp.zeros(
            (observation.shape[0], observation.shape[1], len(state_emission_means)),
            dtype=int,
        )

        pi_vals = pi_vals.at[:, 0].set(jnp.log(posteriors[:, 0]))
        # need to convert the sparse matrix to a dense one for the viterbi algorithm as we need some fancy outer sum things that the sparse format can't handle.
        # pmat_dense = pmat_cp.todense()

    # precompute the transition matrices for the compound promoter state (much faster to do this once than for each timepoint when using GPU)
    T_s = v_get_transition_matrix(nstates, k_ons, k_off)
    pmats = v_matrix_exp(T_s * dt)
    pmats_cp = v_get_cp_transition_matrix(pmats, window_size)
    
    # make iterator
    if verbose:
        iterator = tqdm(list(range(1, observation.shape[1])))
    else:
        iterator = range(1, observation.shape[1])
        
    # run the forward filter
    for i in iterator:
        # Propagate the probability to get the prior for the observation
        pmat_cps = pmats_cp[:, i]
        prior = vmat_prod(
            pmat_cps, posteriors[:, i - 1]
        )  

        # compute the probabilities of observing the data given the state of the promoter model
        p_data = vGaussian_measurement_model(
            observation[:, i], state_emission_means, measurement_error
        )
        # compute the posterior probabilities for the observation using bayes rule
        alpha = p_data * prior
        posteriors = posteriors.at[:, i].set(alpha / jnp.sum(alpha, axis=1)[:, None])

        # compute the log likelihood for the observation 
        LLH += jnp.log(alpha.sum(axis=1)).sum()

        # compute the viterbi variables if needed
        if compute_viterbi:
            # propagation of the most likely probability from the previous timepoint with the probability of observing the measurement. 
            newvals = (
                pi_vals[:, i - 1][:, :, None] + jnp.log(p_data)[:, None]
            ).swapaxes(1, 2) + jnp.log(pmat_cps.todense())
            
            # compute the most likely state and the probability of that state
            amax = jnp.argmax(newvals, axis=-1)
            pi_vals = pi_vals.at[:, i].set(newvals.max(axis=-1))
            
            # store the paths to the most likely state
            Q_vals = Q_vals.at[:, i].set(amax)

    # Compute the most likely state sequence if needed 
    if compute_viterbi:
        # initialize the path with the most likely state at the last timepoint
        path = jnp.zeros(observation.shape, dtype=int)
        path = path.at[:, -1].set(jnp.argmax(pi_vals[:, -1], axis=1))

        # backtrack to get the most likely state sequence
        for i in range(observation.shape[1] - 1, 0, -1):
            next_vals = Q_vals[:, i, :][
                jnp.arange(len(observation)), path[:, i][jnp.arange(len(observation))]
            ]
            path = path.at[:, i - 1].set(next_vals)

        # unwrap the compound state sequence to the single state sequence
        unwrapped_paths = state_sequences[path, -1]
        return posteriors, LLH, path, unwrapped_paths
    return posteriors, LLH


# vmat_prod = jax.vmap(lambda pmat_cp, posterior: pmat_cp @ posterior, in_axes=(0, 0))

# nstates = 2
# k_on = 1.0
# k_off = 1.0
# measurement_error = 0.5
# window_size = 10
# tau = 5
# loading_rates = np.array([1.0, 4.0])
# ntimesteps = 300
# nsamples = 100

# T = get_transition_matrix(nstates, k_on, k_off)
# dt = 0.1
# pmat = jsp.linalg.expm(T * dt)
# pmat_cp = get_cp_transition_matrix(pmat, window_size)

# ts = jnp.arange(ntimesteps)
# k_on_driver = k_on*jnp.ones(ntimesteps)#jnp.exp(-((ts - ntimesteps / 2) ** 2) / 2 / (ntimesteps / 10) ** 2) * k_on
# samples = Generate_sample(
#     k_on_driver,
#     k_off,
#     np.array(
#         [
#             1 - k_on_driver[0] / (k_on_driver[0] + k_off),
#             k_on_driver[0] / (k_on_driver[0] + k_off),
#         ]
#     ),
#     nsamples,
#     verbose=True,
# )

# ts, signal, observation = Gen_MS2_measurement(
#     swap_inds_for_rates(samples, loading_rates),
#     (window_size, tau, measurement_error),
#     dt,
# )[1]

# smap, state_sequences = Generate_state_map(nstates, window_size)

# r1,r2 = loading_rates
# k_on_driver = k_on*jnp.ones(ntimesteps)
# k_ons = jnp.repeat(k_on_driver[:observation.shape[1] ][None, :], nsamples, axis=0)

# posteriors, LLH,path,unwrapped_paths = Run_forward_filter(
# state_sequences, k_off, k_ons, observation, measurement_error,[r1,r2],compute_viterbi=True,verbose=True
# )

# means = Get_emission_means(state_sequences, loading_rates, window_size, tau, dt)
# predicted_signal = means[path]

# i = np.random.randint(0, len(samples))
# fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
# ax[0].plot(dt * jnp.arange(len(samples[i])), samples[i], label="True")
# ax[0].plot(ts, unwrapped_paths[i], "--", label="Predicted")

# ax[0].legend()
# ax[0].set(xlabel="Time", ylabel="State", yticks=[0, 1], yticklabels=["Off", "On"])
# ax[0].legend(loc="lower right")
# tx = ax[1]
# tx.plot(ts, observation[i], label="Observed")
# tx.plot(ts, signal[i], label="True")
# tx.plot(ts, predicted_signal[i], "--", label="Predicted")
# tx.legend(loc="lower right")
# tx.set(xlabel="Time", ylabel="Signal")



# def LLHfunc(params):
#     k_on,k_off,measurement_error,r1,r2 = jnp.exp(params)
#     k_on_driver = k_on*jnp.ones(ntimesteps)
#     k_ons = jnp.repeat(k_on_driver[:observation.shape[1] ][None, :], nsamples, axis=0)
    
#     posteriors, LLH = Run_forward_filter(
#     pmat_cp, state_sequences, k_off, k_ons, observation, measurement_error,[r1,r2],compute_viterbi=False,verbose=False
# )   
#     return -LLH
# vgf = jax.value_and_grad(LLHfunc)

# def runfunc(params):
#     val,grad = vgf(params)
#     if np.isnan(val):
#         val = np.inf
#     if type(params) == np.ndarray:
#         print(np.exp(params),val)
    
    
#     return val,grad

# print(f"Aiming for [{k_on},{k_off},{measurement_error},{loading_rates[0]},{loading_rates[1]}]")

# from scipy.optimize import minimize
# res = minimize(runfunc,np.log([.1,0.6,1.2,0.6,3.]),jac=True,options={"disp":True},bounds=[(np.log(0.2),None),(np.log(0.2),None),(np.log(0.2),None),(np.log(0.2),None),(np.log(0.2),None)])

# koffs = np.linspace(0.5,2,11)
# LLH_out = []
# for koff in tqdm(koffs):
#     LLH_out.append(LLHfunc([res.x[0],koff,res.x[2],res.x[3],res.x[4]]))
# plt.plot(koffs,LLH_out)

# k_ons = jnp.repeat(k_on_driver[window_size-2: ][None, :], nsamples, axis=0)
# (np.arange(ntimesteps)*dt)[window_size-2: ]
# ts


# kon_guesses = np.linspace(0.1,2,11)
# LLH_out = []
# for kguess in tqdm(kon_guesses):
#     ts_here = jnp.arange(ntimesteps)
#     k_on_driver_here = jnp.exp(-((ts_here - ntimesteps/2) ** 2) / 2 / (ntimesteps / 10) ** 2) * kguess
#     k_ons = jnp.repeat(k_on_driver_here[:observation.shape[1] ][None, :], nsamples, axis=0)
#     posteriors, LLH, path, unwrapped_paths = Run_forward_filter(
#         pmat_cp, state_sequences, k_off, k_ons, observation, measurement_error,compute_viterbi=True,verbose=False
#     )
#     LLH_out.append(LLH)

# plt.plot(kon_guesses, LLH_out)
# plt.axvline(1,ls="--",color="k")

# means = Get_emission_means(state_sequences, loading_rates, window_size, tau, dt)
# predicted_signal = means[path]

# i = np.random.randint(0, len(samples))
# fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
# ax[0].plot(dt * jnp.arange(len(samples[i])), samples[i], label="True")
# ax[0].plot(ts, unwrapped_paths[i], "--", label="Predicted")

# ax[0].legend()
# ax[0].set(xlabel="Time", ylabel="State", yticks=[0, 1], yticklabels=["Off", "On"])
# ax[0].legend(loc="lower right")
# tx = ax[1]
# tx.plot(ts, observation[i], label="Observed")
# tx.plot(ts, signal[i], label="True")
# tx.plot(ts, predicted_signal[i], "--", label="Predicted")
# tx.legend(loc="lower right")
# tx.set(xlabel="Time", ylabel="Signal")
# fig.savefig("test.png",dpi=500)


# # test that propagation under both the single and compound promoter models are the same
# dt = 0.01
# nsteps = 100

# blocksize = 1000
# #make the blocks from state_sequences
# nblocks = len(state_sequences)//blocksize
# blocks = jnp.array_split(state_sequences,nblocks)
# block_inds = jnp.array_split(jnp.arange(len(state_sequences)),nblocks)
# remainder = len(state_sequences)%blocksize
# if remainder>0:
#     block_inds.append(jnp.arange(len(state_sequences))[-remainder:])
#     blocks.append(state_sequences[-remainder:])

# # def run_item(i,state_i,T,state_sequences):
# #     relevant_inds = jnp.argwhere(jnp.all(state_sequences[:,1:] == state_i[:-1],axis=1))[:,0]
# #     state_js = state_sequences[relevant_inds]
# #     T_inds_j = state_js[:,-1]
# #     T_inds_i = state_i[-1]
# #     return T[T_inds_i,T_inds_j],jnp.array([jnp.repeat(i,len(relevant_inds)),relevant_inds]).T
# # v_run_item = jax.vmap(run_item,in_axes=(0,0,None,None))

# def compute_transition_entries(ind_i,state_sequences):
#     state_i = state_sequences[ind_i]
#     candidate_transitions = jnp.all(state_sequences[:,1:]==state_i[:-1],axis=1)
#     inds_j = jnp.argwhere(candidate_transitions,size=nstates)[:,0]
#     T_inds_j = state_sequences[inds_j][:,-1]
#     T_inds_i = state_i[-1]
#     return T[T_inds_i,T_inds_j],jnp.array([jnp.repeat(ind_i,len(inds_j)),inds_j]).T
# v_compute_transition_entries = jax.vmap(compute_transition_entries,in_axes=(0,None))


# a,b = v_compute_transition_entries(block_ind,state_sequences)


# for i, state_i in iterator:
#     relevant_inds = jnp.argwhere(jnp.all(state_sequences[:,1:] == state_i[:-1],axis=1))[:,0]
#     for j, state_j in zip(relevant_inds,state_sequences[relevant_inds]):
#         data.append(T[state_i[-1],state_j[-1]])
#         indices.append((i,j))
# T_mat_sp = sparse.BCOO((jnp.array(data),jnp.array(indices)),shape=(nstates**window_size,nstates**window_size))


# # def faster_get_cp_transition_matrix(T,window_size,state_sequences,verbose=False):
# #     """Generate the transition probability matrix for the compund promoter state. Most transitions are not allowed due to the deterministic translocation of all but the last state.

# #     Transitions from a compund state i with substates (a_1,a_2,...,a_n) to a combound state j with substates (b_1,b_2,...,b_n) is only allowed if the last n-1 states (a_2,..,a_n) equal the first n-1 states of the next state (b_1,b_2,...,b_n). If this is the case, the transition occurs at a rate given by the matrix element <b_n|T|a_n>.

# #     The state order is defined as the one output by itertools.product(range(nstates),repeat=window_size). This means that the order is the one one would obtain by using a nested that cycles like an odometer.

# #     Most elements are thus zero, and the matrix is sparse. To take advantage of this, the matrix is converted to the jax BCOO matrix format.

# #     Parameters
# #     ----------
# #     T : array
# #         Transition probability matrix for the single promoter model (note, this is not the rate matrix)
# #     window_size : int
# #         Number of timepoints it takes the polymerase to traverse the gene
# #     state_sequences : ndarray
# #         Array with the compound state sequences as output by Generate_state_map
# #     verbose : bool
# #         Print the progress of the matrix generation, defaults to False

# #     Returns
# #     -------
# #     jax sparce BCOO matrix
# #         Transition probability matrix for the compound promoter state
# #     """
# #     nstates = T.shape[0]
# #     data = []
# #     indices = []
# #     if not np.all(np.sum(T,axis=0) == 1.0):
# #         raise ValueError("The transition matrix is not stochastic. All columns should sum to 1")
# #     if verbose:
# #         iterator = tqdm(list(enumerate(state_sequences)))
# #     else:
# #         iterator = enumerate(state_sequences)

# #     for i, state_i in iterator:
# #         relevant_inds = jnp.argwhere(jnp.all(state_sequences[:,1:] == state_i[:-1],axis=1))[:,0]
# #         for j, state_j in zip(relevant_inds,state_sequences[relevant_inds]):
# #             data.append(T[state_i[-1],state_j[-1]])
# #             indices.append((i,j))
# #     T_mat_sp = sparse.BCOO((jnp.array(data),jnp.array(indices)),shape=(nstates**window_size,nstates**window_size))

# #     return T_mat_sp


# # smap,state_sequences = Generate_state_map(nstates,window_size)

# # def compute_transition_entries(ind_i,state_sequences):
# #     state_i = state_sequences[ind_i]
# #     candidate_transitions = jnp.all(state_sequences[:,1:]==state_i[:-1],axis=1)
# #     inds_j = jnp.argwhere(candidate_transitions)[:,0]
# #     T_inds_j = state_sequences[inds_j][:,-1]
# #     T_inds_i = state_i[-1]
# #     return T[T_inds_i,T_inds_j],jnp.array([jnp.repeat(ind_i,len(inds_j)),inds_j]).T
# # v_compute_transition_entries = jax.vmap(compute_transition_entries,in_axes=(0,None))
# # tvals = v_compute_transition_entries(jnp.arange(nstates**window_size),state_sequences)


# # for i, state_i in iterator:
# #         for j, state_j in enumerate(itertools.product(range(nstates),repeat=window_size)):
# #             # Check if the states are connected
# #             if state_j[1:] == state_i[:-1]:
# #                 data.append(T[state_i[-1],state_j[-1]])
# #                 indices.append((i,j))
