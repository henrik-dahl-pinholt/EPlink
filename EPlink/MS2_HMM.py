import jax
from jax import numpy as jnp
import jax.scipy as jsp
from jax.experimental import sparse
import numpy as np
from matplotlib import pyplot as plt
import itertools
from tqdm import tqdm


def get_transition_matrix(nstates, kon, koff):
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
    # T = np.zeros((nstates,nstates))
    k_on_matrix = np.zeros((nstates, nstates))
    k_off_matrix = np.zeros((nstates, nstates))
    # Fill in the diagonal elements
    for i in range(nstates):
        if i > 0 and i < nstates - 1:
            k_on_matrix[i, i] = -1
            k_off_matrix[i, i] = -1
            # T[i,i] = -1*(kon + koff)
        elif i == 0:
            k_on_matrix[i, i] = -1
            # T[i,i] = -1*kon
        else:
            k_off_matrix[i, i] = -1
            # T[i,i] = -1*koff

    # Fill in the off diagonal elements
    for i in range(nstates - 1):
        k_off_matrix[i, i + 1] = 1
        # T[i,i+1] = koff
        k_on_matrix[i + 1, i] = 1
        # T[i+1,i] = kon
    return jnp.array(k_on_matrix) * kon + jnp.array(k_off_matrix) * koff  # jnp.array(T)


def compute_matrix_scaffold(nstates, window_size, verbose=False):

    if verbose:
        iterator = tqdm(
            list(enumerate(itertools.product(range(nstates), repeat=window_size)))
        )
    else:
        iterator = enumerate(itertools.product(range(nstates), repeat=window_size))

    # make a matrix for each element of T_mat
    # we do not fill them because it then allows for quick change of T_mat without having to regenerate the matrices
    # which is useful when fitting parameters of T_mat
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
    nstates = T_mat.shape[0]
    if not np.all(np.isclose(np.sum(T_mat, axis=0), 1.0)):
        raise ValueError(
            "The transition matrix is not stochastic. All columns should sum to 1"
        )
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

    Transitions from a compund state i with substates (a_1,a_2,...,a_n) to a combound state j with substates (b_1,b_2,...,b_n) is only allowed if the last n-1 states (a_2,..,a_n) equal the first n-1 states of the next state (b_1,b_2,...,b_n). If this is the case, the transition occurs at a rate given by the matrix element <b_n|T|a_n>.

    The state order is defined as the one output by itertools.product(range(nstates),repeat=window_size). This means that the order is the one one would obtain by using a nested that cycles like an odometer.

    Most elements are thus zero, and the matrix is sparse. To take advantage of this, the matrix is converted to the jax BCOO matrix format.

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
    # nstates = T.shape[0]
    # data = []
    # indices = []
    # if not np.all(np.sum(T,axis=0) == 1.0):
    #     raise ValueError("The transition matrix is not stochastic. All columns should sum to 1")
    # if verbose:
    #     iterator = tqdm(list(enumerate(itertools.product(range(nstates),repeat=window_size))))
    # else:
    #     iterator = enumerate(itertools.product(range(nstates),repeat=window_size))

    # for i, state_i in iterator:
    #     for j, state_j in enumerate(itertools.product(range(nstates),repeat=window_size)):
    #         # Check if the states are connected
    #         if state_j[1:] == state_i[:-1]:
    #             data.append(T[state_i[-1],state_j[-1]])
    #             indices.append((i,j))
    # T_mat_sp = sparse.BCOO((jnp.array(data),jnp.array(indices)),shape=(nstates**window_size,nstates**window_size))

    # return T_mat_sp

    return Assemble_matrix(
        *compute_matrix_scaffold(T.shape[0], window_size, verbose), T, window_size
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


def Project_to_single_states(cp_probs, state_map, timepoints, n_single_states):
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
    single_probs = np.zeros((len(timepoints), n_single_states))
    for key, value in state_map.items():
        for i, timepoint in enumerate(timepoints):
            for j in range(n_single_states):
                if key[timepoint] == str(j):
                    single_probs[i, j] += cp_probs[value]

    return np.array(single_probs)


def Unwrap_cp_probabilities(cp_probs, smap, window_size):
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
    nstates = int(len(cp_probs[0]) ** (1 / window_size))

    # to compare with the cp states we unwrap them to the single states
    cp_unwrapped = []

    # unwrap the initial state
    for state in Project_to_single_states(
        cp_probs[0], smap, list(range(window_size)), nstates
    ):
        cp_unwrapped.append(state)

    # unwrap the rest of the states by only adding the newest state
    for i in range(1, len(cp_probs)):
        unwrapped_last = Project_to_single_states(
            cp_probs[i], smap, [window_size - 1], nstates
        )
        cp_unwrapped.append(unwrapped_last[0])
    return cp_unwrapped


def Generate_sample(T_mat, nsteps, state_0, nsamples, verbose=False):
    """Generate a sample from the single timepoint HMM

    Parameters
    ----------
    T_mat : ndarray
        Transition probability matrix for the single timepoint HMM. Note that this is not the rate matrix it is exp(dt*rate_matrix)
    nsteps : int
        Number of steps in the sample
    state_0 : ndarray
        Initial state probabilities
    nsamples : int
        Number of samples to generate
    verbose : bool
        Print the progress of the sample generation

    Returns
    -------
    ndarray
        Samples from the single timepoint HMM
    """
    # check that the transition matrix is stochastic
    if not np.allclose(np.sum(T_mat, axis=0), 1.0):
        raise ValueError(
            "The transition matrix is not stochastic. All columns should sum to 1"
        )

    # Initialize the random number generator
    seed = np.random.randint(0, 1e9)
    key = jax.random.PRNGKey(seed)

    # Initialize the sample containers
    samples = np.zeros((nsamples, nsteps + 1), dtype=int)
    samples[:, 0] = jax.random.choice(
        key,
        len(state_0),
        replace=True,
        p=state_0,
    )

    # Define a function to sample the next state
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
        transition_probs = T_mat.T[samples[:, i]]
        samples[:, i + 1] = v_sample_state(keys, transition_probs)

    return np.array(samples)


def swap_inds_for_rates(arr, rates):
    """Takes an array with integer values and swaps them for the values in the rates array. The rates array should have the same length as the number of unique values in the array and it is assumed that only the values 0,1,2,...,n-1 are present in the array.

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
def MS2_weighting_func(i, tau):
    """Computes the MS2 weighting function for a given timepoint i and a given delay tau

    Parameters
    ----------
    i : float
        Timepoint to compute the weighting function for
    tau : float
        time for the polymerase to go through the MS2 array

    Returns
    -------
    float
        The MS2 weighting function
    """
    return jnp.heaviside(i - tau, 1) + jnp.heaviside(tau - i, 0) * i / tau


def Get_emission_means(state_sequences, loading_rates, window_size, tau, dt):
    """Get the mean of the emission distribution for each state in the model

    Parameters
    ----------
    state_sequences : (nstates**window_size,window_size) array
        Array with the compound state sequences as output by Generate_state_map
    loading_rates : (nstates,) array
        Array with the loading rates for the promoter different states
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
        @ MS2_weighting_func(jnp.arange(window_size), tau)[::-1]
        * dt
    )


def Gaussian_measurement_model(
    x, state_sequences, window_size, loading_rates, sigma, tau, dt
):
    """Computes the vector of probabilities of observing the measurement x given the state of the promoter model. The probabilities are computed for all states in the model.

    Parameters
    ----------
    x : float
        The measurement to compute the probabilities for
    state_sequences : (nstates**window_size,window_size) array
        Array with the compound state sequences as output by Generate_state_map
    window_size : int
        Number of timepoints it takes the polymerase to traverse the gene
    loading_rates : (nstates,) array
        Array with the loading rates for the promoter different states
    sigma : float
        Standard deviation of the measurement
    tau : float
        Time for the polymerase to go through the MS2 array
    dt : float
        Time step of the measurements

    Returns
    -------
    (nstates**window_size,) array
        Array with the probabilities of observing the measurement x given the state of the promoter model
    """
    state_emission_means = Get_emission_means(
        state_sequences, loading_rates, window_size, tau, dt
    )
    return (1 / jnp.sqrt(2 * jnp.pi * sigma**2)) * jnp.exp(
        -0.5 * (x - state_emission_means) ** 2 / sigma**2
    )


def Gen_MS2_measurement(pol2_rates, params, dt, random_pol2=False):
    """Generates a synthetic MS2 measurement from a given pol2 trajectory

    Parameters
    ----------
    pol2_rates : float
        Array with the pol2 rates at each timepoint
    params : array
        array with the parameters for the MS2 measurement model (window_size,tau,noise_std)
    dt : float
        Time step of the measurements
    random_pol2 : bool, optional
        whether to sample random polymerase numbers, by default False

    Returns
    -------
    tuple
        Tuple with the pol2 trajectory and the MS2 measurement
    """
    w, tau, noise_std = params
    kernel_func = MS2_weighting_func(jnp.arange(w), tau)

    trajlen = len(pol2_rates)
    if not random_pol2:
        pol2_nums = pol2_rates
    else:
        pol2_nums = np.random.poisson(pol2_rates)
    MS2_signal = jnp.convolve(pol2_nums, kernel_func, mode="valid") * dt
    ts_MS2 = np.arange(w - 1, trajlen) * dt
    ts = np.arange(trajlen) * dt
    signal = MS2_signal + np.random.normal(0, noise_std, len(MS2_signal))
    return (ts, pol2_nums), (ts_MS2, MS2_signal, signal)


def Run_forward_filter(
    pmat_cp, state_sequences, k_off, k_on, observation, compute_viterbi=False
):

    priors_un_norm = swap_inds_for_rates(
        state_sequences, jnp.array([k_off / (k_off + k_on), k_on / (k_off + k_on)])
    ).prod(axis=1)
    prior = priors_un_norm / jnp.sum(priors_un_norm)
    p_data = Gaussian_measurement_model(
        observation[0],
        state_sequences,
        window_size,
        loading_rates,
        measurement_error,
        tau,
        dt,
    )
    alpha = p_data * prior
    posteriors = [alpha / jnp.sum(alpha)]
    LLH = jnp.log(alpha.sum())
    if compute_viterbi:
        pi_vals = np.zeros((len(observation), len(prior)))
        Q_vals = np.zeros((len(observation), len(prior)))
        pi_vals[0] = jnp.log(posteriors[0])
        pmat_dense = pmat_cp.todense()
    for i in tqdm(list(range(1, len(observation)))):
        prior = pmat_cp @ posteriors[-1]
        p_data = Gaussian_measurement_model(
            observation[i],
            state_sequences,
            window_size,
            loading_rates,
            measurement_error,
            tau,
            dt,
        )
        alpha = p_data * prior
        posteriors.append(alpha / jnp.sum(alpha))
        LLH += jnp.log(alpha.sum())

        if compute_viterbi:
            newvals = (pi_vals[i - 1][:, None] + jnp.log(p_data)).T + jnp.log(
                pmat_dense
            )
            amax = jnp.argmax(newvals, axis=1)
            pi_vals[i] = newvals.max(axis=1)
            Q_vals[i] = amax
    if compute_viterbi:
        path = [jnp.argmax(pi_vals[-1])]
        for i in range(len(observation) - 1, 0, -1):
            path.append(Q_vals[i, path[-1].astype(int)])
        path = jnp.array(path[::-1]).astype(int)
        unwrapped_path = [int(state_sequences[i][-1]) for i in path]
        return posteriors, LLH, path, unwrapped_path
    return posteriors, LLH


nstates = 2
k_on = 1.0
k_off = 1.0
measurement_error = 0.5
window_size = 10
tau = 5
loading_rates = np.array([1.0, 4.0])
ntimesteps = 300

T = get_transition_matrix(nstates, k_on, k_off)

dt = 0.1
pmat = jsp.linalg.expm(T * dt)
pmat_cp = get_cp_transition_matrix(pmat, window_size)

sample = Generate_sample(pmat, ntimesteps, np.array([1, 0]), 1, verbose=True)
ts, signal, observation = Gen_MS2_measurement(
    swap_inds_for_rates(sample[0], loading_rates),
    (window_size, tau, measurement_error),
    dt,
)[1]

smap, state_sequences = Generate_state_map(nstates, window_size)

posteriors, LLH, path, unwrapped_path = Run_forward_filter(
    pmat_cp, state_sequences, k_off, k_on, observation, compute_viterbi=True
)


means = Get_emission_means(state_sequences, loading_rates, window_size, tau, dt)
predicted_signal = jnp.array([means[pi] for pi in path])


fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(dt * jnp.arange(len(sample[0])), sample[0], label="True")
ax[0].plot(ts, unwrapped_path, "--", label="Predicted")

ax[0].legend()
ax[0].set(xlabel="Time", ylabel="State", yticks=[0, 1], yticklabels=["Off", "On"])
ax[0].legend(loc="lower right")
tx = ax[1]
tx.plot(ts, observation, label="Observed")
tx.plot(ts, signal, label="True")
tx.plot(ts, predicted_signal, "--", label="Predicted")
tx.legend(loc="lower right")
tx.set(xlabel="Time", ylabel="Signal")
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
