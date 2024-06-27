from jax import numpy as jnp
import numpy as np
import jax
"""This module contains code to simulate polymer dynamics, generate measurements of enhancer-promoter distances, and infer polymer positions from two-locus microscopy measurements. The polymer is modeled as a chain of beads connected by springs with dynamics described by the Rouse model."""
def Get_eigensystem(N):  
    """Generate the eigenvectors and eigenvalues of the spring matrix in the Fourier basis. These are used to convert between physical coordinates and the Rouse modes in which the dynamics are diagonal.

    Parameters
    ----------
    N : int
        The number of beads in the polymer

    Returns
    -------
    Qmat : (N,N-1) ndarray
        The N-1 eigenvectors of the spring matrix in the Fourier basis.
    eigvals : (N-1) ndarray
        The eigenvalues corresponding to the N-1 eigenvectors.
    """    
    ii,jj = jnp.meshgrid(jnp.arange(1,N+1),jnp.arange(2,N+1))
    Qmat = jnp.sqrt(2/N)*jnp.cos((ii-1/2)*(jj-1)*jnp.pi/N).T
    eigvals = 2*(1-jnp.cos((jnp.pi/N)*(jnp.arange(2,N+1)-1)))
    return Qmat,eigvals

def convert_modes_V(vector,Qmat):
    """Convenience function to convert vector from physical coordinates to Rouse modes.

    Parameters
    ----------
    vector : (N) ndarray
        Vector of bead coordinates
    Qmat : (N,N-1) ndarray
        The N-1 eigenvectors of the spring matrix in the Fourier basis as output by Get_eigensystem.

    Returns
    -------
    (N-1) ndarray
        The vector in the Rouse modes.
    """    
    return jnp.einsum("j,ij->i",vector,Qmat)
def convert_modes_M(matrix,Qmat):
    """Convenience function to convert matrix from physical coordinates to Rouse modes.

    Parameters
    ----------
    matrix : (N,N) ndarray
        Matrix which operates on bead coordinates
    Qmat : (N,N-1) ndarray
        The N-1 eigenvectors of the spring matrix in the Fourier basis as output by Get_eigensystem.

    Returns
    -------
    (N-1,N-1) ndarray
        The matrix in the Rouse modes basis.
    """    
    return jnp.einsum("jk,ij,lk->il",matrix,Qmat,Qmat)
def convert_modes_V_ep(vector,M_vector):
    """Convenience function to convert a vector from Rouse modes to enhancer promoter distance, i.e. a projection of the physical coordinates. This transformation involves two transformations: first a basis transformation to physical coordinates, then a projection to enhancer promoter distance. This is encompassed in the M_vector which must be the left product of the sought projection vector w and the matrix Qmat, i.e. M = w.T Q.
     

    Parameters
    ----------
    vector : (N-1) ndarray
        Vector in Rouse modes basis
    M_vector : (N-1) ndarray
        The left product of the sought projection vector w (in physical coordinate basis) and the matrix Qmat, i.e. M = w.T Q.   

    Returns
    -------
    float
        The enhancer promoter distance described by the rouse mode vector.
    """    
    return jnp.einsum("j,j->",vector,M_vector)

def convert_modes_M_ep(matrix,M_vector):
    """Convenience function to convert a matrix from Rouse modes to enhancer promoter distance basis, see convert_modes_V_ep for a definition of the M_vector.
    

    Parameters
    ----------
    matrix : (N-1,N-1) ndarray
        Matrix in Rouse modes basis
    M_vector : (N-1) ndarray
        The left product of the sought projection vector w (in physical coordinate basis) and the matrix Qmat, i.e. M = w.T Q.

    Returns
    -------
    float
        The matrix operation in the enhancer promoter distance projected space.
    """
    return jnp.einsum("jk,j,k->",matrix,M_vector,M_vector)


@jax.jit
def Propagate_Forward_diagonal(mu,timestep,k,eigvals,D):
    """Propagate a deterministic initial condition (i.e. delta function distribution at the vector mu) for the Rouse modes forward in time

    Parameters
    ----------
    mu : (N-1) ndarray or jax array
        Rouse mode vector at time 0
    timestep : float
        Time step for the propagation
    k : float
        Spring constant of the Rouse model
    eigvals : numpy array or jax array
        Eigenvalues of the spring matrix in the Fourier basis as output by Get_eigensystem
    D : float
        Diffusion constant of the Rouse model

    Returns
    -------
    mu_out : (N-1) jax array
        Rouse mode mean vector after the timestep
    var_out : (N-1) jax array
        Diagonal entries of the Rouse mode covariance matrix after the timestep (all off-diagonal entries are zero for the deterministic initial condition).
    """
    eigexp = jnp.exp(-k*eigvals*timestep)
    mu_out = mu*eigexp
    var_out = (D/(k*eigvals))*(1-jnp.exp(-2*k*eigvals*timestep))
    return mu_out,var_out


forward_d_vmap = jax.vmap(Propagate_Forward_diagonal,in_axes=(0,None,None,None,None))
forward_d_vmap.__doc__ = """Vectorized version of Propagate_Forward_diagonal. Takes the same arguments but the input mus must now have a leading axis to allow for inputting a batch of deterministic initial conditions.
    mus : (batch_size,N-1) ndarray or jax array
        Rouse mode vectors at time 0
    timestep : float
        Time step for the propagation
    k : float
        Spring constant of the Rouse model
    eigvals : numpy array or jax array
        Eigenvalues of the spring matrix in the Fourier basis as output by Get_eigensystem
    D : float
        Diffusion constant of the Rouse model
    Returns
    -------
    mu_out : (batch_size,N-1) jax array
        Rouse mode mean vector after the timestep
    var_out : (batch_size,N-1) jax array
        Diagonal entries of the Rouse mode covariance matrix after the timestep (all off-diagonal entries are zero for the deterministic initial condition).
    """

# def Gen_ss_samp(N,D,k,nsamples):
#     Qmat,eigvals = Get_eigensystem(N)
#     gaussian_samples = np.random.normal(0,1,size=(nsamples,N-1))
#     mean_0,covar_0 = np.zeros((nsamples,N-1)),(D/k/eigvals)*np.ones((nsamples,N-1))
#     Rouse_confs = mean_0+gaussian_samples*np.sqrt(covar_0)
#     confs = np.einsum("jk,lk->jl",Rouse_confs,Qmat) 
#     return confs

# def Generate_trajectory(nsteps,dt,n_trajectories,k,D,N,seed=0):
#     if not type(nsteps) == int:
#         raise ValueError("nsteps must be an integer")
#     Qmat,eigvals = Get_eigensystem(N)
#     key = jax.random.PRNGKey(seed)
#     T = nsteps*dt
#     # nsteps = int(T/dt)
#     a_samples = np.zeros((nsteps,n_trajectories,N-1))
#     mean_0,covar_0 = jnp.zeros((n_trajectories,N-1)),(D/k/eigvals)*jnp.ones((n_trajectories,N-1))
    
#     gaussian_samples = jax.random.normal(key,(n_trajectories,nsteps,N-1))
#     a_samples[0] =  mean_0+gaussian_samples[:,0]*jnp.sqrt(covar_0)
#     for i in range(1,nsteps):
#         new_mean,new_cov = forward_d_vmap(a_samples[i-1],dt,k,eigvals,D)
        
#         a_samples[i] = new_mean+gaussian_samples[:,i]*jnp.sqrt(new_cov)
#     polymer_coords = jnp.einsum("ijk,lk->ijl",a_samples,Qmat) 
#     return np.arange(0,len(polymer_coords))*dt,polymer_coords

def Generate_trajectory(nsteps,dt,n_trajectories,k,D,N,seed=0):
    """Generate a trajectory of a polymer chain with Rouse dynamics starting from the steady-state ensemble

    Parameters
    ----------
    nsteps : int
        Number of time steps to simulate
    dt : float
        Time step for the simulation specified in the same units as 1/k
    n_trajectories : int
        Number of polymer chains to simulate
    k : float
        Spring constant of the Rouse model
    D : float
        Diffusion constant of the Rouse model
    N : int
        Number of beads in the polymer
    seed : int, optional
        seed for the random number generator, by default 0

    Returns
    -------
    times : ndarray 
        Array of times at which the polymer coordinates are sampled
    polymer_coords : (nsteps,ntrajectories,N) ndarray
        Array of polymer coordinates at each time step for each trajectory


    Raises
    ------
    ValueError
        If nsteps is not an integer
    """
    if not type(nsteps) == int:
        raise ValueError("nsteps must be an integer")
    
    # compute the eigensystem of the spring matrix
    Qmat,eigvals = Get_eigensystem(N)
    
    # initialize containers for the polymer coordinates and current Rouse mode vectors
    key = jax.random.PRNGKey(seed)
    a_samples = np.zeros((nsteps,n_trajectories,N-1))
    mean_0,covar_0 = jnp.zeros((n_trajectories,N-1)),(D/k/eigvals)*jnp.ones((n_trajectories,N-1))
    
    # generate Gaussian samples for the Rouse modes (faster to generate a-priori on the GPU than at each time step)
    gaussian_samples = jax.random.normal(key,(n_trajectories,nsteps,N-1))
    a_samples[0] =  mean_0+gaussian_samples[:,0]*jnp.sqrt(covar_0) # set the initial conditions
    
    for i in range(1,nsteps):
        # propagate the Rouse modes forward in time and store
        new_mean,new_cov = forward_d_vmap(a_samples[i-1],dt,k,eigvals,D)
        a_samples[i] = new_mean+gaussian_samples[:,i]*jnp.sqrt(new_cov)
    
    # convert the Rouse modes to physical coordinates and output    
    polymer_coords = jnp.einsum("ijk,lk->ijl",a_samples,Qmat) 
    times = np.arange(0,len(polymer_coords))*dt
    return times,polymer_coords

def Generate_measurements(traj,w,measurement_error,seed=0):
    """Generate measurements of enhancer-promoter distances from a polymer trajectory. The measurements are generated by projecting the polymer coordinates onto a vector w and adding Gaussian noise.

    Parameters
    ----------
    traj : (nsteps,ntrajectories,N) ndarray
        Array of polymer coordinates at each time step for each trajectory
    w : (N) ndarray
        Vector to project the polymer coordinates onto. Defined by the positions on the polymer that are observed. For example, an observation of the difference between positions of bead 2 and 3 in a five bead polymer would be represented by w = [0,1,-1,0,0].
    measurement_errors : float
        Standard deviation of the Gaussian measurement noise
    seed : int, optional
        seed for the random number generator, by default 0

    Returns
    -------
    _type_
        _description_
    """
    #add localization errors
    key = jax.random.PRNGKey(seed)
    loc_errs = jax.random.normal(key,traj.shape)*measurement_error#measurement_errors[None,:,None]
    traj_w_error = traj+loc_errs
    
    #project the polymer coordinates onto the vector w
    ep_trajs = jnp.einsum("ijk,k->ij",traj,w).T
    ep_trajs_w_error = jnp.einsum("ijk,k->ij",traj_w_error,w).T
    return ep_trajs,ep_trajs_w_error