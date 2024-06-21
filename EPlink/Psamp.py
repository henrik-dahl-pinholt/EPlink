from jax import numpy as jnp
import numpy as np
import jax

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
    eigexp = jnp.exp(-k*eigvals*timestep)
    return mu*eigexp,(D/(k*eigvals))*(1-jnp.exp(-2*k*eigvals*timestep))
forward_d_vmap = jax.vmap(Propagate_Forward_diagonal,in_axes=(0,None,None,None,None))

def Gen_ss_samp(N,D,k,nsamples):
    Qmat,eigvals = Get_eigensystem(N)
    gaussian_samples = np.random.normal(0,1,size=(nsamples,N-1))
    mean_0,covar_0 = np.zeros((nsamples,N-1)),(D/k/eigvals)*np.ones((nsamples,N-1))
    Rouse_confs = mean_0+gaussian_samples*np.sqrt(covar_0)
    confs = np.einsum("jk,lk->jl",Rouse_confs,Qmat) 
    return confs

def Generate_trajectory(nsteps,dt,n_trajectories,k,D,N,seed=0):
    if not type(nsteps) == int:
        raise ValueError("nsteps must be an integer")
    Qmat,eigvals = Get_eigensystem(N)
    key = jax.random.PRNGKey(seed)
    T = nsteps*dt
    # nsteps = int(T/dt)
    a_samples = np.zeros((nsteps,n_trajectories,N-1))
    mean_0,covar_0 = jnp.zeros((n_trajectories,N-1)),(D/k/eigvals)*jnp.ones((n_trajectories,N-1))
    
    gaussian_samples = jax.random.normal(key,(n_trajectories,nsteps,N-1))
    a_samples[0] =  mean_0+gaussian_samples[:,0]*jnp.sqrt(covar_0)
    for i in range(1,nsteps):
        new_mean,new_cov = forward_d_vmap(a_samples[i-1],dt,k,eigvals,D)
        
        a_samples[i] = new_mean+gaussian_samples[:,i]*jnp.sqrt(new_cov)
    polymer_coords = jnp.einsum("ijk,lk->ijl",a_samples,Qmat) 
    return np.arange(0,len(polymer_coords))*dt,polymer_coords

def Generate_measurements(traj,w,measurement_errors,seed=0):
    key = jax.random.PRNGKey(seed)
    loc_errs = jax.random.normal(key,traj.shape)*measurement_errors[None,:,None]
    traj_w_error = traj+loc_errs
    ep_trajs = jnp.einsum("ijk,k->ij",traj,w).T
    ep_trajs_w_error = jnp.einsum("ijk,k->ij",traj_w_error,w).T
    return ep_trajs,ep_trajs_w_error