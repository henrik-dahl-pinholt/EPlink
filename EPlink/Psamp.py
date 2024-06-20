from jax import numpy as jnp
import numpy as np
import jax

def Get_eigensystem(N):  
    """_summary_

    Parameters
    ----------
    N : _type_
        _description_

    Returns
    -------
    A tuple of the form (Qmat,eigvals)

    
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
    return jnp.einsum("j,ij->i",vector,Qmat)
def convert_modes_M(matrix,Qmat):
    return jnp.einsum("jk,ij,lk->il",matrix,Qmat,Qmat)
def convert_modes_V_ep(vector,M_matrix):
    return jnp.einsum("j,j->",vector,M_matrix)
def convert_modes_M_ep(matrix,M_matrix):
    return jnp.einsum("jk,j,k->",matrix,M_matrix,M_matrix)


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