from jax import numpy as jnp
import numpy as np
import jax
from tqdm import tqdm
"""This module contains code to simulate polymer dynamics, generate measurements of enhancer-promoter distances, and infer polymer positions from two-locus microscopy measurements. The polymer is modeled as a chain of beads connected by springs with dynamics described by the Rouse model."""
def Get_eigensystem(N):  
    """Generate the eigenvectors and eigenvalues of the spring matrix in the Fourier basis. These are used to convert between physical coordinates and the Rouse modes in which the dynamics are diagonal.

    Parameters
    ----------
    N : int
        The number of beads in the polymer

    Returns
    -------
    Qmat : (N,N-1) array
        The N-1 eigenvectors of the spring matrix in the Fourier basis.
    eigvals : (N-1) array
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
    vector : (N) array
        Vector of bead coordinates
    Qmat : (N,N-1) array
        The N-1 eigenvectors of the spring matrix in the Fourier basis as output by Get_eigensystem.

    Returns
    -------
    (N-1) array
        The vector in the Rouse modes.
    """    
    return jnp.einsum("j,ij->i",vector,Qmat)
def convert_modes_M(matrix,Qmat):
    """Convenience function to convert matrix from physical coordinates to Rouse modes.

    Parameters
    ----------
    matrix : (N,N) array
        Matrix which operates on bead coordinates
    Qmat : (N,N-1) array
        The N-1 eigenvectors of the spring matrix in the Fourier basis as output by Get_eigensystem.

    Returns
    -------
    (N-1,N-1) array
        The matrix in the Rouse modes basis.
    """    
    return jnp.einsum("jk,ij,lk->il",matrix,Qmat,Qmat)
def convert_modes_V_ep(vector,M_vector):
    """Convenience function to convert a vector from Rouse modes to enhancer promoter distance, i.e. a projection of the physical coordinates. This transformation involves two transformations: first a basis transformation to physical coordinates, then a projection to enhancer promoter distance. This is encompassed in the M_vector which must be the left product of the sought projection vector w and the matrix Qmat, i.e. M = w.T Q.
     

    Parameters
    ----------
    vector : (N-1) array
        Vector in Rouse modes basis
    M_vector : (N-1) array
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
    matrix : (N-1,N-1) array
        Matrix in Rouse modes basis
    M_vector : (N-1) array
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
    mu : (N-1) jax array
        Rouse mode vector at time 0
    timestep : float
        Time step for the propagation
    k : float
        Spring constant of the Rouse model
    eigvals : jax array
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


@jax.jit
def Propagate_Forward(mu,Sigma,timestep,k,D,eigvals):
    """Propagate an arbitrary Gaussian distribution Rouse modes Gaussian distribution of Rouse modes forward in time. To be contrasted with Propagate_Forward_diagonal which is specialized for deterministic initial conditions.

    Parameters
    ----------
    mu : (N-1) jax array
        Rouse mode mean vector at time 0
    Sigma : (N-1,N-1) jax array
        Rouse mode covariance matrix at time 0
    timestep : float
        Time step for the propagation
    k : float
        Spring constant of the Rouse model
    D : float
        Diffusion constant of the Rouse model
    eigvals : jax array
        Eigenvalues of the spring matrix in the Fourier basis as output by Get_eigensystem

    Returns
    -------
    mu_out : (N-1) jax array
        Rouse mode mean vector after the timestep
    var_out : (N-1,N-1) jax array   
        Rouse mode covariance matrix after the timestep
    """
    diag_res = jnp.eye(mu.shape[-1])*((D/(k*eigvals))*(1-jnp.exp(-2*k*eigvals*timestep)))
    eigexp = jnp.exp(-k*eigvals*timestep)
    mu_out = mu*eigexp
    var_out = diag_res+Sigma*jnp.outer(eigexp,eigexp)
    return mu_out, var_out

@jax.jit
def Update(z,w,x,P,R):
    """Update the mean and covariance of a Gaussian distribution (initially mean x, covariance P) given a measurement z obtained by projecting with w and adding Gaussian noise with variance R. z is projection to a float through w. The measurement is assumed to be a linear projection of the state x with a projection vector w.

    Parameters
    ----------
    z : float
        Measurement
    w : jax array of shape x.shape
        Projection vector
    x : jax array
        Mean of the Gaussian distribution
    P : jax array
        Covariance of the Gaussian distribution
    R : float
        Variance of the measurement noise

    Returns
    -------
    x_hat : jax array
        Updated mean of the Gaussian distribution
    P_hat : jax array
        Updated covariance of the Gaussian distribution
    """
    innovation = z-w@x
    S = w@P@w.T+R
    K = P@w.T/S
    x_hat = x+K*innovation
    P_hat = (jnp.eye(len(x))-jnp.diag(K*w))@P
    return x_hat,P_hat


Update_vmap = jax.vmap(Update,in_axes=(0,None,0,0,0))
Update_vmap.__doc__ = """Vectorized version of Update. Takes the same arguments but the input mus must now have a leading axis to allow for inputting a batch of Gaussian initial conditions.
    Parameters
    ----------
    z : (batch_size) jax array
    
"""
Propagate_Forward_vmap = jax.vmap(Propagate_Forward,in_axes=(0,0,None,None,None,None))
Propagate_Forward_vmap.__doc__ = """Vectorized version of Propagate_Forward. Takes the same arguments but the input mus must now have a leading axis to allow for inputting a batch of Gaussian initial conditions.
    Parameters
    ----------
    mus : (batch_size,N-1) jax array
        Rouse mode mean vectors at time 0
    Sigmas : (batch_size,N-1,N-1) jax array
        Rouse mode covariance matrices at time 0
    timestep : float
        Time step for the propagation
    k : float
        Spring constant of the Rouse model
    D : float
        Diffusion constant of the Rouse model
    eigvals : jax array
        Eigenvalues of the spring matrix in the Fourier basis as output by Get_eigensystem
    Returns
    -------
    mu_out : (batch_size,N-1) jax array
        Rouse mode mean vectors after the timestep
    var_out : (batch_size,N-1,N-1) jax array
        Rouse mode covariance matrices after the timestep        
"""

forward_d_vmap = jax.vmap(Propagate_Forward_diagonal,in_axes=(0,None,None,None,None))
forward_d_vmap.__doc__ = """Vectorized version of Propagate_Forward_diagonal. Takes the same arguments but the input mus must now have a leading axis to allow for inputting a batch of deterministic initial conditions.
    Parameters
    ----------
    mus : (batch_size,N-1) jax array
        Rouse mode vectors at time 0
    timestep : float
        Time step for the propagation
    k : float
        Spring constant of the Rouse model
    eigvals : jax array
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

def Generate_trajectory(nsteps,dt,n_trajectories,k,D,N,seed=0,verbose=False):
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
    verbose : bool, optional
        Print progress of the simulation, by default False

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
    if verbose:
        iterator = tqdm(list(range(1,nsteps)))
    else:
        iterator = range(1,nsteps)
    for i in iterator:
        # propagate the Rouse modes forward in time and store
        new_mean,new_cov = forward_d_vmap(a_samples[i-1],dt,k,eigvals,D)
        a_samples[i] = new_mean+gaussian_samples[:,i]*jnp.sqrt(new_cov)
    
    # convert the Rouse modes to physical coordinates and output    
    polymer_coords = jnp.einsum("ijk,lk->ijl",a_samples,Qmat) 
    times = np.arange(0,len(polymer_coords))*dt
    return times,polymer_coords

def Generate_measurements(traj,w,measurement_errors,seed=0):
    """Generate measurements of enhancer-promoter distances from a polymer trajectory. The measurements are generated by projecting the polymer coordinates onto a vector w and adding Gaussian noise.

    Parameters
    ----------
    traj : (nsteps,ndim,N) ndarray
        Array of polymer coordinates at each time step 
    w : (N) ndarray
        Vector to project the polymer coordinates onto. Defined by the positions on the polymer that are observed. For example, an observation of the difference between positions of bead 2 and 3 in a five bead polymer would be represented by w = [0,1,-1,0,0].
    measurement_errors : (ndim) ndarray
        Standard deviation of the Gaussian measurement noise
    seed : int, optional
        seed for the random number generator, by default 0

    Returns
    -------
    ep_trajs : (nsteps,ntrajectories) ndarray
        Array of subtracted enhancer-promoter positions at each time step for each trajectory
    ep_trajs_w_error : (nsteps,ntrajectories) ndarray
        Same as ep_trajs with added Gaussian noise
    """
    #add localization errors
    key = jax.random.PRNGKey(seed)
    loc_errs = jax.random.normal(key,traj.shape)*measurement_errors[None,:,None]
    traj_w_error = traj+loc_errs
    
    #project the polymer coordinates onto the vector w
    ep_trajs = jnp.einsum("ijk,k->ij",traj,w).T
    ep_trajs_w_error = jnp.einsum("ijk,k->ij",traj_w_error,w).T
    return ep_trajs,ep_trajs_w_error


class ForwardFilter:
    def __init__(self,N,k,D,measurements,observation_times,measurement_errors,w):
        # Store the model parameters
        self.D,self.k,self.N = D,k,N
        self.w = w
        self.measurements = measurements
        self.observation_times = observation_times
        try:
            _ = len(measurement_errors)
        except:
            measurement_errors = jnp.ones(len(measurements))*measurement_errors
        self.measurement_errors = measurement_errors
        
        # Initialize the prior ensemble
        self.Qmat,self.eigvals = Get_eigensystem(N)
        self.M_matrix =  jnp.einsum("k,kj->j",w,self.Qmat)
        prior_cov = (D/k/self.eigvals)*jnp.ones(N-1)
    
        # Initialize containers for the results
        self.prior_means = np.zeros((len(measurements),len(observation_times),N-1))
        self.prior_covs = np.zeros((len(measurements),len(observation_times),N-1,N-1))
        self.prior_covs[:,0] = jnp.diag(prior_cov)
        self.post_means = np.zeros((len(measurements),len(observation_times),N-1))
        self.post_covs = np.zeros((len(measurements),len(observation_times),N-1,N-1))
        
        self.hasrun = False
        
    def Run(self):
        # Loop over the observations
        for i in tqdm(list(range(len(self.observation_times)))):
            # Update to get the posterior ensemble
            # self.post_means[:,i],self.post_covs[:,i] = Update_vmap(self.prior_means[:,i],self.prior_covs[:,i],self.measurements[:,i],2*self.measurement_error**2,self.M_matrix)
            self.post_means[:,i],self.post_covs[:,i] = Update_vmap(self.measurements[:,i],self.M_matrix,self.prior_means[:,i],self.prior_covs[:,i],2*self.measurement_errors**2)

            # Propagate the posterior ensemble forward
            if i < len(self.observation_times)-1:
                self.prior_means[:,i+1],self.prior_covs[:,i+1] = Propagate_Forward_vmap(self.post_means[:,i],self.post_covs[:,i],self.observation_times[i+1]-self.observation_times[i],self.k,self.D,self.eigvals)
        self.hasrun = True
            
    def Get_Posteriors(self):
        return jnp.einsum("ijk,k->ij",self.post_means,self.M_matrix),jnp.einsum("ijkl,k,l->ij",self.post_covs,self.M_matrix,self.M_matrix)
    def Get_priors(self):
        return jnp.einsum("ijk,k->ij",self.prior_means,self.M_matrix),jnp.einsum("ijkl,k,l->ij",self.prior_covs,self.M_matrix,self.M_matrix)
    def __getitem__(self,time):
        ind = np.max(np.argwhere(self.observation_times<time)) if time>self.observation_times[0] else 0
        t_closest = self.observation_times[ind]
        t_diff = time-t_closest
        return Propagate_Forward_vmap(self.post_means[:,ind],self.post_covs[:,ind],t_diff,self.k,self.D,self.eigvals)
    def Convert_to_observations(self,means,covs):
        return jnp.einsum("ijk,k->ij",means,self.M_matrix),jnp.einsum("ijkl,k,l->ij",covs,self.M_matrix,self.M_matrix)
    def Get_filter_at_continous_times(self,times,convert=True):            
        if not self.hasrun:
            print("Running the filter at the measurements")
            self.Run()
        else:
            print("Computing the filter at continuous times")
            ms,covs = np.zeros((len(self.measurements),len(times),self.N-1)),np.zeros((len(self.measurements),len(times),self.N-1,self.N-1))
            for i in tqdm(range(len(times))):
                ms[:,i],covs[:,i] = self[times[i]]
        if convert:
            return self.Convert_to_observations(ms,covs)
        else:
            return ms,covs
            
class PosteriorSampler:
    def __init__(self,N,k,D,measurements,observation_times,measurement_errors,w):
        self.D,self.k,self.N = D,k,N
        self.measurements = measurements
        self.observation_times = observation_times
        self.measurement_error = measurement_errors
        self.Qmat,self.eigvals = Get_eigensystem(N)
        self.M_matrix =  jnp.einsum("k,kj->j",w,self.Qmat)
        
        print("Running the forward filter")
        self.forward_filter = ForwardFilter(N,k,D,measurements,observation_times,measurement_errors,w)
        self.forward_filter.Run()
        
    def sample(self,nsamples,dt,seed=0,compute_density=False,verbose=False):
        if dt>1/self.k:
            # raise warning that the time step is too large
            print("Warning: The time step may be too large for the sampling to be accurate. It may blow up")
        nsteps = int(self.observation_times[-1]/dt)
        times = np.arange(0,nsteps)[::-1]*dt 
        sub_samps = np.zeros((len(self.measurements),nsamples,len(times)))
        if compute_density:
            means = np.zeros((len(self.measurements),len(times)))
            covs = np.zeros((len(self.measurements),len(times)))
        
        init_mean,init_cov = self.forward_filter.post_means[:,-1],self.forward_filter.post_covs[:,-1]
        subkey = jax.random.PRNGKey(seed)
        
        # sub_samps[:,:,0] = np.array([jax.random.multivariate_normal(subkey,init_mean[k],init_cov[k],shape=(nsamples,)) for k in range(len(init_mean))])
        curr_val = np.array([jax.random.multivariate_normal(subkey,init_mean[k],init_cov[k],shape=(nsamples,)) for k in range(len(init_mean))])
        sub_samps[:,:,0] = curr_val@self.M_matrix.T
        if compute_density:
            means[:,0],covs[:,0] = init_mean@self.M_matrix.T,jnp.einsum("ikj,k,j->i",init_cov,self.M_matrix,self.M_matrix)
            curr_mean,curr_cov = init_mean,init_cov
        
        print("Computing the paths")
        if verbose:
            iterator = tqdm(list(range(1,len(times))))
        else:
            iterator = range(1,len(times))
        for i in iterator:
            fmean,fvar =  self.forward_filter[times[i-1]]
            invdrift = jnp.array([jnp.linalg.inv(f) for f in fvar])
            # fmean_repeated = np.array([fmean for _ in range(nsamples)]).swapaxes(0,1)
            
            key,subkey = jax.random.split(subkey)
            # gauss_samp = jax.random.normal(key,sub_samps[:,:,i].shape)
            gauss_samp = jax.random.normal(key,curr_val.shape)
            
            F_t = np.array([jnp.diag(self.eigvals*self.k) for _ in range(len(fvar))])
            deterministic_term = jnp.einsum("ikl,iml->imk",F_t,curr_val)
            
            curr_val = curr_val+dt*(deterministic_term-2*self.D*jnp.einsum("ikl,ilm->ikm",(curr_val-fmean[:,None,:]),invdrift))+jnp.sqrt(2*self.D*dt)*gauss_samp
            if compute_density:
                
                GQGPF = jnp.einsum("ij,kjm->kim",jnp.eye(self.N-1)*2*self.D,invdrift)
                drift_mat = F_t-GQGPF
                curr_cov = curr_cov+dt*(jnp.einsum("ikj,ijm->ikm",drift_mat,curr_cov)+jnp.einsum("ikj,ijm->ikm",curr_cov,drift_mat)+jnp.eye(self.N-1)*2*self.D)
                
                deterministic_term = jnp.einsum("ikl,il->ik",F_t,curr_mean)
                data_term = 2*self.D*jnp.einsum("ikl,il->ik",invdrift,curr_mean-fmean)
                curr_mean = curr_mean +dt*(deterministic_term-data_term)
                means[:,i],covs[:,i] = curr_mean@self.M_matrix.T,jnp.einsum("ikj,k,j->i",curr_cov,self.M_matrix,self.M_matrix)
                
                
                
            # sub_samps[:,:,i] = sub_samps[:,:,i-1]+dt*(self.eigvals*self.k-2*self.D*jnp.einsum("ikl,ilm->ikm",(sub_samps[:,:,i-1]-fmean[:,None,:]),invdrift))+jnp.sqrt(2*self.D*dt)*gauss_samp
            sub_samps[:,:,i] = curr_val@self.M_matrix.T
        # print("Converting the paths to observations")
        # output = np.zeros((len(self.measurements),nsamples,len(times)))    
        # for i in tqdm(list(range(len(times)))):    
        #     # output[:,:,i] = jnp.einsum("ijk,k->ij",sub_samps[:,:,i],self.M_matrix)
        #     output[:,:,i] = sub_samps[:,:,i]@self.M_matrix.T
        # return times,output
        if not compute_density:
            return times,sub_samps
        else:
            return times,sub_samps,means,covs