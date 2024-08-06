
import jax

import scipy
from jax import numpy as jnp
from jax import scipy as jscipy
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.optimize import minimize
"""This module contains functions to infer polymerase loadings, compute likelihoods for MS2-type datasets, and simulate simple transcription models
"""

def Generate_multistate_trajectory(
    k_on, k_off, n_states, input_profile, dt, verbose=False, initcond=None
):
    n_steps = len(input_profile)
    onmatrix, offmatrix = np.zeros((n_states, n_states)), np.zeros((n_states, n_states))
    indices = np.arange(n_states - 1)
    onmatrix[indices, indices + 1] = k_off
    offmatrix[indices, indices + 1] = k_off
    onmatrix[indices + 1, indices] = k_on
    onmatrix[np.arange(n_states), np.arange(n_states)] = -np.sum(onmatrix, axis=0)
    offmatrix[np.arange(n_states), np.arange(n_states)] = -np.sum(offmatrix, axis=0)

    on_prop = scipy.linalg.expm(onmatrix * dt)
    off_prop = scipy.linalg.expm(offmatrix * dt)

    states = np.zeros((n_steps + 1, n_states))
    trajectory = np.zeros(n_steps + 1).astype(int)
    if int(input_profile[0]) == 1 and initcond is None:
        vec = np.zeros(n_states)
        vec[0] = 1 / k_on
        if n_states > 2:
            vec[1:] = np.array(
                [k_on ** (k - 2) / k_off ** (k - 1) for k in range(2, n_states + 1)]
            )
        else:
            vec[1:] = np.array([1 / k_off])
        vec = vec / np.sum(vec)
        states[0] = vec
        trajectory[0] = np.random.choice(np.arange(n_states), p=vec)
    else:
        if initcond is None:
            states[0, 0] = 1
            trajectory[0] = 0
        else:
            states[0] = initcond
            trajectory[0] = np.random.choice(np.arange(n_states), p=initcond)
    if verbose:
        iterator = tqdm(list(range(n_steps)))
    else:
        iterator = range(n_steps)
    for i in iterator:
        if input_profile[i] == 1:
            propagator = on_prop 
        else:
            propagator = off_prop
        states[i + 1] = np.dot(propagator, states[i])
        p_choice = propagator[:, trajectory[i]]
        trajectory[i + 1] = np.random.choice(
            n_states, p=p_choice
        )  

    return states[1:], trajectory[1:]

def Gen_MS2_measurement(pol2_rates,params,dt,random_pol2=False):
    w,kappa,noise_std = params
    kernel_func = MS2_kernel(jnp.arange(w)[::-1],w*kappa)
    
    trajlen = len(pol2_rates)
    if not random_pol2:
        pol2_nums = pol2_rates
    else:
        pol2_nums = np.random.poisson(pol2_rates)
    MS2_signal = jnp.convolve(pol2_nums,kernel_func,mode="valid")*dt
    ts_MS2 = np.arange(w-1,trajlen)*dt
    ts = np.arange(trajlen)*dt
    signal = MS2_signal + np.random.normal(0,noise_std,len(MS2_signal))
    return (ts,pol2_nums),(ts_MS2,MS2_signal,signal)


@jax.jit
def MS2_kernel(t,tau):
    return jnp.heaviside(-(t-tau),1)*t/tau + jnp.heaviside(t-tau,0)

@jax.jit
def Sigma_f_SE(t,T,l,sigma):
    """MS2 signal covariance function for a squared exponential (SE) kernel

    Parameters
    ----------
    t : float
        Time difference between two points
    T : float
        Residence time of pol2 on the gene
    l : float
        Length scale of the SE kernel
    sigma : float
        Amplitude of the SE kernel

    Returns
    -------
    float
        Covariance between two points separated by time t
    """
    abs_t = jnp.abs(t)
    term1 = l*jnp.sqrt(2/jnp.pi)*(jnp.exp(-(abs_t+T)**2/2/l**2)+jnp.exp(-(abs_t-T)**2/2/l**2)-2*jnp.exp(-abs_t**2/2/l**2))
    term2 = -2*abs_t*jscipy.special.erf(abs_t/jnp.sqrt(2)/l)+(abs_t-T)*jscipy.special.erf((abs_t-T)/jnp.sqrt(2)/l)+(abs_t+T)*jscipy.special.erf((abs_t+T)/jnp.sqrt(2)/l)
    return sigma**2*l*jnp.sqrt(jnp.pi/2)*(term1+term2)

@jax.jit
def Sigma_f_E(t,T,l,sigma):
    """MS2 signal covariance function for a exponential (E) kernel

    Parameters
    ----------
    t : float
        Time difference between two points
    T : float
        Residence time of pol2 on the gene
    l : float
        Length scale of the E kernel
    sigma : float
        Amplitude of the E kernel

    Returns
    -------
    float
        Covariance between two points separated by time t
    """
    abs_t = jnp.abs(t)
    term1 = l**2*(jnp.exp(-(abs_t+T)/2/l)-jnp.exp(-(abs_t-T)/2/l))**2
    term2 = (jnp.exp(-(abs_t+T)/l)+jnp.exp((abs_t-T)/l)-2*jnp.exp(-abs_t/l))*l**2+2*l*(T-abs_t)
    result = sigma**2*(jnp.heaviside(abs_t-T,1)*term1+jnp.heaviside(T-abs_t,0)*term2)
    return jnp.where(jnp.isnan(result) | jnp.isinf(result), 0.0, result)


@jax.jit
def Sigma_f_D(t,T,sigma):
    """MS2 signal covariance function for a delta function (D) kernel

    Parameters
    ----------
    t : float
        Time difference between two points
    T : float
        Residence time of pol2 on the gene
    sigma : float
        Amplitude of the D kernel

    Returns
    -------
    float
        Covariance between two points separated by time t
    """
    abs_t = jnp.abs(t)
    
    return (T-abs_t)*sigma**2*jnp.heaviside(T-abs_t,1)



@jax.jit
def Sigma_fx_SE(t,T,l,sigma):
    """MS2 signal covariance function between the loaded polymerases (x) and the MS2 signal (f) for a squared exponential (SE) kernel

    Parameters
    ----------
    t : float
        Time difference between two points
    T : float
        Residence time of pol2 on the gene
    l : float
        Length scale of the SE kernel
    sigma : float
        Amplitude of the SE kernel

    Returns
    -------
    float
        Covariance between two points separated by time t
    """
    return l*jnp.sqrt(jnp.pi/2)*sigma**2*(jscipy.special.erf(t/jnp.sqrt(2)/l)-jscipy.special.erf((t-T)/jnp.sqrt(2)/l))

@jax.jit
def Sigma_fx_E(t,T,l,sigma):
    """MS2 signal covariance function between the loaded polymerases (x) and the MS2 signal (f) for a exponential (E) kernel

    Parameters
    ----------
    t : float
        Time difference between two points
    T : float
        Residence time of pol2 on the gene
    l : float
        Length scale of the E kernel
    sigma : float
        Amplitude of the E kernel

    Returns
    -------
    float
        Covariance between two points separated by time t
    """
    t = -t
    term1 = -1+jnp.exp(T/l)-(-1+jnp.exp(T/l))*jnp.heaviside(-t,1)
    term2 = jnp.exp((2*t+T)/l)*(-1-jnp.exp(T/l)*(-1+jnp.heaviside(t+T,1))+jnp.heaviside(t+T,1)-jnp.heaviside(-t,1)*jnp.heaviside(t+T,1))
    term3 = -jnp.heaviside(-t,1)*jnp.heaviside(t+T,1)+2*jnp.exp((t+T)/l)*jnp.heaviside(-t,1)*jnp.heaviside(t+T,1)
    result = jnp.exp(-(t+T)/l)*l*sigma**2*(term1+term2+term3)
    return jnp.where(jnp.isnan(result) | jnp.isinf(result), 0.0, result)

@jax.jit
def Sigma_fx_D(t,T,sigma):
    """MS2 signal covariance function between the loaded polymerases (x) and the MS2 signal (f) for a delta (D) kernel

    Parameters
    ----------
    t : float
        Time difference between two points
    T : float
        Residence time of pol2 on the gene
    sigma : float
        Amplitude of the D kernel

    Returns
    -------
    float
        Covariance between two points separated by time t
    """
    return sigma**2*jnp.heaviside(t,1)*jnp.heaviside(-t+T,1)


@jax.jit
def Sigma_x_SE(t,l,sigma):
    """MS2 signal covariance function between the loaded polymerases (x) for a squared exponential (SE) kernel

    Parameters
    ----------
    t : float
        Time difference between two points
    l : float
        Length scale of the SE kernel
    sigma : float
        Amplitude of the SE kernel

    Returns
    -------
    float
        Covariance between two points separated by time t
    """
    return sigma**2*jnp.exp(-t**2/2/l**2)

@jax.jit
def Sigma_x_E(t,l,sigma):
    """MS2 signal covariance function between the loaded polymerases (x) for a exponential (E) kernel

    Parameters
    ----------
    t : float
        Time difference between two points
    l : float
        Length scale of the E kernel
    sigma : float
        Amplitude of the E kernel

    Returns
    -------
    float
        Covariance between two points separated by time t
    """
    return sigma**2*jnp.exp(-jnp.abs(t)/2/l)


@jax.jit
def Sigma_x_D(t,sigma):
    """MS2 signal covariance function between the loaded polymerases (x) for a exponential (D) kernel

    Parameters
    ----------
    t : float
        Time difference between two points
    sigma : float
        Amplitude of the D kernel

    Returns
    -------
    float
        Covariance between two points separated by time t
    """
    return sigma**2*jnp.isclose(t,0)

@jax.jit
def inverse_prod(matrix,vector):
    """Computes the product of the inverse of a matrix and a vector

    Parameters
    ----------
    matrix : array
        Matrix to be inverted
    vector : array
        Vector to be multiplied

    Returns
    -------
    array
        Result of the product
    """
    return jscipy.linalg.solve(matrix,vector,assume_a="pos")

class MS2_GPDC:
    def __init__(self,kernel):
        if kernel=="SE":
            self.Kx = Sigma_x_SE
            self.Kf = Sigma_f_SE
            self.Kxy = Sigma_fx_SE
        elif kernel=="E":
            self.Kx = Sigma_x_E
            self.Kf = Sigma_f_E
            self.Kxy = Sigma_fx_E
        elif kernel =="D":
            self.Kx = Sigma_x_D
            self.Kf = Sigma_f_D
            self.Kxy = Sigma_fx_D
        else:
            raise ValueError(f"Kernel {kernel} not recognized")
        
    def Get_Predfunc(self,params,t_data,data):
        params = jnp.abs(params)
        
        posterior_mean = jnp.mean(data)
        prior_mean = jnp.mean(data)/params[0]
        
        t1s,t2s = jnp.meshgrid(t_data,t_data)
        K_y = self.Kf(t1s-t2s,*params[:-1])+params[-1]**2*jnp.eye(len(t_data))
        mu_func = jax.vmap(lambda t: prior_mean+self.Kxy(t_data-t,*params[:-1])@inverse_prod(K_y,data-posterior_mean))
        cov_func = jax.vmap(lambda t1,t2: self.Kx(t1-t2,*params[1:-1])-self.Kxy(t1-t_data,*params[:-1])@inverse_prod(K_y,self.Kxy(t_data-t2,*params[:-1])) )
        MS2_mu_func = jax.vmap(lambda t: posterior_mean+self.Kf(t_data-t,*params[:-1])@inverse_prod(K_y,data-posterior_mean))
        MS2_cov_func = jax.vmap(lambda t1,t2: self.Kf(t1-t2,*params[:-1])-self.Kf(t1-t_data,*params[:-1])@inverse_prod(K_y,self.Kf(t_data-t2,*params[:-1])) )
        return mu_func,cov_func,MS2_mu_func,MS2_cov_func
    
    def Generate_dataset(self,params,tmin,tmax,ntimepoints,regularization = 1e-2):   
        params = jnp.abs(params)
        
        key = jax.random.PRNGKey(np.random.randint(0,1000))
        ts = jnp.linspace(tmin,tmax,ntimepoints)
        dt = ts[1]-ts[0]
        w = int(params[0]//dt)
        
        ts_y = ts[w-1:]
        t1s,t2s  = np.meshgrid(ts,ts)
        K_x = self.Kx(t1s-t2s,*params[1:-2])+regularization**2*jnp.eye(len(ts))
        K_x_cholesky = jnp.linalg.cholesky(K_x)
        conv_filter = jnp.ones(w)

        xs = K_x_cholesky@jax.random.normal(key,shape=(len(ts),))
        ys = jnp.convolve(xs,conv_filter,mode='valid')*(ts[1]-ts[0])+params[-1]*jax.random.normal(key,shape=(len(ts)-w+1,))
        return (ts,xs),(ts_y,ys)
    
    def Marginal_LLH(self,params,t_data,data):
        params = jnp.abs(params)
        posterior_mean = jnp.mean(data)
        
        t1s,t2s = jnp.meshgrid(t_data,t_data)
        K_f = self.Kf(t1s-t2s,*params[:-1]) 
        K_y = K_f+params[-1]**2*jnp.eye(len(t_data))
        K_y_chol = jnp.linalg.cholesky(K_y)
        log_det = 2*jnp.sum(jnp.log(jnp.abs(jnp.diag(K_y_chol))))
        n = len(t_data)
        
        return -(n/2)*jnp.log(2*jnp.pi)-0.5*log_det-0.5*(data-posterior_mean)@inverse_prod(K_y,(data-posterior_mean))
    def Sum_values_above_nsig(self,params,t_data,data,ntestpoints=30,nsig=2.):
        params = jnp.abs(params)
        posterior_mean = jnp.mean(data)
        prior_mean = jnp.mean(data)/params[0]
    
        t1s,t2s = jnp.meshgrid(t_data,t_data)
        K_y = self.Kf(t1s-t2s,*params[:-1])+params[-1]**2*jnp.eye(len(t_data))
        mu_func = jax.vmap(lambda t: prior_mean+self.Kxy(t_data-t,*params[:-1])@inverse_prod(K_y,data-posterior_mean))
        cov_func = jax.vmap(lambda t1,t2: self.Kx(t1-t2,*params[1:-1])-self.Kxy(t1-t_data,*params[:-1])@inverse_prod(K_y,self.Kxy(t_data-t2,*params[:-1])) )
        
        test_points = jnp.linspace(t_data[0]*1.02,t_data[-1]*0.98,ntestpoints)
        means = mu_func(test_points)
        stds = jnp.sqrt(cov_func(test_points,test_points))
        
        return -jnp.sum(jax.nn.relu(-means+nsig*stds))+1

    def Sum_agreement(self,params,epsilon,t_data,data):
        params = jnp.abs(params)
        posterior_mean = jnp.mean(data)
    
        t1s,t2s = jnp.meshgrid(t_data,t_data)
        K_y = self.Kf(t1s-t2s,*params[:-1])+params[-1]**2*jnp.eye(len(t_data))
        MS2_mu_func = jax.vmap(lambda t: posterior_mean+self.Kf(t_data-t,*params[:-1])@inverse_prod(K_y,data-posterior_mean))
                                                                                          
        return -jnp.sum(jax.nn.relu(-(epsilon-jnp.abs(MS2_mu_func(t_data)-data))))+1
    
    def callback(self,xk,run_silent=False):
        if not run_silent:
            print(xk)
        if not self.deterministic:
            ind_choices = np.random.choice(len(self.Y),self.batch_size,replace=False)
            self.X_batch, self.Y_batch = [],[]
            for i in ind_choices:
                xdat,ydat = self.X[i],self.Y[i]
                self.X_batch.append(xdat)
                self.Y_batch.append(ydat)
    
    def Fit_params(self,X,Y,initial_guess,epsilon,batch_size=10):
        if batch_size>len(X):
            batch_size = len(X)
        self.batch_size = batch_size            
        self.X,self.Y = X,Y
        if len(X)==batch_size: # deterministic gradient descent
            print("Deterministic")
            self.X_batch,self.Y_batch = X,Y
            self.deterministic = True
        else:
            print("Stochastic")
            self.deterministic = False
            self.callback(initial_guess,run_silent=True)
        
        minfunc = lambda params: jnp.mean(jnp.array([-self.Marginal_LLH(params,xdat,ydat) for xdat,ydat in zip(self.X_batch,self.Y_batch)]))
        grad = jax.grad(minfunc)
        con_fun = lambda params: jnp.mean(jnp.array([self.Sum_values_above_nsig(params,xdat,ydat) for xdat,ydat in zip(self.X_batch,self.Y_batch)]))
        grad_con = jax.grad(con_fun)
        con_fun2 = lambda params: jnp.mean(jnp.array([self.Sum_agreement(params,epsilon,xdat,ydat) for xdat,ydat in zip(self.X_batch,self.Y_batch)]))
        grad_con2 = jax.grad(con_fun2)
        constraints = [{"type":"ineq","fun":con_fun,"jac":grad_con},{"type":"ineq","fun":con_fun2,"jac":grad_con2}]
        
        out = minimize(minfunc,initial_guess,constraints=constraints,jac=grad,callback=self.callback,options={"disp":True,"maxiter":1000})
    
        return out


# sigma = 1
# l = 2.
# loc_err = 0.1
# ntimepoints = 300
# tmin,tmax = 0,10
# T = 0.5
# # w = ntimepoints//5
# GPDC = MS2_GPDC([T,l,sigma,loc_err],"E")
# xdat,ydat = GPDC.Generate_dataset(tmin,tmax,ntimepoints)


# mu_func,v_cov_func,MS2_mu_func,MS2_cov_func = GPDC.Get_Predfunc(ydat[0],ydat[1])

# from matplotlib import pyplot as plt
# fig,ax = plt.subplots(1,2,figsize=(10,5))
# ax[0].plot(*ydat,zorder=-1)
# ax[0].plot(ydat[0],MS2_mu_func(ydat[0]))
# ax[0].fill_between(ydat[0],MS2_mu_func(ydat[0])-jnp.sqrt(MS2_cov_func(ydat[0],ydat[0])),MS2_mu_func(ydat[0])+jnp.sqrt(MS2_cov_func(ydat[0],ydat[0])),alpha=0.3,color="C1")
# ax[1].plot(*xdat)
# ax[1].plot(ydat[0],mu_func(ydat[0]))
# ax[1].fill_between(ydat[0],mu_func(ydat[0])-jnp.sqrt(v_cov_func(ydat[0],ydat[0])),mu_func(ydat[0])+jnp.sqrt(v_cov_func(ydat[0],ydat[0])),alpha=0.3,color="C1")

# # pred_x = m_func(full_ts)



# fig,ax = plt.subplots(2,1,figsize=(10,5),sharex=True)
# ax[0].plot(data_ts,y,label="Data")
# ax[0].plot(data_ts,pred_y,label="Predicted data")
# ax[0].legend()
# ax[1].plot(full_ts,pred_x,label="Predicted latent")
# ax[1].plot(full_ts,x,label="True latent")
# ax[1].legend()


# # def Gen_MS2_measurement(pol2_rates,w,kappa,noise_std,random_pol2=False):
# #     trajlen = len(pol2_rates)
# #     if not random_pol2:
# #         pol2_nums = pol2_rates
# #     else:
# #         pol2_nums = np.random.poisson(pol2_rates)
# #     kernel_func = jnp.array([MS2_kernel(i,w*kappa) for i in range(w)])[::-1]
# #     kernel_func = kernel_func/np.sum(kernel_func)
# #     MS2_signal = jnp.convolve(pol2_nums,kernel_func,mode="valid")
# #     ts_MS2 = np.arange(w-1,trajlen)
# #     signal = MS2_signal + np.random.normal(0,noise_std,len(MS2_signal))
# #     return ts_MS2,MS2_signal,signal

        
        
# # kernel_func = MS2_kernel(jnp.arange(100).astype(float)[::-1],1.)

# # pol2_nums = np.random.poisson(100,10000000)
# # trace = jnp.convolve(pol2_nums,np.ones(100),mode="valid")
# # trace = trace/np.mean(trace)

# # corrfun = []
# # ts = np.arange(0,200,2)
# # for i in ts:
# #     if i %300 == 0:
# #         print(i)
# #     if i==0:
# #         corrfun.append( jnp.mean(trace**2))
# #     else:
# #         corrfun.append( jnp.mean(trace[i:]*trace[:-i]))
# # from matplotlib import pyplot as plt
# # plt.plot(ts,np.array(corrfun))
# # plt.plot(ts,np.heaviside(100-ts,1)*(100-ts)/100/100**2+1)



# from matplotlib import pyplot as plt


# trajlen = 100
# w = 10
# kappa = 0.01

# kplus,kminus = 1,1
# lambda_scale = 10
# noise_std = np.sqrt(lambda_scale*w)
# dt = 0.1
# kernel_l = 1.


# states,traj = Generate_multistate_trajectory(kplus,kminus,2,np.ones(trajlen),dt=dt)
# loading_rate = (traj/0.9+0.1).astype(float)*lambda_scale
# _,pol2_nums,MS2_signal_Rpol,signal_Rpol = Gen_MS2_measurement(loading_rate,[w,kappa,noise_std],random_pol2=True)


# times = jnp.arange(0,trajlen-w+1)*dt
# t1s,t2s = jnp.meshgrid(times,times,indexing="ij")
# # def Marginal_LLH(theta):
# #     T,scale = theta
# #     n = len(traj)-w+1
# #     Ky = Sigma_f_SE(t1s-t2s,jnp.abs(T),kernel_l,scale)+noise_std**2*jnp.eye(len(times))
# #     (sign, logabsdet) = jnp.linalg.slogdet(Ky)
# #     sigvals = signal_Rpol-jnp.mean(signal_Rpol)#(trajectory-jnp.mean(trajectory))/jnp.std(trajectory)#signal_Rpol-sigmean
# #     inv_Ky = jnp.linalg.inv(Ky)
# #     return (n/2)*jnp.log(jnp.pi/2)+0.5*logabsdet+0.5*sigvals@inv_Ky@sigvals

# # grad = jax.grad(Marginal_LLH)
# # out = minimize(Marginal_LLH,[w*dt,1.],jac=grad,bounds=[(dt,100*dt),(0.01,np.std(signal_Rpol)*10)])
# # out

# T_here,scale_here = out.x
# kernel_l_here = kernel_l
# Kx = Sigma_x_SE(t1s-t2s,kernel_l_here,scale_here)
# Kf = Sigma_f_SE(t1s-t2s,np.abs(T_here),kernel_l_here,scale_here)
# Ky = Kf+noise_std**2*jnp.eye(len(times))
# Kxy = Sigma_fx_SE(t1s-t2s,np.abs(T_here),kernel_l_here,scale_here)
# Ky_inv = np.linalg.inv(Ky)

# signal_pred = Kxy@Ky_inv@signal_Rpol
# signal_covar = Kx-Kxy@Ky_inv@Kxy.T

# _,_,predsig,_ = Gen_MS2_measurement(signal_pred,[w,kappa,noise_std],random_pol2=False)

# # sample = np.random.multivariate_normal(signal_pred,signal_covar)
# # ts,_,predsig_samp,_ = Gen_MS2_measurement(signal_pred,[w,kappa,noise_std],random_pol2=False)
# fig,ax = plt.subplots(1,2,figsize=(10,5))
# ax[0].plot(signal_Rpol,label="Observed")
# # ax[0].plot(predsig,label="Predicted")
# ax[0].plot(Kf@Ky_inv@(signal_Rpol),label="Predicted")
# # ax[0].plot(ts,predsig_samp,".")
# # ax[1].plot(pol2_nums,label="True")
# ax[1].plot(signal_pred,label="Predicted")
# ax[1].plot(loading_rate,label="True")
# # ax[1].plot(sample)

# # ax[1].fill_between(np.arange(len(signal_pred)),signal_pred-np.sqrt(np.diag(signal_covar)),signal_pred+np.sqrt(np.diag(signal_covar)),alpha=0.2)


# # np.random.seed(10)

# # # analyzer = MS2_Analyzer(w,kappa,noise_std)
# # # ts_MS2,MS2_signal,signal = analyzer.Gen_MS2_measurement(loading_rate)
# # nsamples = 10
# # X = [jnp.arange(w-1,trajlen)*dt]*nsamples
# # Y = []
# # Y_true = []
# # Y_true_deconv = []
# # for i in tqdm(list(range(nsamples))):
# #     states,traj = Generate_multistate_trajectory(kplus,kminus,2,np.ones(trajlen),dt=dt)
# #     loading_rate = traj.astype(float)*lambda_scale
# #     _,pol2_nums,MS2_signal_Rpol,signal_Rpol = Gen_MS2_measurement(loading_rate,[w,kappa,noise_std],random_pol2=True)
# #     Y.append(signal_Rpol)
# #     Y_true.append(MS2_signal_Rpol)
# #     Y_true_deconv.append(pol2_nums)


# # trajectory = Y[6]



# # t1s,t2s = jnp.meshgrid(X[6],X[6],indexing="ij")
# # @jax.jit
# # def Marginal_LLH(theta):
# #     T,kernel_l,scale = theta
# #     n = len(traj)-w+1
# #     Ky = Sigma_f_SE(t1s-t2s,jnp.abs(T),kernel_l,scale)+0.5**2*jnp.eye(len(X[6]))
# #     (sign, logabsdet) = jnp.linalg.slogdet(Ky)
# #     sigvals = trajectory#(trajectory-jnp.mean(trajectory))/jnp.std(trajectory)#signal_Rpol-sigmean
# #     inv_Ky = jnp.linalg.inv(Ky)
# #     return (n/2)*jnp.log(jnp.pi/2)+0.5*logabsdet+0.5*sigvals@inv_Ky@sigvals

# # grad = jax.grad(Marginal_LLH)
# # out = minimize(Marginal_LLH,[w*dt,0.1,1.],jac=grad)




# # T_here,kernel_l_here,scale_here = [w*dt,0.1,lambda_scale]
# # noise_here = 0.5
# # Kf = Sigma_f_SE(t1s-t2s,T_here,kernel_l_here,scale_here)
# # Ky = Kf+noise_here**2*jnp.eye(len(X[6]))
# # Ky_inv = jnp.linalg.inv(Ky)
# # pred = Kf@Ky_inv@trajectory
# # plt.plot(pred)
# # plt.plot(Y_true[6])

# # Kx = Sigma_x_SE(t1s-t2s,kernel_l_here,scale_here)   
# # Kxy = Sigma_fx_SE(t1s-t2s,T_here,kernel_l_here,scale_here)
# # loading_pred =  Kxy@Ky_inv@(trajectory)
# # plt.plot(loading_pred)
# # plt.plot(Y_true_deconv[6])


# # GPDC = MS2_GPDC(nkerneltries=10)
# # # bounds = [(5*dt,2*w*dt),(0.001,1.),(0.01,np.std(Y)),(0.1,np.std(Y))]
# # means,covars,MS2_means,MS2_covars = GPDC.fit(X,Y,[w*dt,1.,lambda_scale,noise_std])


# # plt.plot(MS2_means[5])
# # plt.plot(Y[5])

# # plt.plot(means[5]/100)
# # plt.plot(Y_true_deconv[5])


# # # np.random.seed(42)
# # # sample_choice = np.random.randint(0,nsamples)
# # # fig,ax = plt.subplots(2,1,figsize=(10,5))
# # # ax[0].plot(np.arange(w-1,trajlen),Y[sample_choice],label="Observed MS2")
# # # ts,pred_MS2,sampled_MS2,sig = Gen_MS2_measurement(means[sample_choice],[w,kappa,noise_std],random_pol2=False)
# # # ax[0].plot(np.arange(w-1,trajlen),Y_true[sample_choice],label="True MS2")
# # # ax[0].plot(ts,sampled_MS2,label="Predicted MS2")
# # # ax[1].plot(np.arange(w-1,trajlen),means[sample_choice],label="Predicted loading rate")
# # # ax[1].fill_between(np.arange(w-1,trajlen),means[sample_choice]-np.sqrt(np.diag(covars[sample_choice])),means[sample_choice]+np.sqrt(np.diag(covars[sample_choice])),alpha=0.2)
# # # ax[1].plot(np.arange(trajlen),Y_true_deconv[sample_choice],label="True loading rate",alpha=0.5)
# # # ax[0].legend()
# # # ax[1].legend()

# # GPDC.params
# # GPDC.time_grids[sample_choice][0]-GPDC.time_grids[sample_choice][1]

# # np.random.seed(42)
# # sample_choice = np.random.randint(0,nsamples)
# # times = jnp.arange(0,len(traj)-w+1)*dt
# # t1s,t2s = jnp.meshgrid(times,times,indexing="ij")
# # T_use,kernel_l_use,lambda_scale_use,noise_std_use = np.abs(GPDC.params)#[w*dt,0.01,lambda_scale*5,noise_std]#GPDC.params
# # Kx = Sigma_x_SE(t1s-t2s, kernel_l_use,lambda_scale_use)
# # Ky = Sigma_f_SE(t1s-t2s,T_use,kernel_l_use,lambda_scale_use)+noise_std_use**2*jnp.eye(len(times))
# # Kxy = Sigma_fx_SE(t1s-t2s,T_use,kernel_l_use,lambda_scale_use)
# # Ky_inv = jnp.linalg.inv(Ky)
# # signal_pred = Kxy@Ky_inv@(Y[sample_choice])
# # signal_covar = Kx-Kxy@Ky_inv@Kxy.T
# # plt.plot(signal_pred,label="pred")
# # plt.plot(means[sample_choice])
# # plt.plot(Y_true[sample_choice],label="True")

# # plt.legend()

# # # GPDC.Marginal_LLH([ 4.198e+00,1.131e+01,1.161e+02,1.124e-01])#[w*dt,kernel_l,lambda_scale,noise_std])


# # GPDC.fit_kernelparams(boundbounds,s)

# # Sigma_f_SE(GPDC.time_grids[0][1]-GPDC.time_grids[0][0],w*dt,0.001,1)[0]

# # (GPDC.time_grids[0][1]-GPDC.time_grids[0][0])[0]
# # plt.imshow(GPDC.Sigma_f(GPDC.time_grids[0][1]-GPDC.time_grids[0][0],w*dt,0.001,1))
# # T,l,sigma

# # sigmean = np.mean(signal_Rpol)

# # times = jnp.arange(0,len(traj)-w+1)*dt
# # t1s,t2s = jnp.meshgrid(times,times,indexing="ij")
# # plt.plot(np.arange(w-1,len(traj)),signal_Rpol)

# # @jax.jit
# # def Marginal_LLH(theta):
# #     T,kernel_l,lambda_scale,noise_std = theta
# #     n = len(traj)-w+1
# #     Ky = vSigma_f(t1s-t2s,jnp.abs(T),kernel_l,lambda_scale)+noise_std**2*jnp.eye(len(times))
# #     det_K = jnp.linalg.det(Ky)
# #     print(theta,det_K)
# #     sigvals = signal_Rpol-sigmean
# #     inv_Ky = jnp.linalg.inv(Ky)
# #     return (n/2)*jnp.log(jnp.pi/2)+0.5*jnp.log(det_K)+0.5*sigvals@inv_Ky@sigvals
# # grad = jax.grad(Marginal_LLH)
# # from scipy.optimize import minimize
# # out = minimize(Marginal_LLH,np.array([w*dt+np.random.normal(0,w*dt/2),kernel_l+np.random.normal(0,kernel_l/2),lambda_scale+np.random.normal(0,lambda_scale/2),noise_std+np.random.normal(0,noise_std/2)]),jac=grad,bounds=[(dt,100*dt),(dt,len(signal_Rpol)*dt),(0.01,jnp.std(signal_Rpol)*10),(0.01,jnp.std(signal_Rpol))])

# # T_use,kernel_l_use,lambda_scale_use,noise_std_use = out.x
# # Kx = vSigma_x(t1s-t2s, kernel_l_use,lambda_scale_use)
# # Ky = vSigma_f(t1s-t2s,T_use,kernel_l_use,lambda_scale_use)+noise_std_use**2*jnp.eye(len(times))
# # Kxy = vSigma_fx(t1s-t2s,T_use,kernel_l_use,lambda_scale_use)


# # Ky_inv = jnp.linalg.inv(Ky)
# # signal_pred = Kxy@Ky_inv@(signal_Rpol-sigmean)+sigmean
# # signal_covar = Kx-Kxy@Ky_inv@Kxy.T

# # fig,ax = plt.subplots(2,1,figsize=(10,5))
# # ax[0].plot(np.arange(w-1,len(traj)),signal_Rpol,label="Observed MS2")
# # ax[0].plot(np.arange(w-1,len(traj)),MS2_signal_Rpol,label="True MS2")
# # ts,_,pred_MS2,sampled_MS2 = analyzer.Gen_MS2_measurement(signal_pred,random_pol2=False)
# # ax[0].plot(ts,pred_MS2,label="Predicted MS2")
# # ax[0].legend()
# # # x,res,rank,sing = jnp.linalg.lstsq(convolution_matrix,signal_pred)
# # ax[1].fill_between(np.arange(len(signal_pred)),signal_pred-np.sqrt(np.diag(signal_covar)),signal_pred+np.sqrt(np.diag(signal_covar)),alpha=0.2,color="C0")
# # ax[1].plot(np.arange(len(signal_pred)),signal_pred,color="C0",label="Predicted loading rate")
# # ax[1].plot(np.arange(len(traj)),traj*lambda_scale,color="C1",label="True loading rate",alpha=0.5)
# # ax[1].plot(np.arange(len(traj)),pol2_nums,color="C1",label="True Pol2",alpha=0.5)
# # ax[1].legend()

# # # trajlen = 1000000
# # # w = 100
# # # kappa = 0.01
# # # noise_std = 0.5
# # # kplus,kminus = 1,0.3
# # # dt = 0.01
# # # # np.random.seed(10)
# # # states,traj = Generate_multistate_trajectory(kplus,kminus,2,np.ones(trajlen),dt=dt)


# # # loading_rate = traj.astype(float)*100
# # # analyzer = MS2_Analyzer(w,kappa,noise_std)
# # # # ts_MS2,MS2_signal,signal = analyzer.Gen_MS2_measurement(loading_rate)
# # # _,MS2_signal_Rpol,signal_Rpol = analyzer.Gen_MS2_measurement(loading_rate,random_pol2=True)



# # # corrfun = []
# # # ts = np.arange(0,500,2)
# # # for i in ts:
# # #     if i %300 == 0:
# # #         print(i)
# # #     if i==0:
# # #         corrfun.append( jnp.mean(signal_Rpol**2))
# # #     else:
# # #         corrfun.append( jnp.mean(signal_Rpol[i:]*signal_Rpol[:-i]))

# # # mean_subtract = np.array(corrfun)-np.mean(signal_Rpol)**2
# # # plt.plot(ts*dt*kminus,mean_subtract/mean_subtract[0],label="Simulated")
# # # def MS2_corrfun(x,alpha,loading_rate, p_star,kminus,T,sigma,dt):
# # #     noise_term = sigma**2
# # #     steady_state_term = (alpha*loading_rate*p_star)**2
# # #     term1 = (1-np.exp(kminus*T))**2
# # #     term2 = 1-2*np.exp(kminus*T)+np.exp(2*kminus*x)+2*kminus*(T-x)*np.exp(kminus*(T+x))
# # #     correlation_term =((alpha/kminus)*loading_rate*p_star)**2* np.exp(-kminus*(x+T))*(np.heaviside(x-T,1)*term1+np.heaviside(T-x,0)*term2)
    
# # #     out = 0 
# # #     if x==0:
# # #         out+=noise_term
# # #     out+=correlation_term #+ steady_state_term
# # #     return out
# # # plt.plot(ts*dt*kminus,[MS2_corrfun(i,1,np.max(loading_rate),kplus/(kplus+kminus),2.5*kminus,w*dt,noise_std,dt)/MS2_corrfun(0,1,np.max(loading_rate),kplus/(kplus+kminus),2.5*kminus,w*dt,noise_std,dt)  for i in ts*dt],label="Analytical")
# # # plt.plot(ts*dt*kminus,np.exp(-2*kplus*ts*dt),label="Exponential")
# # # plt.axvline(w*dt*kminus)
# # # plt.legend()
# # # plt.yscale("log")

# # # np.max(loading_rate)*kplus/(kminus+kplus)
# # # (np.max(loading_rate)*kplus/(kminus+kplus))**2
# # # # # plt.plot(traj)
# # # # tx = plt.twinx()
# # # # plt.plot(ts_MS2,signal)
# # # # tx.plot(ts_MS2,MS2_signal_Rpol)
# # # from scipy.optimize import minimize
# # # current_val = jnp.ones(trajlen)+0.1*np.random.normal(0,1,trajlen)
# # # gradfun = jax.grad(MS2_LLH)
# # # hessfun = jax.hessian(MS2_LLH)
# # # out = minimize(MS2_LLH,current_val,args=(signal,analyzer.kernel_func),jac=gradfun,hess=hessfun,method="trust-exact")
# # # MS2_LLH(traj,signal,analyzer.kernel_func),out.fun


# # # fig,ax = plt.subplots(2,1,figsize=(10,5))

# # # ax[0].plot(out.x**2)
# # # ax[0].plot(traj)

# # # ax[1].plot(signal)
# # # ax[1].plot(jnp.convolve(out.x**2,analyzer.kernel_func,mode="valid"))
# # # MS2_LLH(out.x,signal,analyzer.kernel_func)