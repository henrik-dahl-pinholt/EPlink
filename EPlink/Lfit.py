from jax import config
import jax.numpy as jnp
import numpy as np
from jax import jit, random, jacfwd, scipy, lax, vmap,jacrev
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult
from functools import partial
from tqdm import tqdm


@jit
def Gamma_gauss(x, a, b, bias):
    # mean, var = jnp.abs(a * b), jnp.abs(a * b**2)
    mean,var = jnp.abs(a),jnp.abs(b)
    return (
        (1 / jnp.sqrt(2 * jnp.pi * var)) * jnp.exp(-0.5 * (x - mean) ** 2 / var) * bias
    )


@jit
def stable_gamma(x, a, b, bias):
    a_bool, x_bool = a > 10, x == 0
    expr_1 = (
        lambda x, a, b: jnp.sqrt(1 / jnp.abs(a))
        * jnp.exp(-jnp.abs(a))
        * jnp.sqrt(2 * jnp.pi)
        * (1 / (jnp.abs(b) * jnp.abs(a))) ** jnp.abs(a)
    )
    expr_2 = lambda x, a, b: (jnp.abs(b)) ** jnp.abs(a) / scipy.special.gamma(
        jnp.abs(a)
    )
    expr_3 = (
        lambda x, a, b: ((12 * jnp.abs(a) - 1) / (12 * jnp.sqrt(2 * jnp.pi)))
        * jnp.sqrt(1 / jnp.abs(a))
        * jnp.exp(
            jnp.abs(a) * jnp.log(x / (jnp.abs(a) * jnp.abs(b)))
            + jnp.abs(a)
            - x / jnp.abs(b)
        )
        / x
    )
    expr_4 = (
        lambda x, a, b: (1 / jnp.abs(b)) ** jnp.abs(a)
        / scipy.special.gamma(jnp.abs(a))
        * x ** (jnp.abs(a) - 1)
        * jnp.exp(-(1 / jnp.abs(b)) * x)
    )
    return (
        lax.cond(
            x_bool,
            lambda x, a, b: lax.cond(a_bool, expr_1, expr_2, x, a, b),
            lambda x, a, b: lax.cond(a_bool, expr_3, expr_4, x, a, b),
            x,
            a,
            b,
        )
        * bias
    )


stable_gamma = vmap(stable_gamma, in_axes=(0, None, None, None))
Gamma_gauss = vmap(Gamma_gauss, in_axes=(0, None, None, None))


def rolling_window(a: jnp.ndarray, window: int):
    idx = jnp.arange(len(a) - window + 1)[:, None] + jnp.arange(window)[None, :]
    return a[idx][1:]


# @partial(jit, static_argnums=(1,))
# def compute_T(trajectory, window_size, params, dt):
#     # activation = jnp.exp(-trajectory**2/2/params[0]**2)
#     # differences = jnp.arange(0,window_size)[::-1]*dt
#     # window_view = rolling_window(trajectory,start_ind)
#     # i = window_size
#     # rates = []
#     # for i in range(0,len(trajectory)-window_size):
#     #     rates.append(jnp.dot(activation[i:window_size+i],Erlang_distribution(differences, *params[1:])))
#     # return rates

#     window_view = rolling_window(
#         trajectory, window_size
#     )  # jnp.lib.stride_tricks.sliding_window_view(trajectory,start_ind)
#     time_differences = jnp.arange(0, window_view.shape[1])[::-1] * dt
#     contacts = jnp.exp(-(window_view**2) / (2 * params[0] ** 2)) / jnp.sqrt(
#         2 * jnp.pi * params[0] ** 2
#     )
#     weights = stable_gamma(time_differences, params[1], params[2], params[3]) * dt

#     return contacts @ weights


func_dict = {
    "Gamma_gauss": Gamma_gauss,
    "stable_gamma": stable_gamma,
}


@partial(jit, static_argnums=(1, 4))
def compute_T(trajectory, window_size, params, dt, weighting_kernel):
    time_differences = jnp.arange(0, window_size + 1)[::-1] * dt
    weights = (
        func_dict[weighting_kernel](time_differences, params[1], params[2], params[3])
        * dt
    )
    contacts = jnp.exp(-jnp.sum(trajectory**2,axis=-1) / (2 * params[0] ** 2)) / jnp.sqrt(
        2 * jnp.pi * params[0] ** 2
    )
    return jnp.convolve(contacts, weights, mode="valid")  # contacts@weights


# compute_T = vmap(compute_T, (0, None, None, None, None))  # , static_argnums=(1,)



def Propagate(
    mu_old, var_old, mu_increment, var_increment, mu_mult_factor, var_mult_factor
):
    mu_new = mu_old * mu_mult_factor + mu_increment
    var_new = var_old * var_mult_factor + var_increment
    return mu_new, var_new


@jit
def Update_forward(y, mu, var, measurement_error):
    mu_new = (mu / var + y / measurement_error**2) / (
        1 / measurement_error**2 + 1 / var
    )
    var_new = 1 / (1 / measurement_error**2 + 1 / var)
    return mu_new, var_new

@partial(jit, static_argnums=(2,4,5))
def Compute_increments(params, trajectory, window_size, dt, upscale_factor, weighting_kernel,gamma_trans,D_trans):
        T_rate = compute_T(
            trajectory, window_size, params, dt, weighting_kernel
        )
        
        rate_blocks = T_rate.reshape(
            len(T_rate) // upscale_factor,
            upscale_factor,
        )
        dt_data = dt * upscale_factor
        t_diffs = jnp.arange(0, upscale_factor)[::-1] * dt
        mu_increments = jnp.sum(
            jnp.exp(-gamma_trans * (t_diffs)) * dt * rate_blocks,
            axis=-1,
        )
        mu_mult_factor = jnp.exp(-gamma_trans * (dt_data))
        var_mult_factor = jnp.exp(-2 * gamma_trans * (dt_data))
        var_increment = (D_trans / gamma_trans) * (
            1 - jnp.exp(-2 * gamma_trans * (dt_data))
        )
        return mu_increments, mu_mult_factor, var_mult_factor, var_increment, rate_blocks, T_rate
    

# @partial(jit, static_argnums=(2,4,5))
# def Compute_increments(params, trajectory, window_size, dt, upscale_factor, weighting_kernel,gamma_trans,D_trans):
#         T_rate = compute_T(
#             trajectory, window_size, params, dt, weighting_kernel
#         )
#         rate_blocks = T_rate.reshape(
#             trajectory.shape[0],
#             T_rate.shape[1] // upscale_factor,
#             upscale_factor,
#         )
#         dt_data = dt * upscale_factor
#         t_diffs = jnp.arange(0, upscale_factor)[::-1] * dt
#         mu_increments = jnp.sum(
#             jnp.exp(-gamma_trans * (t_diffs)) * dt * rate_blocks,
#             axis=-1,
#         )
#         mu_mult_factor = jnp.exp(-gamma_trans * (dt_data))
#         var_mult_factor = jnp.exp(-2 * gamma_trans * (dt_data))
#         var_increment = (D_trans / gamma_trans) * (
#             1 - jnp.exp(-2 * gamma_trans * (dt_data))
#         )
#         return mu_increments, mu_mult_factor, var_mult_factor, var_increment, rate_blocks, T_rate

def GP_predict(params,data,posterior_traject,window_size,dt,upscale_factor,kernel,gamma_trans,D_trans,measurement_error):
    """Predict the mean and variance of the process driving a transcription signal assuming a Gaussian process linear growth model (with additive noise) for the signal where the rate of transcription arises from the posterior EP samples as a kernel-weighted sum of the past contact values. The number timepoints in each posterior samples must be an integer multiple of the number of timepoints in the data.

    Parameters
    ----------
    params : (1+nkernelparams,) jax array
        The parameters of the GP model. The first is the contact radius, and the last three are the kernel parameters.
    data : (n_timepoints,) jax array
        Transcription data to predict
    posterior_traject : (n_timepoints*upscale_factor,ndim) jax array
        The posterior samples for the polymer trajectories. The number of timepoints in each posterior sample must be an integer multiple (upscale_factor) of the number of timepoints in the data.
    window_size : int
        The size of the window to use for the kernel weighted sum
    dt : float
        The time interval between data timepoints
    upscale_factor : int
        The number of polymer timepoints per data timepoint. Should be an integer.
    kernel : str
        type of kernel to use for the weighting of the past contacts. Options are 'Gamma_gauss' and 'stable_gamma'
    gamma_trans : float
        The rate removal of transcription signal
    D_trans : float
        The diffusion constant of the transcription signal
    measurement_error : float
        The standard deviation of the measurement error
    
    Returns
    ----------
    mus : (n_timepoints) jax array
        The prediction for the mean of the hidden process at each timepoint
    vars : (n_timepoints) jax array
        The prediction for the variance of the hidden process at each timepoint
    """
    # compute the increments for the GP
    mu_increments, mu_mult_factor, var_mult_factor, var_increment, rate_blocks, T_rate = Compute_increments(
        params, posterior_traject, window_size, dt, upscale_factor, kernel,gamma_trans,D_trans)
    mus, vars = jnp.zeros(
        (rate_blocks.shape[0]+1)
    ), jnp.zeros((rate_blocks.shape[0]+1))
    mus = mus.at[0].set(T_rate[0] / gamma_trans)
    vars = vars.at[0].set(
        jnp.ones_like(T_rate[0]) * D_trans / gamma_trans
    )
    for i in range(1, rate_blocks.shape[0]+1):

        updated = Update_forward(
            data[ i - 1], mus[i - 1], vars[i - 1], measurement_error
        )
        

        out1, out2 = Propagate(
            updated[0],
            updated[1],
            mu_increments[i - 1],
            var_increment,
            mu_mult_factor,
            var_mult_factor,
        )
        mus = mus.at[i].set(out1)
        vars = vars.at[i].set(out2)
    return mus, vars
    # mu_increments, mu_mult_factor, var_mult_factor, var_increment, rate_blocks, T_rate = Compute_increments(
    # params, posterior_traject, window_size, dt, upscale_factor, kernel,gamma_trans,D_trans)
    # mus, vars = jnp.zeros(
    #     (posterior_traject.shape[0], rate_blocks.shape[1])
    # ), jnp.zeros((posterior_traject.shape[0], rate_blocks.shape[1]))

    # mus = mus.at[:, 0].set(T_rate[:, 0] / gamma_trans)
    # vars = vars.at[:, 0].set(
    #     jnp.ones_like(T_rate[:, 0]) * D_trans / gamma_trans
    # )
    # for i in range(1, rate_blocks.shape[1]):
    #     updated = Update_forward(
    #         #data[:, i - 1], mus[:, i - 1], vars[:, i - 1], measurement_error
    #         data[ i - 1], mus[:, i - 1], vars[:, i - 1], measurement_error
    #     )

    #     out1, out2 = Propagate(
    #         updated[0],
    #         updated[1],
    #         mu_increments[:, i - 1],
    #         var_increment,
    #         mu_mult_factor,
    #         var_mult_factor,
    #     )
    #     mus = mus.at[:, i].set(out1)
    #     vars = vars.at[:, i].set(out2)
    #     # mus[:,i],vars[:,i] = Propagate(updated[0],updated[1],self.mu_increments[i-1],self.var_increment,self.mu_mult_factor,self.var_mult_factor)
    # return mus, vars

@jit
def GP_LogLikelihood(prediction, data,measurement_error):
        LLHval = -0.5 * jnp.mean(
            jnp.log(2 * jnp.pi)
            + jnp.log(prediction[1] + measurement_error**2)
            + (prediction[0] - data) ** 2 / (prediction[1] + measurement_error**2)
        )
        return -LLHval
    
class GaussianProcessTranscription:
    def __init__(
        self,
        trajectories,
        window_size,
        dt,
        upscale_factor,
        gamma_trans,
        D_trans,
        measurement_error,
        weighting_kernel="stable_gamma",
    ):
        self.trajectory = trajectories
        self.window_size = window_size
        self.dt = dt
        self.upscale_factor = upscale_factor
        self.gamma_trans = gamma_trans
        self.D_trans = D_trans
        self.measurement_error = measurement_error
        self.weighting_kernel = weighting_kernel
        self.dt_data = dt * upscale_factor

    def Generate_trajectory(self, params,initval=None, seed=5,verbose=False):
        
        mu_increments, mu_mult_factor, var_mult_factor, var_increment, rate_blocks, T_rate = Compute_increments(
            params, self.trajectory, self.window_size, self.dt, self.upscale_factor, self.weighting_kernel,self.gamma_trans,self.D_trans)
        key = random.PRNGKey(seed)
        norm_samps = random.normal(key, (rate_blocks.shape[0],))
        print()
        if initval is not None:
            mu_val, var_val = initval
        else:
            average_rate = T_rate.mean()
            mu_val, var_val = (
                average_rate / self.gamma_trans,
                 self.D_trans / self.gamma_trans,
            )
        x_out = np.zeros(rate_blocks.shape[0])
        
        x_out[0] = norm_samps[0] * jnp.sqrt(var_val) + mu_val
        if verbose:
            iterator = tqdm(list(range(1, rate_blocks.shape[0])))
        else:
            iterator = range(1, rate_blocks.shape[0])
        for i in iterator:
            mu_val, var_val = Propagate(
                x_out[i-1],
                0,
                mu_increments[i - 1],
                var_increment,
                mu_mult_factor,
                var_mult_factor,
            )
            x_out[i] = norm_samps[i] * jnp.sqrt(var_val) + mu_val
            
            
        

        # return (norm_samps) * jnp.sqrt(vars) + mus, mus, vars
        return x_out

    
    
    # def Generate_trajectory(self, params,initval=None, seed=5,verbose=False):
        
    #     mu_increments, mu_mult_factor, var_mult_factor, var_increment, rate_blocks, T_rate = Compute_increments(
    #         params, self.trajectory, self.window_size, self.dt, self.upscale_factor, self.weighting_kernel,self.gamma_trans,self.D_trans)
    #     key = random.PRNGKey(seed)
    #     norm_samps = random.normal(key, (self.trajectory.shape[0], rate_blocks.shape[1]))
        
    #     if initval is not None:
    #         mu_val, var_val = initval
    #     else:
    #         average_rate = T_rate.mean(axis=1)
    #         mu_val, var_val = (
    #             average_rate / self.gamma_trans,
    #             jnp.ones_like(T_rate[:, 0]) * self.D_trans / self.gamma_trans,
    #         )
    #     x_out = np.zeros((self.trajectory.shape[0], rate_blocks.shape[1]))
        
    #     x_out[:, 0] = norm_samps[:, 0] * jnp.sqrt(var_val) + mu_val
    #     if verbose:
    #         iterator = tqdm(list(range(1, rate_blocks.shape[1])))
    #     else:
    #         iterator = range(1, rate_blocks.shape[1])
    #     for i in iterator:
    #         mu_val, var_val = Propagate(
    #             x_out[:, i-1],
    #             0,
    #             mu_increments[:, i - 1],
    #             var_increment,
    #             mu_mult_factor,
    #             var_mult_factor,
    #         )
    #         x_out[:,i] = norm_samps[:, i] * jnp.sqrt(var_val) + mu_val
    #     # return (norm_samps) * jnp.sqrt(vars) + mus, mus, vars
    #     return x_out

    # def Fit(self, initial_guess, data, use_hessian=True,fixed_params=None, **kwargs):
    #     pred_func = vmap(self.Predict,(None,0))
    #     def LLH(params):
    #         if fixed_params is not None:
    #             params = jnp.concatenate([params,fixed_params])

    #         LLHval = self.LogLikelihood(pred_func(jnp.array(params), data), data)
    #         print("Ran an iteration")
    #         return LLHval

    #     grad_LLh = jacrev(LLH)
    #     LLH_hessian = jacfwd(grad_LLh)#jacfwd(grad_LLh)
    #     param_save = []
    #     def print_progress(intermediate_result: OptimizeResult):
    #         params = intermediate_result.x
    #         LLHval = intermediate_result.fun
    #         if fixed_params is not None:
    #             params = jnp.concatenate([params,fixed_params])
    #         print(
    #             f"rc: {params[0]}, k: {params[1]}, lam: {params[2]}, bias: {params[3]}, LLH: {LLHval}"
    #         )
    #         param_save.append((params, LLHval))
    #     paramdict = {"callback": print_progress, "jac": grad_LLh}
    #     if use_hessian:
    #         paramdict["hess"] = LLH_hessian
    #         paramdict["method"] = "Newton-CG"

    #     for key, value in kwargs.items():
    #         paramdict[key] = value
    #     print("Running optimization")
    #     out = minimize(
    #         LLH,
    #         initial_guess,
    #         **paramdict,
    #         # jac=grad_LLh,
    #         # hess=LLH_hessian,
    #         # method="Newton-CG",
    #     )
    #     print("\tOptimization done")
    #     return out, LLH_hessian(out.x), param_save

# class GaussianProcessTranscription:
#     def __init__(
#         self,
#         trajectories,
#         window_size,
#         dt,
#         upscale_factor,
#         gamma_trans,
#         D_trans,
#         measurement_error,
#         weighting_kernel="stable_gamma",
#     ):
#         self.trajectory = trajectories
#         self.window_size = window_size
#         self.dt = dt
#         self.upscale_factor = upscale_factor
#         self.gamma_trans = gamma_trans
#         self.D_trans = D_trans
#         self.measurement_error = measurement_error
#         self.weighting_kernel = weighting_kernel
#         self.dt_data = dt * upscale_factor

    

#     def Predict(self, params, data):
#         mu_increments, mu_mult_factor, var_mult_factor, var_increment, rate_blocks, T_rate = Compute_increments(
#             params, self.trajectory, self.window_size, self.dt, self.upscale_factor, self.weighting_kernel,self.gamma_trans,self.D_trans)
#         mus, vars = jnp.zeros(
#             (self.trajectory.shape[0], rate_blocks.shape[1])
#         ), jnp.zeros((self.trajectory.shape[0], rate_blocks.shape[1]))

#         mus, vars = jnp.zeros(
#             (self.trajectory.shape[0], rate_blocks.shape[1])
#         ), jnp.zeros((self.trajectory.shape[0], rate_blocks.shape[1]))

#         mus = mus.at[:, 0].set(T_rate[:, 0] / self.gamma_trans)
#         vars = vars.at[:, 0].set(
#             jnp.ones_like(T_rate[:, 0]) * self.D_trans / self.gamma_trans
#         )
        
#         for i in range(1, rate_blocks.shape[1]):
#             updated = Update_forward(
#                 #data[:, i - 1], mus[:, i - 1], vars[:, i - 1], self.measurement_error
#                 data[ i - 1], mus[:, i - 1], vars[:, i - 1], self.measurement_error
#             )

#             out1, out2 = Propagate(
#                 updated[0],
#                 updated[1],
#                 mu_increments[:, i - 1],
#                 var_increment,
#                 mu_mult_factor,
#                 var_mult_factor,
#             )
#             mus = mus.at[:, i].set(out1)
#             vars = vars.at[:, i].set(out2)
#             # mus[:,i],vars[:,i] = Propagate(updated[0],updated[1],self.mu_increments[i-1],self.var_increment,self.mu_mult_factor,self.var_mult_factor)
#         return mus, vars

#     def LogLikelihood(self, prediction, data):
#         LLHval = -0.5 * jnp.mean(
#             jnp.log(2 * jnp.pi)
#             + jnp.log(prediction[1] + self.measurement_error**2)
#             + (prediction[0] - data) ** 2 / (prediction[1] + self.measurement_error**2)
#         )
#         return -LLHval

#     def Generate_trajectory(self, params,initval=None, seed=5,verbose=False):
        
#         mu_increments, mu_mult_factor, var_mult_factor, var_increment, rate_blocks, T_rate = Compute_increments(
#             params, self.trajectory, self.window_size, self.dt, self.upscale_factor, self.weighting_kernel,self.gamma_trans,self.D_trans)
#         key = random.PRNGKey(seed)
#         norm_samps = random.normal(key, (self.trajectory.shape[0], rate_blocks.shape[1]))
        
#         if initval is not None:
#             mu_val, var_val = initval
#         else:
#             average_rate = T_rate.mean(axis=1)
#             mu_val, var_val = (
#                 average_rate / self.gamma_trans,
#                 jnp.ones_like(T_rate[:, 0]) * self.D_trans / self.gamma_trans,
#             )
#         x_out = np.zeros((self.trajectory.shape[0], rate_blocks.shape[1]))
        
#         x_out[:, 0] = norm_samps[:, 0] * jnp.sqrt(var_val) + mu_val
#         if verbose:
#             iterator = tqdm(list(range(1, rate_blocks.shape[1])))
#         else:
#             iterator = range(1, rate_blocks.shape[1])
#         for i in iterator:
#             mu_val, var_val = Propagate(
#                 x_out[:, i-1],
#                 0,
#                 mu_increments[:, i - 1],
#                 var_increment,
#                 mu_mult_factor,
#                 var_mult_factor,
#             )
#             x_out[:,i] = norm_samps[:, i] * jnp.sqrt(var_val) + mu_val
            
            
        

#         # return (norm_samps) * jnp.sqrt(vars) + mus, mus, vars
#         return x_out

#     def Fit(self, initial_guess, data, use_hessian=True,fixed_params=None, **kwargs):
#         pred_func = vmap(self.Predict,(None,0))
#         def LLH(params):
#             if fixed_params is not None:
#                 params = jnp.concatenate([params,fixed_params])

#             LLHval = self.LogLikelihood(pred_func(jnp.array(params), data), data)
#             print("Ran an iteration")
#             return LLHval

#         grad_LLh = jacrev(LLH)
#         LLH_hessian = jacfwd(grad_LLh)#jacfwd(grad_LLh)
#         param_save = []
#         def print_progress(intermediate_result: OptimizeResult):
#             params = intermediate_result.x
#             LLHval = intermediate_result.fun
#             if fixed_params is not None:
#                 params = jnp.concatenate([params,fixed_params])
#             print(
#                 f"rc: {params[0]}, k: {params[1]}, lam: {params[2]}, bias: {params[3]}, LLH: {LLHval}"
#             )
#             param_save.append((params, LLHval))
#         paramdict = {"callback": print_progress, "jac": grad_LLh}
#         if use_hessian:
#             paramdict["hess"] = LLH_hessian
#             paramdict["method"] = "Newton-CG"

#         for key, value in kwargs.items():
#             paramdict[key] = value
#         print("Running optimization")
#         out = minimize(
#             LLH,
#             initial_guess,
#             **paramdict,
#             # jac=grad_LLh,
#             # hess=LLH_hessian,
#             # method="Newton-CG",
#         )
#         print("\tOptimization done")
#         return out, LLH_hessian(out.x), param_save
