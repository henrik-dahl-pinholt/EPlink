
import GPUtil,os,psutil
from tqdm import tqdm
import jax
from jax import numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
from EPlink import Lfit
from scipy.optimize import minimize
"""This module contains code to simplify batch evaluation of functions from other modules in large computations."""

# class Runner:
#     """This class contains the runner class. An object sought to streamline batch evaluation of large computations in a common framework which easily works with JAXs computational model. The class was introduced to display progress, allow batch computation to combat memory overloads, and to excecute jax code in a distributed fashion across multiple hosts. 
    
#     The runner object takes an iterable, and a function upon initialization. The function is called on each element of the iterable to perform the computation. As an optional argument, an aggregation function can be passed to the class in which case outputs are collected in a list and passed to the aggregation function when the iterator is empty. By letting it be up to the iterable to feed data, the size of the calculations can be easily managed, and parallelization across GPUs is easily accomplished by sharding the data fed by the iterable."""
#     def __init__(self,iterable,func,aggregation_func=None):
#         """Constructor for the Runner object.
        
#         Parameters
#         ----------
#         iterable : iterable
#             The iterable to be fed to the function func. 
#         func : function
#             The function to be called on each element of the iterable. 
#         aggregation_func : function, optional
#             The function to be called on a list of the collected result of func on the elements of the iterator. Should take as input a list of the results. If specified, calling the Run() method will return the result of this function. If not specified, the Run() method will just return the list of outputs directly.
#         """
        
#         self.iterable = iterable
#         self.func = func
#         self.agg_f = aggregation_func


#     def get_system_metrics(self):
#         """Grabs system metrics such as CPU usage, memory usage, GPU usage, GPU memory usage, GPU memory used, and GPU memory total.

#         Returns
#         -------
#         tuple
#             A tuple containing the CPU usage (in percent), memory usage (in percent), GPU usage (in percent), GPU memory used (in MB), GPU memory total (in MB), and GPU memory usage (in percent).
#         """
#         cpu_usage = psutil.cpu_percent(interval=1)
#         memory_info = psutil.virtual_memory()
#         memory_usage = memory_info.percent
#         gpus = GPUtil.getGPUs()
#         if gpus:
#             gpu = gpus[0]
#             gpu_usage = gpu.load * 100
#             gpu_memory_used = gpu.memoryUsed
#             gpu_memory_total = gpu.memoryTotal
#             gpu_memory_usage = gpu.memoryUtil * 100
#         else:
#             gpu_usage = 0
#             gpu_memory_used = 0
#             gpu_memory_total = 0
#             gpu_memory_usage = 0
#         return cpu_usage, memory_usage, gpu_usage, gpu_memory_used, gpu_memory_total, gpu_memory_usage

#     def Run(self,print_stats=True):
#         """Runs the function on the iterable.

#         Parameters
#         ----------
#         print_stats : bool, optional
#             Whether to print system statistics during the run. Default is True. Keep in mind that getting system statistics makes iterations take longer (~1s more) which may be significant depending on how long a function call takes. Default is True.

#         Returns
#         -------
#         list or any
#             If an aggregation function was specified, the result of this function is returned. Otherwise, the list of outputs is returned.
#         """
#         results = []
#         for i in tqdm(self.iterable):
#             if print_stats:
#                 # Print usage statistics
#                 cpu_usage, memory_usage, gpu_usage, gpu_memory_used, gpu_memory_total, gpu_memory_usage = self.get_system_metrics()
#                 tqdm.write(f"CPU Usage: {cpu_usage:4.2f}% | Memory Usage: {memory_usage:4.2f}% | GPU Usage: {gpu_usage:4.2f}% | GPU Memory Usage: {gpu_memory_usage:4.2f}% ({gpu_memory_used:4.2f}/{gpu_memory_total:4.2f} MB)")
                    
#             # Run the function
#             results.append(self.func(i))
            
#         # Post-process the result
#         if not self.agg_f is None:
#             return self.agg_f(results)
#         else:
#             return results
class GP_model:      
    def __init__(self,data,posterior_folder,paramdict):
        self.data = data
        self.posterior_folder = posterior_folder
        
        # check that the parameters are defined
        must_be_defined = ["window_size", "dt","upscale_factor","gamma_trans","D_trans","measurement_error","weighting_kernel"]
        for param in must_be_defined:
            if not param in paramdict:
                raise ValueError(f"Parameter {param} must be defined.")
        self.paramdict = paramdict
        
        # Check that the number of datapoints matches the number of posterior samples
        files = os.listdir(self.posterior_folder)
        self.max_num = max([int(file.split("_")[-1].split(".")[0]) for file in files if "Post_samps" in file])
        
        if self.max_num+1 != len(self.data):
            raise ValueError(f"Number of self.datapoints ({len(data)}) does not match number of posterior batches ({self.max_num+1}).")
        
        #initialize the likelihood, gradient, and hessian
        # self.LLHfunc = jax.jit(self.single_LLH)
        self.gradient = jax.value_and_grad(self.single_LLH,0)
        # self.hessian = jax.jacfwd(jax.jacrev(self.gradient,0),0)

        
    def __len__(self):
        return self.max_num
    
    def __iter__(self):
        # initialize counter
        self.counter = 0
        return self
    
    def __next__(self):
        if self.counter > self.max_num:
            raise StopIteration
        samples = jnp.load(os.path.join(self.posterior_folder,f"Post_samps_{self.counter}.npy"))
        self.counter +=1
        
        return samples
        
    def callback(self,intermediate_result):
        params = intermediate_result.x
        llh = intermediate_result.fun
        # print(f"Current parameters: {params} | Current log-likelihood: {llh}")
        
        # tqdm.write(f"CPU Usage: {cpu_usage:4.2f}% | Memory Usage: {memory_usage:4.2f}% | GPU Usage: {gpu_usage:4.2f}% | GPU Memory Usage: {gpu_memory_usage:4.2f}% ({gpu_memory_used:4.2f}/{gpu_memory_total:4.2f} MB)")
        self.call_count += 1
        self.pbar.update()
        self.pbar.set_description(f"Params: {params} | LLH: {llh}")
    def single_LLH(self,params,sample,data_traj):
        predictions = Lfit.GP_predict(params,data_traj,sample,self.paramdict["window_size"],self.paramdict["dt"],self.paramdict["upscale_factor"],self.paramdict["weighting_kernel"],self.paramdict["gamma_trans"],self.paramdict["D_trans"],self.paramdict["measurement_error"])
                
        # compute log-likelihood and derivatives
        LLH = Lfit.GP_LogLikelihood(predictions,data_traj,self.paramdict["measurement_error"])
        return LLH
    
    def Compute_LLH(self,params):
        #initialize counters, gradients, and hessian funcs
        n = 0
        
        curr_LLH, curr_grad, curr_hess = [],[],[]
        # iterate over samples
        for samples in tqdm(self, leave=False):
            LLHs, grads,hess = [],[],[]
            for sample in samples[:2]:
                # Compute prediction for current sample
                # LLH = self.LLHfunc(params,sample,self.data[n])
                LLH,LLH_grad = self.gradient(params,sample,self.data[n])
                # LLH_hess = self.hessian(params,sample,self.data[n])
                
                # append to lists
                LLHs.append(LLH)
                grads.append(LLH_grad)
                # hess.append(LLH_hess)
            n += 1
            # Average over samples and append to current lists
            curr_LLH.append(jnp.mean(jnp.array(LLHs)))
            curr_grad.append(jnp.mean(jnp.array(grads),axis=0))
            # curr_hess.append(jnp.mean(jnp.array(hess),axis=0))
        # Compute average log-likelihood, gradient, and hessian over all samples
        self.LLH = jnp.mean(jnp.array(curr_LLH))
        self.jac = jnp.mean(jnp.array(curr_grad),axis=0)
        # self.hess = jnp.mean(jnp.array(curr_hess),axis=0)
        # print(f"Current log-likelihood: {self.LLH}, Current params: {params}")
        return self.LLH
    def GetJac(self,params):
        return self.jac
    # def GetHess(self,params):
    #     return self.hess
    def Fit(self,init_guess):
        # Initialize the callback counter 
        self.call_count = 0
        self.pbar =  tqdm()
        
        # Run the optimization
        out = minimize(self.Compute_LLH,init_guess,callback=self.callback,jac=self.GetJac,tol=1e-3)
        
        # Close the progress bar
        self.pbar.close()
        
        return out
    
    
          

# class ZeroGenerator:
#     def __init__(self,zero_shape,nbatches):
#         self.nbatches = nbatches
#         self.counter = 0
#         self.shape = zero_shape
    
#     def __iter__(self):
#         self.counter = 0
#         return self
#     def __next__(self):
#         data = jax.numpy.zeros(self.shape)
#         if self.counter < self.nbatches:
#             self.counter += 1
#             return data
#         else:
#             raise StopIteration
# class NormalGenerator:
#     """Generates unit normal random variates of a given shape. 
#     """
#     def __init__(self,shape,nbatches,seed):
#         """Initializes the NormalGenerator object.

#         Parameters
#         ----------
#         shape : tuple
#             The shape of the random variates to be generated in each iteration.
#         nbatches : int
#             Number of batches (iterations) to generate. The iterator will stop after this number of batches.
#         seed : int
#             Seed for the random number generator.
#         """
#         self.nbatches = nbatches
#         self.counter = 0
#         self.shape = shape
#         self.seed = seed
#     def __len__(self):
#         return self.nbatches
#     def __iter__(self):
#         self.key = jax.random.PRNGKey(self.seed)
#         self.counter = 0
#         return self
#     def __next__(self):
#         """Splits key and draws random numbers from a normal distribution.

#         Returns
#         -------
#         jax.Array
#             array of random numbers drawn from a normal distribution.

#         Raises
#         ------
#         StopIteration
#             Stops iteration if the number of batches has been reached.
#         """
#         #split key to not draw same random numbers every time
#         old_key, self.key = jax.random.split(self.key)
#         #draw random numbers
#         data = jax.random.normal(self.key,shape=self.shape)
#         if self.counter < self.nbatches:
#             self.counter += 1
#             return data
#         else:
#             raise StopIteration
