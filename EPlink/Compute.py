import psutil
import GPUtil
from tqdm import tqdm
import jax
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
"""This module contains code to simplify batch evaluation of functions from other modules in large computations."""

class Runner:
    """This class contains the runner class. An object sought to streamline batch evaluation of large computations in a common framework which easily works with JAXs computational model. The class was introduced to display progress, allow batch computation to combat memory overloads, and to excecute jax code in a distributed fashion across multiple hosts. 
    
    The runner object takes an iterable, and a function upon initialization. The function is called on each element of the iterable to perform the computation. As an optional argument, an aggregation function can be passed to the class in which case outputs are collected in a list and passed to the aggregation function when the iterator is empty. By letting it be up to the iterable to feed data, the size of the calculations can be easily managed, and parallelization across GPUs is easily accomplished by sharding the data fed by the iterable."""
    def __init__(self,iterable,func,aggregation_func=None):
        """Constructor for the Runner object.
        
        Parameters
        ----------
        iterable : iterable
            The iterable to be fed to the function func. 
        func : function
            The function to be called on each element of the iterable. 
        aggregation_func : function, optional
            The function to be called on a list of the collected result of func on the elements of the iterator. Should take as input a list of the results. If specified, calling the Run() method will return the result of this function. If not specified, the Run() method will just return the list of outputs directly.
        """
        
        self.iterable = iterable
        self.func = func
        self.agg_f = aggregation_func


    def get_system_metrics(self):
        """Grabs system metrics such as CPU usage, memory usage, GPU usage, GPU memory usage, GPU memory used, and GPU memory total.

        Returns
        -------
        tuple
            A tuple containing the CPU usage (in percent), memory usage (in percent), GPU usage (in percent), GPU memory used (in MB), GPU memory total (in MB), and GPU memory usage (in percent).
        """
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_usage = gpu.load * 100
            gpu_memory_used = gpu.memoryUsed
            gpu_memory_total = gpu.memoryTotal
            gpu_memory_usage = gpu.memoryUtil * 100
        else:
            gpu_usage = 0
            gpu_memory_used = 0
            gpu_memory_total = 0
            gpu_memory_usage = 0
        return cpu_usage, memory_usage, gpu_usage, gpu_memory_used, gpu_memory_total, gpu_memory_usage

    def Run(self,print_stats=True):
        """Runs the function on the iterable.

        Parameters
        ----------
        print_stats : bool, optional
            Whether to print system statistics during the run. Default is True. Keep in mind that getting system statistics makes iterations take longer (~1s more) which may be significant depending on how long a function call takes. Default is True.

        Returns
        -------
        list or any
            If an aggregation function was specified, the result of this function is returned. Otherwise, the list of outputs is returned.
        """
        results = []
        for i in tqdm(self.iterable):
            if print_stats:
                # Print usage statistics
                cpu_usage, memory_usage, gpu_usage, gpu_memory_used, gpu_memory_total, gpu_memory_usage = self.get_system_metrics()
                tqdm.write(f"CPU Usage: {cpu_usage:4.2f}% | Memory Usage: {memory_usage:4.2f}% | GPU Usage: {gpu_usage:4.2f}% | GPU Memory Usage: {gpu_memory_usage:4.2f}% ({gpu_memory_used:4.2f}/{gpu_memory_total:4.2f} MB)")
                    
            # Run the function
            results.append(self.func(i))
            
        # Post-process the result
        if not self.agg_f is None:
            return self.agg_f(results)
        else:
            return results
            

class ZeroGenerator:
    def __init__(self,zero_shape,nbatches):
        self.nbatches = nbatches
        self.counter = 0
        self.shape = zero_shape
    
    def __iter__(self):
        self.counter = 0
        return self
    def __next__(self):
        data = jax.numpy.zeros(self.shape)
        if self.counter < self.nbatches:
            self.counter += 1
            return data
        else:
            raise StopIteration
