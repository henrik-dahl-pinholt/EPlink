
import os
import sys
sys.path.insert(0, os.path.abspath("../"))
import EPlink.Compute
import subprocess
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
import dill
import jax



"""Functions to test multi-host code which requires running the same function on two GPU-compatible computers"""


def remote_function(function_to_run):
    
    # Define how to execute function remotely on host2 via SSH
    # Serialize function_to_run using pickle
    serialized_func = dill.dumps(function_to_run)

    # Convert the serialized function to a safe string for the shell command
    serialized_func_str = repr(serialized_func)

    # Name of the conda environment you want to activate
    conda_env_name = "EPlink"

    # SSH command to activate conda environment and execute the Python script
    ssh_command = (
        f'ssh proteome "source ~/miniconda3/etc/profile.d/conda.sh && '
        f'conda activate {conda_env_name} && '
        f'python -c \\"import dill; dill.loads({serialized_func_str})()\\\""'
    )

    # SSH command to execute a Python script on host2 and pass the base64 encoded function
    
    result = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)
    return result#.stdout.strip()



def execute_simultaneously(function_to_run):
        
    def run_dill_encoded(payload):
        fun, args,kwargs = dill.loads(payload)
        return fun(*args,**kwargs)


    def apply_async(pool, fun, args,kwargs):
        payload = dill.dumps((fun, args,kwargs))
        return pool.apply_async(run_dill_encoded, (payload,))

    def runfunc(*args, **kwargs):

        # Start pool process using multiprocessing for asynchronous execution
        p_remote_function = partial(remote_function, function_to_run)
        pool = Pool(processes=1)
        
        # start the remote process 
        kwargs["process_id"] = 1 # set the process id to 1 for the remote process
        remote_result = apply_async(pool, p_remote_function, args, kwargs)
        
        # run the local function
        local_result = function_to_run(*args, **kwargs)
        
        
        # Get the results from the pool (join blocks until all processes are done)
        pool.close()
        # local_out = local_result.get()
        # remote_out = remote_result.get()
        # pool.join()

        return local_result, remote_result
    return runfunc


def test_runner():
    """Tests the runner class"""
    # Tests that the runner class goes through all elements of the iterable.
    iterable = range(100)
    func = lambda x: x
    runner = EPlink.Compute.Runner(iterable,func)
    result = runner.Run(print_stats=False)
    assert len(result) == 100
    assert result == list(iterable)
    
    # Check that the result works while printing stats
    iterable = range(4)
    func = lambda x: x
    runner = EPlink.Compute.Runner(iterable,func)
    result = runner.Run(print_stats=True)
    assert len(result) == 4
    assert result == list(iterable)

    


# print(b.get(),a.get())

