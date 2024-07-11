"""The data in this folder has been generated using the following code."""

import sys,os
CUDA_VISIBLE_DEVICES=0


# add the path to the library and load it
sys.path.insert(0, os.path.abspath("../../../"))
import numpy as np
from EPlink import Polymer_inference as PI
from tqdm import tqdm
savepath = "/net/levsha/share/hdp/Simulation_data/Projects/E-P_inference/2024_04_13_Rouse_LLHcomp_numerically_stable"


n_traj = 3
# These parameters corresponds to a bead diameter of ~40nm and approximately 3000bp per bead as far as I recall
D = 2810  # nm^2/s
k = 1 / 0.177  # 1/s
dt = 17.7
T = 7200
N = 167
w = np.zeros(N)
w[N//3] = 1
w[2*N//3] = -1
locerrs = np.array([50,50,120]) #nm

n_datapoints = 5
n_posteriorsamps = 5
for i in range(n_datapoints):
    print(f"Generating data point {i}/{n_datapoints} ({(i/n_datapoints)*100:.2f}%)")
    data_steps = int(T/dt)
    fine_steps = data_steps*1000
    fine_times,traj_fine = PI.Generate_trajectory(fine_steps,dt/1000,n_traj,k,D,N,seed = i,verbose=True)
    fine_times_cropped,traj_cropped = fine_times[:(data_steps-1)*1000+1],traj_fine[:(data_steps-1)*1000+1]
    times,traj = fine_times[::1000][:data_steps],traj_fine[::1000][:data_steps]
    
    ep_traj,ep_traj_w_err = PI.Generate_measurements(traj,w,locerrs)
    ep_traj_fine,ep_traj_w_err_fine = PI.Generate_measurements(traj_fine,w,locerrs)
    Posterior_Sampler = PI.PosteriorSampler(N,k,D,ep_traj_w_err,times,locerrs,w)
    traj_time,trajs = Posterior_Sampler.sample(n_posteriorsamps,dt/1000,verbose=True)#1/k/10)
    np.save(f"ep_traj_{i}.npy",ep_traj)
    np.save(f"ep_traj_fine_{i}.npy",ep_traj_fine)
    np.save(f"ep_traj_w_err_{i}.npy",ep_traj_w_err)
    np.save(f"Post_samps_{i}.npy",trajs.swapaxes(0,1).swapaxes(1,2))

    if i == 0:
        np.save(f"simulation_times.npy",traj_time)
        np.save(f"observation_times.npy",times)
        
