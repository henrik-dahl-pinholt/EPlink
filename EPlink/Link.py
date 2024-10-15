import jax.numpy as jnp
import numpy as np
from jax import jit, vmap, scipy, lax
from functools import partial
from tqdm import tqdm
from .MS2_HMM import (
    Generate_sample,
    Gen_MS2_measurement,
    swap_inds_for_rates,
    Run_forward_filter,
    Generate_state_map,
)


@jit
def Gaussian(x, a, b):
    # mean, var = jnp.abs(a * b), jnp.abs(a * b**2)
    mean, std = jnp.abs(a), jnp.abs(b) ** 2
    return (1 / jnp.sqrt(2 * jnp.pi * std)) * jnp.exp(-0.5 * (x - mean) ** 2 / std)


Gaussian = vmap(Gaussian, in_axes=(0, None, None))


func_dict = {
    "Gaussian": Gaussian,
}


@partial(jit, static_argnums=(1, 5))
def run_convolution(contacts, window_size, mean, std, dt, weighting_kernel):

    time_differences = jnp.arange(0, window_size + 1) * dt
    weights = func_dict[weighting_kernel](time_differences, mean, std)

    weights = weights / weights.sum()

    return jnp.convolve(contacts, weights, mode="valid")  # contacts@weights


v_run_convolution = vmap(
    run_convolution, (0, None, None, None, None, None)
)  # , static_argnums=(1,)


class EPlinker:
    def __init__(
        self,
        EPtrajectories,
        window_size,
        dt_polymer,
        upscale_factor,
        MS2_window,
        tau,
        npromoter_states=2,
        weighting_kernel="Gaussian",
    ):
        self.nstates = npromoter_states
        self.MS2_window = MS2_window
        self.tau = tau
        self.trajectory = EPtrajectories
        self.window_size = window_size
        self.dt_polymer = dt_polymer
        self.weighting_kernel = weighting_kernel
        self.upscale_factor = upscale_factor

        self.smap, self.state_sequences = Generate_state_map(
            self.nstates, self.MS2_window
        )

    def Compute_Onrate(self, params, coarse_grain=True):
        if coarse_grain:
            # contacts = jnp.exp(-self.trajectory**2 / (2 * params[0] ** 2))

            # contact_blocks = contacts.reshape(
            #     self.trajectory.shape[0],
            #     contacts.shape[1] // self.upscale_factor,
            #     self.upscale_factor,
            # )
            # coarse_grained_contacts = contact_blocks.mean(axis=2)
            coarse_grained_contacts = self.Compute_contacts(
                params[0], coarse_grain=True
            )
            k_ons = v_run_convolution(
                coarse_grained_contacts,
                self.window_size,
                params[1],
                params[2],
                self.dt_polymer * self.upscale_factor,
                self.weighting_kernel,
            )

        else:
            # contacts = jnp.exp(-self.trajectory**2 / (2 * params[0] ** 2))
            contacts = self.Compute_contacts(params[0], coarse_grain=False)
            k_ons_fine = v_run_convolution(
                contacts,
                self.window_size * self.upscale_factor,
                params[1],
                params[2],
                self.dt_polymer,
                self.weighting_kernel,
            )
            rate_blocks = k_ons_fine.reshape(
                self.trajectory.shape[0],
                k_ons_fine.shape[1] // self.upscale_factor,
                self.upscale_factor,
            )

            k_ons = rate_blocks.mean(axis=2)
        return k_ons

    def Compute_contacts(self, rc, coarse_grain=True):
        if coarse_grain:
            contacts = jnp.exp(-self.trajectory**2 / (2 * rc**2))

            contact_blocks = contacts.reshape(
                self.trajectory.shape[0],
                contacts.shape[1] // self.upscale_factor,
                self.upscale_factor,
            )
            coarse_grained_contacts = contact_blocks.mean(axis=2)
        else:
            contacts = jnp.exp(-self.trajectory**2 / (2 * rc**2))
            coarse_grained_contacts = contacts
        return coarse_grained_contacts

    def Sample_trajectory(
        self,
        nsamples,
        kernel_params,
        promoter_params,
        pol2_rates,
        measurement_error,
        state_0=None,
        nstates=2,
        verbose=False,
        seed=None,
        coarse_grain=True,
    ):
        kon, koff = promoter_params
        k_ons = self.Compute_Onrate(kernel_params, coarse_grain=coarse_grain) * kon

        if state_0 is None:
            avg_on = k_ons.mean()
            p_on = avg_on / (avg_on + koff)
            state_0 = jnp.array([1 - p_on, p_on])
        true_out = []
        MS2_out = []
        MS2_out_no_noise = []
        for k_val in tqdm(k_ons):
            samps = Generate_sample(
                k_val,
                koff,
                state_0,
                nsamples,
                nstates,
                self.dt_polymer * self.upscale_factor,
                seed=seed,
                verbose=verbose,
            )
            true_out.append(samps)
            prates = swap_inds_for_rates(samps, pol2_rates)
            _, MS2_measurement = Gen_MS2_measurement(
                prates,
                (self.MS2_window, self.tau, measurement_error),
                self.dt_polymer * self.upscale_factor,
            )
            MS2_out.append(MS2_measurement[2])
            MS2_out_no_noise.append(MS2_measurement[1])
            MS2_times = MS2_measurement[0]
            promoter_times = (
                jnp.arange(samps.shape[-1]) * self.dt_polymer * self.upscale_factor
            )
        return (
            promoter_times,
            MS2_times,
            k_ons,
            jnp.array(true_out),
            jnp.array(MS2_out_no_noise),
            jnp.array(MS2_out),
        )

    def Set_MS2_data(self, MS2_data):
        self.MS2_data = MS2_data

    def LLH(self, params, coarse_grain=True, verbose=False):
        # check if MS2 data is set
        if not hasattr(self, "MS2_data"):
            raise ValueError("MS2 data not set")
        # load params
        kernel_params = params[:3]
        kon, koff = params[3:5]
        loading_rates = params[5:7]
        measurement_error = params[7]

        # compute on rates
        k_ons = self.Compute_Onrate(kernel_params, coarse_grain=coarse_grain) * kon

        # compute likelihood
        posteriors, LLH = Run_forward_filter(
            self.state_sequences,
            koff,
            k_ons,
            self.MS2_data,
            measurement_error,
            loading_rates,
            self.MS2_window,
            self.tau,
            self.dt_polymer * self.upscale_factor,
            verbose=verbose,
        )
        return LLH

    def Viterbi(self, params, coarse_grain=True, verbose=False):
        # check if MS2 data is set
        if not hasattr(self, "MS2_data"):
            raise ValueError("MS2 data not set")
        # load params
        kernel_params = params[:3]
        kon, koff = params[3:5]
        loading_rates = params[5:7]
        measurement_error = params[7]

        # compute on rates
        k_ons = self.Compute_Onrate(kernel_params, coarse_grain=coarse_grain) * kon

        # compute likelihood
        posteriors, LLH, path, unwrapped_path = Run_forward_filter(
            self.state_sequences,
            koff,
            k_ons,
            self.MS2_data,
            measurement_error,
            loading_rates,
            self.MS2_window,
            self.tau,
            self.dt_polymer * self.upscale_factor,
            verbose=verbose,
            compute_viterbi=True,
        )
        return LLH, unwrapped_path
