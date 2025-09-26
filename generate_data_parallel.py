###-------------------------------------------------------------------------###
###-------------  Generate data (Parallel Version) ---------------------###
###-------------------------------------------------------------------------###
"""This file generates the data,
then saves it in a .npz file, ready
to be plotted."""
###-------------------------------------------------------------------------###
###-------------------------- Import Modules -------------------------------###
###-------------------------------------------------------------------------###

import numpy as np
from Electron_density_function import electron_density_values, Psi_CEL_vec
from scipy.special import ellipk
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
import time
import os
import concurrent.futures
from functools import partial
from tqdm import tqdm # For progress bars

# Wrapper for elliptic integral (SciPy uses m = k^2)
def K(k):
    return ellipk(np.asarray(k, dtype=float)**2)

# --- Worker Function for the expensive Monte Carlo step (with BATCHING) ---
def _worker_h_values(r_value, no_samples, N, state, V):
    """
    Worker that calculates electron density for a single r_value.
    It processes the large number of samples in smaller batches to save memory.
    """
    np.random.seed() # Re-seed for each process

    # --- Batching Parameters ---
    # Choose a batch size that fits comfortably in memory. 1 million is a good start.
    # 1M samples * (6-1) * 2 * 8 bytes ≈ 80 MB, which is very safe.
    batch_size = 1_000_000
    num_batches = int(np.ceil(no_samples / batch_size))
    
    total_value = 0.0

    # The new batching loop
    for _ in range(num_batches):
        # This is the code from your original worker, but using batch_size
        X_samples = np.empty((batch_size, N - 1, 2))
        X_samples[..., 0] = np.random.uniform(0, np.pi, (batch_size, N - 1))
        X_samples[..., 1] = np.random.uniform(0, 2 * np.pi, (batch_size, N - 1))

        jacobian_factors = np.prod(np.sin(X_samples[..., 0]), axis=1)

        theta = 2 * np.arctan(r_value)
        Omega = np.array([theta, 0.0])
        Omega_expanded = np.broadcast_to(Omega, (batch_size, 1, 2))
        positions = np.concatenate([Omega_expanded, X_samples], axis=1)

        # Assuming Psi_CEL_vec is the correct vectorized wavefunction function
        # You'll need to import it or define it in this file.
        psi_vals = Psi_CEL_vec(positions, N, True) 

        # Sum the mean of the values for this batch, weighted by batch size
        total_value += np.sum(jacobian_factors * np.abs(psi_vals)**2)

    # Calculate the final mean across all samples
    mean_value = total_value / no_samples
    rho_value = V * mean_value
    
    return rho_value

# --- Worker Function for the integration step ---
def _worker_epsilon_values(r, r0, h_interp):
    """
    Worker that performs the numerical integration for a single r_value.
    """
    if r == 0.0:
        val, _ = quad(lambda rp: h_interp(rp), 0.0, r0)
        return 2 * np.pi * val

    def integrand(rp):
        denom = r + rp
        m = 4.0 * r * rp / (denom ** 2)
        # Clip m to avoid domain errors in ellipk for values === 1.0
        m = np.clip(m, 0.0, 1.0 - 1e-12)
        return h_interp(rp) * rp * (4.0 / denom) * ellipk(m)

    val, _ = quad(integrand, 0.0, r0, points=[r])
    return val

# --- Main function rewritten to use the parallel workers ---
def epsilon_values_and_h(r_values, N, no_samples, state):
    # Define V here so it can be passed to the workers.
    V = (2 * np.pi**2)**(N - 1)
    # --- Part 1: Parallel calculation of h(r) ---
    print("Step 1/2: Calculating electron density (h_values) in parallel...")
    # Use functools.partial to pre-fill the arguments that are the same for every job
    h_worker_func = partial(_worker_h_values, no_samples=no_samples, N=N, state=state, V=V)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # map applies the worker function to each r_value across all available CPU cores
        h_results = list(tqdm(executor.map(h_worker_func, r_values), total=len(r_values)))
    
    h_values = np.array(h_results)

    # --- Part 2: Parallel calculation of epsilon(r) ---
    print("\nStep 2/2: Calculating epsilon values in parallel...")
    r0 = max(r_values)
    # Interpolate h(r) for continuous integration (this is fast)
    h_interp = interp1d(r_values, h_values, kind='linear', fill_value=0.0, bounds_error=False)

    epsilon_worker_func = partial(_worker_epsilon_values, r0=r0, h_interp=h_interp)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        epsilon_results = list(tqdm(executor.map(epsilon_worker_func, r_values), total=len(r_values)))
    
    epsilon_vals = np.array(epsilon_results)

    return epsilon_vals, h_values

def generate_and_save_data(N, no_samples, state):
    start_time = time.time()  # Start timer

    r1 = np.linspace(0.01, 1, 70)  # dense region
    r2 = np.linspace(1, 10, 60)[1:] # skip duplicate at 1
    r_values = np.concatenate((r1, r2))

    # Get epsilon(r) and h(r) using the new parallel function
    epsilon_vals, h_vals = epsilon_values_and_h(r_values, N, no_samples, state)

    # Directory to store data
    os.makedirs("data", exist_ok=True)
    
    # Save as .npz
    filename = f"data/dispersion_N{N}_{state}.npz"
    np.savez(
        filename,
        r_values=r_values,
        epsilon_vals=epsilon_vals,
        h_vals=h_vals,
        N=N,
        state=state,
        no_samples=no_samples,
    )

    runtime_min = (time.time() - start_time) / 60
    print(f"\nData saved to {filename}")
    print(f"Total Runtime: {runtime_min:.2f} minutes")


# --- Main execution block ---
# This "if __name__ == '__main__':" guard is ESSENTIAL for multiprocessing
if __name__ == '__main__':
    generate_and_save_data(N=6, no_samples=int(1E9), state="CEL_vec")