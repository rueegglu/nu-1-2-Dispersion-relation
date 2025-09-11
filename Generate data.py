###-------------------------------------------------------------------------###
###-------------  Generate data ---------------------###
###-------------------------------------------------------------------------###
"""This file generates the data,
then saves it in a .npz file, ready
to be plotted."""
###-------------------------------------------------------------------------###
###-------------------------- Import Modules -------------------------------###
###-------------------------------------------------------------------------###


import numpy as np
from Electron_density_function import electron_density_values
from scipy.special import ellipk
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Wrapper for elliptic integral (SciPy uses m = k^2)
def K(k):
    return ellipk(np.asarray(k, dtype=float)**2)

def epsilon_values_and_h(r_values, N, Vortex, no_samples, state):
    r0 = max(r_values)
    # h(r) = electron density values on the given grid
    h_values = electron_density_values(r_values, no_samples, N, Vortex, state)

    # Interpolate h(r) for continuous integration
    h_interp = interp1d(r_values, h_values, kind='linear', fill_value=0.0, bounds_error=False)

    epsilon_vals = np.zeros_like(r_values)

    for i, r in enumerate(r_values):
        if r == 0.0:  # special case: r=0
            val, _ = quad(lambda rp: h_interp(rp), 0.0, r0)
            epsilon_vals[i] = 2 * np.pi * val
            continue

        def integrand(rp):
            denom = r + rp
            m = 4.0 * r * rp / (denom ** 2)
            m = np.clip(m, 0.0, 1.0 - 1e-12)
            return h_interp(rp) * rp * (4.0 / denom) * ellipk(m)

        val, _ = quad(integrand, 0.0, r0, points=[r])
        epsilon_vals[i] = val

    return epsilon_vals , h_values



import time
def generate_and_save_data(N, Vortex, no_samples, state):
    start_time = time.time()  # Start timer

    r1 = np.linspace(0.01, 1, 70)  # dense region
    r2 = np.linspace(1, 10, 60)[1:] # skip duplicate at 1
    r_values = np.concatenate((r1, r2))

    # Get epsilon(r) and h(r)
    epsilon_vals, h_vals = epsilon_values_and_h(r_values, N, Vortex, no_samples, state)
    
    import os

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
    Vortex=Vortex
)


    runtime_min = (time.time() - start_time) / 60
    print(f"Data saved to {filename}")
    print(f"Runtime: {runtime_min:.2f} minutes")


# Example usage
generate_and_save_data(N=6
                       , Vortex=True, no_samples=10000, state="Laughlin")