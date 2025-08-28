###-------------------------------------------------------------------------###
###-------------  Dispersion relation ---------------------###
###-------------------------------------------------------------------------###
"""This file plots the dispersion relation
for the Laughlin state"""
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

def epsilon_values(r_values, N, Vortex, no_samples, state):
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

    return epsilon_vals

import time
def plot_dispersion(N, Vortex, no_samples, state):
    start_time = time.time()  # Start timer

    r1 = np.linspace(0.01, 1, 70)  # dense region
    r2 = np.linspace(1, 5, 30)[1:] # skip duplicate at 1
    r_values = np.concatenate((r1, r2))

    epsilon_vals = epsilon_values(r_values, N, Vortex, no_samples, state)

    # Fit quadratic of the form a*r^2 + c using least squares
    num_fit_points = 10  # fit using first 10 points near origin
    r_fit = r_values[:num_fit_points]
    epsilon_fit = epsilon_vals[:num_fit_points]

    # Design matrix for [r^2, constant]
    A = np.vstack([r_fit**2, np.ones_like(r_fit)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, epsilon_fit, rcond=None)
    a, c = coeffs

    # Generate quadratic for full range
    r_quad = r_values
    epsilon_quad = a * r_quad**2 + c

    # Stop timer and calculate runtime in minutes
    end_time = time.time()
    runtime_min = (end_time - start_time) / 60

    # Plot original data
    plt.figure(figsize=(6, 4))
    plt.plot(r_values, epsilon_vals, marker='o', linestyle='-', color='b',markersize=3, label=r'$\epsilon(r)$')

    # Plot quadratic fit across entire range
    plt.plot(r_quad, epsilon_quad, linestyle=':', color='r', label=f'Fit: {a:.3g}$r^2$ + {c:.3g}')

    plt.xlabel(r'$r$')
    plt.ylabel(r'$\epsilon(r)$')
    plt.ylim((-0.2, 1.2*np.max(epsilon_vals)))
    plt.title(f'Dispersion relation for {state} state (N={N}, Vortex={Vortex}, '
              f'number of samples = {no_samples})\nRuntime: {runtime_min:.2f} min')
    plt.grid(True)
    plt.legend()
    plt.show()




# Example usage:
plot_dispersion(N=6, Vortex=True, no_samples =100000, state = "CEL")
