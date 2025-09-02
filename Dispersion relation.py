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
def plot_dispersion(N, Vortex, no_samples, state):
    start_time = time.time()  # Start timer

    r1 = np.linspace(0.01, 1, 70)  # dense region
    r2 = np.linspace(1, 10, 60)[1:] # skip duplicate at 1
    r_values = np.concatenate((r1, r2))

    # Get epsilon(r) and h(r)
    epsilon_vals, h_vals = epsilon_values_and_h(r_values, N, Vortex, no_samples, state)

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

    # Create stacked plots: h(r) on top, epsilon(r) on bottom
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    # Top: h(r)
    axes[0].plot(r_values, h_vals, color='g', linestyle='-', marker='o', markersize=3, label=r'$h(r)$')
    axes[0].set_ylabel(r'$h(r)$')
    axes[0].set_title(f"Electron-vortex correlation function for {state} state (N={N}, samples={no_samples})\nRuntime: {runtime_min:.2f} min")  # <-- Added title
    axes[0].grid(True)
    axes[0].legend()

    # Bottom: epsilon(r) and quadratic fit
    axes[1].plot(r_values, epsilon_vals, marker='o', linestyle='-', color='b', markersize=3, label=r'$\epsilon(r)$')
    axes[1].plot(r_quad, epsilon_quad, linestyle=':', color='r', label=f'Fit: {a:.3g}$r^2$ + {c:.3g}')
    axes[1].set_xlabel(r'$r$')
    axes[1].set_ylabel(r'$\epsilon(r)$')
    limit = 1.2 * np.max(np.abs(epsilon_vals))
    axes[1].set_ylim(-limit, limit)
    axes[1].set_title('CE dispersion relation')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()




# Example usage:
plot_dispersion(N=6, Vortex=True, no_samples =1000000, state = "CEL")
