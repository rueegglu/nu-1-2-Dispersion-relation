###-------------------------------------------------------------------------###
###------------- Vectorised Electron Density for Laughlin ------------------###
###-------------------------------------------------------------------------###
import numpy as np
import time
from Laughlin_wavefunction import Psi_Laughlin
from CEL_wavefunction import Psi_CEL

pi = np.pi

def electron_density_values(r_values, no_samples, N, state):
    """
    Compute electron density values for given r_values using vectorised sampling.
    r_values : array of radii
    no_samples : number of Monte Carlo samples
    N : number of electrons
    state : 'Laughlin' or 'CEL'
    """
    theta_values = 2 * np.arctan(r_values)
    V = (2 * pi**2)**(N - 1)
    rho_values = np.zeros_like(r_values)

    start_time = time.time()

    # Choose which wavefunction to use
    if state == "Laughlin":
        psi_func = Psi_Laughlin
    elif state == "CEL":
        psi_func = Psi_CEL
    else:
        raise ValueError("state must be 'Laughlin' or 'CEL'.")

    for i, theta in enumerate(theta_values):
        iter_start = time.time()

        # ---- 1. Generate all samples in one go ----
        X_samples = np.empty((no_samples, N-1, 2))
        X_samples[..., 0] = np.random.uniform(0, pi,  (no_samples, N-1))
        X_samples[..., 1] = np.random.uniform(0, 2*pi, (no_samples, N-1))

        # ---- 2. Compute Jacobian factors ----
        jacobian_factors = np.prod(np.sin(X_samples[...,0]), axis=1)  # shape: (no_samples,)

        # ---- 3. Build positions array for all samples ----
        Omega = np.array([theta, 0.0])  # fixed probe particle
        Omega_expanded = np.broadcast_to(Omega, (no_samples,1,2))
        positions = np.concatenate([Omega_expanded, X_samples], axis=1)  # (no_samples,N,2)

        # ---- 4. Evaluate the wavefunction in batches ----
        try:
            # If Psi supports batch input: pass positions directly
            psi_vals = psi_func(positions, N, True)
        except Exception:
            # Fallback: evaluate sample by sample
            psi_vals = np.array([psi_func(p, N, True) for p in positions])

        # ---- 5. Accumulate ----
        values = jacobian_factors * np.abs(psi_vals)**2
        rho_values[i] = V * np.mean(values)

        # Timing estimates
        if i == 0:
            single_time = time.time() - iter_start
            est_total = single_time * len(r_values)
            print(f"Estimated total runtime: {est_total:.2f} s ({est_total/60:.2f} min)")
        progress = (i+1)/len(r_values)*100
        print(f"{i+1}/{len(r_values)} completed ({progress:.2f}%)")

    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} s ({total_time/60:.2f} min)")

    # ---- 6. Normalisation ----
    rho_values = (rho_values - rho_values[-1]) / rho_values[-1]

    return rho_values


###-------------------------------------------------------------------------###
###-------------------------- Testing Utility ------------------------------###
###-------------------------------------------------------------------------###
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def plot_electron_density(N=3, no_samples=2000, state="Laughlin"):
        r_values = np.linspace(0.01, 7, 50)  # fewer points for a quick test
        rho_values = electron_density_values(r_values, no_samples, N, state)
        plt.figure(figsize=(6, 4))
        plt.plot(r_values, rho_values, 'bo-', label=fr'$\rho(r)$')
        plt.xlabel(r'$r$')
        plt.ylabel(r'$\rho(r)$')
        plt.title(f'Electron Density ({state}), N={N}, samples={no_samples}')
        plt.grid(True)
        plt.legend()
        plt.ylim((-1.2, 1.2 * np.max(rho_values)))
        plt.show()

    plot_electron_density()
