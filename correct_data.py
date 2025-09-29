import numpy as np
from scipy.special import ellipk
from scipy.interpolate import interp1d
from scipy.integrate import quad
import os
from tqdm import tqdm # For a progress bar during recalculation

# --- Configuration ---
# Just change this filename to the path of the file you want to correct.
INPUT_FILENAME = "data/dispersion_N6_CEL_vec.npz" 
# ---------------------

# Wrapper for elliptic integral, needed for recalculating epsilon
def K(k):
    """SciPy's elliptic integral uses the parameter m = k^2."""
    return ellipk(np.asarray(k, dtype=float)**2)

def correct_data_file(filename):
    """
    Loads a data file with unnormalized electron density, corrects it,
    recalculates the dispersion, and saves a new file.
    """
    if not os.path.exists(filename):
        print(f"Error: File not found at '{filename}'")
        return

    # 1. Define the name for the new, corrected file
    base, ext = os.path.splitext(filename)
    output_filename = f"{base}{ext}"

    # 2. Load the incorrect data
    print(f"Loading data from '{filename}'...")
    with np.load(filename) as data:
        # The 'h_vals' in the old file are actually the raw, unnormalized rho_values
        raw_rho_values = data['h_vals']
        r_values = data['r_values']
        # We will keep the other metadata as is
        N = data['N']
        state = data['state']
        no_samples = data['no_samples']

    print("Data loaded successfully.")
    print(f"Shape of h_vals (raw rho): {raw_rho_values.shape}")
    print(f"Value at largest r (rho_infinity): {raw_rho_values[-1]:.6f}")

    # 3. --- FIX 1: Normalize h_vals correctly ---
    print("\nNormalizing h_vals...")
    rho_infinity = raw_rho_values[-1]
    corrected_h_vals = (raw_rho_values - rho_infinity) / rho_infinity
    print("Normalization complete.")
    print(f"New last value of h_vals (should be 0.0): {corrected_h_vals[-1]:.6f}")

    # 4. --- FIX 2: Recalculate epsilon_vals from corrected h_vals ---
    print("\nRecalculating epsilon_vals from the corrected h_vals...")
    r0 = np.max(r_values)
    # Create a new interpolation function with the CORRECT h_values
    h_interp = interp1d(r_values, corrected_h_vals, kind='linear', fill_value=0.0, bounds_error=False)
    
    recalculated_epsilon_vals = np.zeros_like(r_values)

    # Loop through r_values to perform the integration
    for i in tqdm(range(len(r_values)), desc="Integrating"):
        r = r_values[i]
        if r == 0.0:
            val, _ = quad(lambda rp: h_interp(rp), 0.0, r0)
            recalculated_epsilon_vals[i] = 2 * np.pi * val
            continue

        def integrand(rp):
            denom = r + rp
            m = 4.0 * r * rp / (denom**2)
            # Clip m to avoid domain errors in ellipk for values exactly 1.0
            m = np.clip(m, 0.0, 1.0 - 1e-15)
            return h_interp(rp) * rp * (4.0 / denom) * K(np.sqrt(m))

        val, _ = quad(integrand, 0.0, r0, points=[r])
        recalculated_epsilon_vals[i] = val
    
    print("Recalculation of epsilon_vals complete.")

    # 5. Save all the corrected data to the new file
    print(f"\nSaving corrected data to '{output_filename}'...")
    np.savez(
        output_filename,
        r_values=r_values,
        epsilon_vals=recalculated_epsilon_vals, # Use the new values
        h_vals=corrected_h_vals,               # Use the new values
        N=N,
        state=state,
        no_samples=no_samples,
    )
    print("Done!")

# --- Main execution block ---
if __name__ == '__main__':
    correct_data_file(INPUT_FILENAME)
