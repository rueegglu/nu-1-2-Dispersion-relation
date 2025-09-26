import numpy as np
import math

# --- Helper functions (mostly unchanged) ---
# It's good practice to use np.math.comb for newer NumPy/Python versions
# but your nCr works fine.
def nCr(n, r):
    if not (float(n).is_integer() and float(r).is_integer()):
        raise TypeError("n and r must be integers or integer-valued floats.")
    n, r = int(n), int(r)
    return 0 if n < 0 or r < 0 or r > n else math.comb(n, r)

def factorial(x):
    if not float(x).is_integer():
        raise TypeError("factorial input must be an integer or integer-valued float.")
    x = int(x)
    if x < 0: raise ValueError("factorial input must be non-negative.")
    return math.factorial(x)

pi = np.pi
Q = 1/2 # Effective charge

def Normalisation(q, n, m):
    # This function doesn't depend on sample data, so it remains unchanged.
    return np.sqrt(
        (2 * q + 2 * n + 1) / (4 * pi)
        * factorial(q + n - m) * factorial(q + n + m)
        / factorial(n) / factorial(2 * q + n)
    )

# --- Vectorized Wavefunction Components ---

def R_vectorized(j, a, b, u, v):
    """
    Vectorized R function.
    j: index of the particle of interest.
    a, b: integer powers.
    u, v: spinor coordinates of ALL particles for ALL samples.
          Shape: (no_samples, N)
    """
    no_samples, N = u.shape

    # Select the u_j and v_j for all samples and expand dims for broadcasting
    # Shapes will be (no_samples, 1) to operate against the (no_samples, N) arrays.
    u_j = u[:, j, np.newaxis]
    v_j = v[:, j, np.newaxis]

    # Calculate the denominator for all k at once. Shape: (no_samples, N)
    # np.where handles the k=j case to avoid division by zero.
    denominator = u_j * v - v_j * u
    inv_denominator = np.where(denominator != 0, 1.0 / denominator, 0)

    if a == 0 and b == 0: return np.ones(no_samples)
    if a == 1 and b == 0: return np.sum(v * inv_denominator, axis=1)
    if a == 0 and b == 1: return np.sum(-u * inv_denominator, axis=1)
    
    # For higher-order terms, calculate sums first
    sum_a1_b0 = np.sum(v * inv_denominator, axis=1)
    sum_a0_b1 = np.sum(u * inv_denominator, axis=1)
    
    if a == 2 and b == 0:
        sum_sq = np.sum((v * inv_denominator)**2, axis=1)
        return sum_a1_b0**2 - sum_sq
    if a == 0 and b == 2:
        sum_sq = np.sum((u * inv_denominator)**2, axis=1)
        return sum_a0_b1**2 - sum_sq
    if a == 1 and b == 1:
        sum_prod = np.sum((u * v) * inv_denominator**2, axis=1)
        return -sum_a1_b0 * sum_a0_b1 + sum_prod

    raise ValueError(f"R function not defined for a={a}, b={b}")

def tilde_Y_vectorized(i, j, u, v):
    """
    Vectorized tilde_Y. Computes the (i,j)-th matrix element for all samples.
    u, v shape: (no_samples, N)
    Returns an array of shape (no_samples,).
    """
    no_samples, N = u.shape
    q = N - 1/2
    
    # Get n, m from lookup table (this part is not vectorized, which is fine)
    nm_values = [(0, 1/2), (0, -1/2), (1, 3/2), (1, 1/2), (1, -1/2), (1, -3/2),
                 (2, 5/2), (2, 3/2), (2, 1/2), (2, -1/2), (2, -3/2), (2, -5/2)]
    n, m = nm_values[i]
    
    # --- Part 1: Normalisation factor ---
    # This is a single scalar value, same for all samples.
    norm_factor = (Normalisation(Q, n, m)
                   * (-1)**(Q + n - m)
                   * factorial(2 * q + 1)
                   / factorial(2 * q + n + 1))
    
    # --- Part 2: j-dependent terms ---
    # Select u_j and v_j for all samples. Shape: (no_samples,)
    u_j = u[:, j]
    v_j = v[:, j]
    
    e_to_i_phi_j = v_j**2 + np.conjugate(u_j)**2
    
    # `total` is now an array of shape (no_samples,)
    total = norm_factor * (u_j**(Q + m) * v_j**(Q - m) * e_to_i_phi_j**Q)
    
    # --- Part 3: Sum over s ---
    # The loop over s is small and independent of N, so keeping it is fine.
    s_sum = np.zeros(no_samples, dtype=complex)
    for s in range(n + 1):
        # The call to R is now to the vectorized version
        r_vals = R_vectorized(j, s, n - s, u, v)
        term = ((-1)**s
                * nCr(n, s)
                * nCr(2 * Q + n, Q + n - m - s)
                * v_j**(n - s)
                * u_j**s
                * r_vals)
        s_sum += term
        
    return total * s_sum

def determinant_factor_vectorized(u, v):
    no_samples, N = u.shape
    
    # Create a stack of N x N matrices, one for each sample.
    # Shape: (no_samples, N, N)
    matrix_stack = np.zeros((no_samples, N, N), dtype=complex)
    
    for i in range(N):
        for j in range(N):
            # Each call to tilde_Y fills the (i, j) element for all samples.
            matrix_stack[:, i, j] = tilde_Y_vectorized(i, j, u, v)
    
    # np.linalg.det works on stacks of matrices!
    return np.linalg.det(matrix_stack)

def Jastrow_factor_vectorized(u, v):
    no_samples, N = u.shape
    
    # Use broadcasting to create (N, N) matrices for each sample
    u_i = u[:, :, np.newaxis] # Shape: (no_samples, N, 1)
    u_j = u[:, np.newaxis, :] # Shape: (no_samples, 1, N)
    v_i = v[:, :, np.newaxis]
    v_j = v[:, np.newaxis, :]

    # Term 1: (u_i * v_j - u_j * v_i)**2
    pair_term = (u_i * v_j - u_j * v_i)**2

    # Term 2: e_to_i_phi factors
    e_phi = v**2 + np.conjugate(u)**2
    e_phi_i = e_phi[:, :, np.newaxis]
    e_phi_j = e_phi[:, np.newaxis, :]

    # Combine all pairwise terms. Shape: (no_samples, N, N)
    all_pairs_matrix = pair_term * e_phi_i * e_phi_j

    # We only want the product for i < j. We can get the upper triangle indices.
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Select the upper triangle elements for all samples
    upper_triangle_terms = all_pairs_matrix[:, i_idx, j_idx]
    
    # Return the product over these pairs for each sample
    return np.prod(upper_triangle_terms, axis=1)

def vortex_factor_vectorized(u, v, Vortex):
    if not Vortex:
        return 1
    
    # This is a simple element-wise calculation
    mod_z_values = np.abs(v / u)
    term = mod_z_values**2 / (1 + mod_z_values**2)
    
    # Product along the particle axis (axis=1)
    return np.prod(term, axis=1)**2

def Psi_CEL_vec(positions, N, Vortex):
    """
    Fully vectorized version of the CEL wavefunction calculation.
    positions: A NumPy array of particle coordinates.
               Shape: (no_samples, N, 2)
    """
    # 1. Convert all spherical coordinates to spinor coordinates at once
    theta = positions[..., 0] # Shape: (no_samples, N)
    phi = positions[..., 1]   # Shape: (no_samples, N)
    
    u = np.cos(theta / 2) * np.exp(-1j * phi / 2)
    v = np.sin(theta / 2) * np.exp(1j * phi / 2)
    
    # 2. Calculate each factor using the vectorized functions
    jastrow = Jastrow_factor_vectorized(u, v)
    determinant = determinant_factor_vectorized(u, v)
    vortex = vortex_factor_vectorized(u, v, Vortex)
    
    # 3. Combine results
    return jastrow * determinant * vortex