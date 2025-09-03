import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 5
state = "Laughlin"

# Load saved data
filename = f"data/dispersion_N{N}_{state}.npz"
data = np.load(filename)

# Extract arrays and metadata
r_values = data["r_values"]
epsilon_vals = data["epsilon_vals"]
h_vals = data["h_vals"]
N_meta = int(data["N"])
state_meta = str(data["state"])
no_samples = int(data["no_samples"])
Vortex = bool(data["Vortex"])

# Select fitting range: 0 <= r <= r_max
r_max=0.5
mask = (r_values >= 0) & (r_values <= r_max)
r_fit = r_values[mask]
epsilon_fit = epsilon_vals[mask]

# Fit general polynomial up to degree 4
poly_coeffs = np.polyfit(r_fit, epsilon_fit, deg=4)
epsilon_poly4 = np.polyval(poly_coeffs, r_values)

# Create polynomial equation string in LaTeX
terms = []
for i, coef in enumerate(poly_coeffs):
    power = 4 - i
    if power == 0:
        terms.append(f"{coef:.3g}")
    elif power == 1:
        terms.append(f"{coef:.3g}r")
    else:
        terms.append(f"{coef:.3g}r^{power}")
poly_eq = r"$\epsilon(r)\approx" + " + ".join(terms) + "$"

# Plot
fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
fig.suptitle(f"Electron-vortex correlation function and \n composite-electron dispersion relation \n for N={N_meta}, {state_meta} state, {no_samples} samples")

# h(r) plot
axes[0].plot(r_values, h_vals, 'go-', markersize=3, label=r'$h(r)$')
axes[0].set_ylabel(r'$h(r)$')
axes[0].grid(True)
axes[0].legend()

# epsilon(r) plot
axes[1].plot(r_values, epsilon_vals, 'bo-', markersize=3, label=r'$\epsilon(r)$')
axes[1].plot(r_values, epsilon_poly4, 'm--', label=f'4th-order poly fit up to r={r_max}')
axes[1].set_xlabel(r'$r$')
axes[1].set_ylabel(r'$\epsilon(r)$')
axes[1].set_ylim(min(epsilon_vals) *1.2, max(epsilon_vals) * 1.05)  # Tight y-limits
axes[1].grid(True)
axes[1].legend()

# Add polynomial coefficients as text inside the second plot
axes[1].text(0.05, 0.95, poly_eq, transform=axes[1].transAxes,
             fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
