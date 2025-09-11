import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# High-quality plotting settings
plt.rcParams.update({
    "figure.dpi": 150,         
    "font.size": 12,           
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.markersize": 4,
})

# Parameters
N = 6
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

# Fitting range
r_max = 1
mask_fit = (r_values >= 0) & (r_values <= r_max)
r_fit = r_values[mask_fit]
epsilon_fit = epsilon_vals[mask_fit]

# Polynomial fit
poly_coeffs = np.polyfit(r_fit, epsilon_fit, deg=4)
epsilon_poly4 = np.polyval(poly_coeffs, r_values)

# Polynomial equation string in LaTeX
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

# Restrict epsilon plotting range
r_plot_max = 1.0
mask_eps = (r_values >= 0) & (r_values <= r_plot_max)
r_eps = r_values[mask_eps]
epsilon_plot = epsilon_vals[mask_eps]
epsilon_poly4_plot = epsilon_poly4[mask_eps]

# Main plot: epsilon(r)
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle(f"Composite-electron dispersion relation \n"
              f"with inset electron-vortex correlation function\n"
             f"N={N_meta}, {state_meta} state, {no_samples} samples", fontsize=14)

ax.plot(r_eps, epsilon_plot, 'bo-', label=r'$\epsilon(r)$')
ax.plot(r_eps, epsilon_poly4_plot, 'm--', label=f'4th-order poly fit up to r={r_max}')
ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$\epsilon(r)$')
ax.set_ylim(-0.25, 5.25)
ax.grid(True)
ax.legend()

# Polynomial coefficients text (top left)
ax.text(0.05, 0.95, poly_eq, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.8))

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Inset plot: h(r) (bottom right, slightly higher), smooth thin line
ax_inset = inset_axes(
    ax,
    width="40%",
    height="40%",
    bbox_to_anchor=(0.35, 0.05, 0.5, 0.7),
    bbox_transform=ax.transAxes
)

# Create smooth curve
from scipy.ndimage import gaussian_filter1d

# Choose standard deviation for smoothing
sigma = 0.1  # larger sigma = smoother curve

# Apply Gaussian smoothing to h_vals
h_smooth = gaussian_filter1d(h_vals, sigma=sigma)

# Plot in the inset (thin line, no markers)
ax_inset.plot(r_values, h_smooth, 'g-', lw=1)


ax_inset.set_title(r'$h(r)$', fontsize=10)
ax_inset.set_ylim(-1, 0.4)
ax_inset.set_yticks([-1, 0])  # keep only -1 and 0
ax_inset.grid(True)
ax_inset.tick_params(axis='both', which='major', labelsize=8)




plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

