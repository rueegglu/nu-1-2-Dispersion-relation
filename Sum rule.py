import numpy as np

#parameters
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

# Compute integral using the trapezoidal rule
integral = np.trapz(h_vals * r_values, r_values)

print(f"Sum rule integral ∫ h(r) r dr = {integral:.4f}")

# Check closeness to -2
if np.isclose(integral, -2, atol=0.1):
    print("✅ Sum rule is approximately satisfied.")
else:
    print("⚠️ Sum rule deviates from -2.")