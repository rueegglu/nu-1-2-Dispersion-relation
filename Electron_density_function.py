###-------------------------------------------------------------------------###
###------------- Electron density function for Laughlin---------------------###
###-------------------------------------------------------------------------###
"""This file contains a function that 
calculates the electorn density for the Laughlin state
at a particular r,
with and without a vortex."""
###-------------------------------------------------------------------------###
###-------------------------- Import Modules -------------------------------###
###-------------------------------------------------------------------------###

import numpy as np
from Laughlin_wavefunction import Psi_Laughlin
from CEL_wavefunction import Psi_CEL

pi =np.pi

def sin(x):
    return np.sin(x)

def electron_density_values(r_values, no_samples,N, Vortex, state):
    #Outputs an array of electron density values
    # whicch can then be plotted
    #This is gonna be vectorised to speed things up
    theta_values = 2* np.arctan(r_values)
    V = (2*pi**2)**(N-1)
    rho_values = np.zeros(len(r_values))
    for i in range(len(theta_values)):
        m_sum =0
        theta=theta_values[i]
        for m in range(no_samples):
            X_m = np.zeros((N-1,2)) 
            # X_m[i][0]= theta_{i+2}
            # X_m[i][1] = phi_{i+2}
            for j in range(N-1):
                X_m[j][0]=np.random.uniform(0,pi) #sampling uniformly - sin(theta) in Jacobian ensures uniform sampling over the sphere.
                X_m[j][1]=np.random.uniform(0, 2*pi)
            #print(X_m)
            m_sum += f(theta, X_m,N, Vortex, state)
        rho_values[i] = V / no_samples *m_sum
    #Normalise so that rho_values rises from -1 at r=0 to 0 at r=infty.
    for i in range(len(rho_values)):
        rho_values[i] = (rho_values[i]-rho_values[-1])/rho_values[-1]
    return rho_values

def f(theta, X_m,N, Vortex, state):
    total=1 
    for i in range(N-1):
        theta_i_m = X_m[i][0]
        total *= sin(theta_i_m)
    Omega = np.array([theta, 0])
    positions = np.vstack([Omega, X_m])  # shape (N, 2)
    if state == "Laughlin":
        total *= np.abs(Psi_Laughlin(positions, N, Vortex))**2
    elif state == "CEL":
        total *= np.abs(Psi_CEL(positions, N, Vortex))**2
    return total 



###-------------------------------------------------------------------------###
###--------------------- Tests to check -------------------------------###
###------------------------------------------------------------------------###
"""
import matplotlib.pyplot as plt
import time
pi = np.pi


def plot_electron_density(N, no_samples, Vortex):
    r_values = np.linspace(0.01,7, 100) #avoid singular endpoints
    rho_values = electron_density_values(r_values, no_samples, N, Vortex)
    plt.figure(figsize=(6, 4))
    plt.plot(r_values, rho_values, marker='o', linestyle='-', color='b', label=r'$\rho(\theta)$')
    plt.xlabel(r'$r$ ')
    plt.ylabel(r'$\rho(r)$')
    plt.title(f'Electron Density for Laughlin state, N={N}, no_samples={no_samples}, Vortex = {Vortex}')
    plt.grid(True)
    plt.legend()
    plt.ylim((-1.2, 1.2*max(rho_values)))
    plt.show()

plot_electron_density(1, 1000, True)
"""
    