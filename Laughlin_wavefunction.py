###-------------------------------------------------------------------------###
###----------------------------- CEL wavefunction -----------------------------###
###-------------------------------------------------------------------------###
"""This file contains the Psi_Laughlin() function,
 which takes in N positions of particles 
 on the sphere and returns the many
 body wavefunction for the nu=1/3 state. It has the option of having a vortex"""
###-------------------------------------------------------------------------###
###-------------------------- Import Modules -------------------------------###
###-------------------------------------------------------------------------###

import numpy as np


def tan(x):
    return np.tan(x)
def exp(x):
    return np.exp(x)


def Psi_Laughlin(positions,N, Vortex):
    #takes in positions, spits out wavefunction 
    #u,v coords of particles 
    #Vortex is True or False
    z_coords = np.zeros(N, dtype = complex)
    for i in range(N):
        theta_i = positions[i][0]
        phi_i = positions[i][1]
        z_coords[i] = exp(1j * phi_i) * tan(theta_i/2)
    return (Jacobian_factor(z_coords,N) * normalisation(z_coords,N))**3 * Vortex_Factor(z_coords,N, Vortex)

def Jacobian_factor(z_coords,N):
    total =1
    for i in range(N):
        for j in range(i):
            total *= (z_coords[i]-z_coords[j])
    return total
def normalisation(z_coords, N):
    Q_1= N/2 - 1/2
    total =1
    for i in range(N):
        total *= (1+ np.abs(z_coords[i])**2)**(-Q_1)
    return total

def Vortex_Factor(z_coords,N, Vortex):
    total =1
    if Vortex == False:
        return 1
    else:
        for m in range(N):
            total *= np.abs(z_coords[m])**2 / (1+np.abs(z_coords[m])**2)
        return total
