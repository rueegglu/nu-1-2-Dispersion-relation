###-------------------------------------------------------------------------###
###----------------------------- CEL wavefunction -----------------------------###
###-------------------------------------------------------------------------###
"""This file contains the Psi_CEL() function,
 which takes in N positions of particles 
 on the sphere and returns the many
 body wavefunction for the nu=1/2 state. It has the option of having a vortex"""
###-------------------------------------------------------------------------###
###-------------------------- Import Modules -------------------------------###
###-------------------------------------------------------------------------###


import math
import numpy as np


def nCr(n, r):
    # Accept float if it's an integer value
    if not (float(n).is_integer() and float(r).is_integer()):
        raise TypeError("n and r must be integers or integer-valued floats.")
    n, r = int(n), int(r)
    if n < 0 or r < 0:
        return 0
    elif r > n:
        return 0  # By definition, nCr = 0 when r > n
    else:
        return math.comb(n, r)

def sec(x):
    return 1 / np.cos(x)

def tan(x):
    return np.tan(x)

def cos(x):
    return np.cos(x)

def sin(x):
    return np.sin(x)

def exp(x):
    return np.exp(x)

def factorial(x):
    # Accept float if it's an integer value
    if not float(x).is_integer():
        raise TypeError("factorial input must be an integer or integer-valued float.")
    x = int(x)
    if x < 0:
        raise ValueError("factorial input must be non-negative.")
    return math.factorial(x)

pi = np.pi



###-------------------------------------------------------------------------###


###-------------------------------------------------------------------------###
###----------------------------- Parameters --------------------------------###
###-------------------------------------------------------------------------###


#Effective charge
Q=1/2



#normalisation factor with Q=0
def Normalisation(q,n,m):
    return np.sqrt( (2*q+2*n+1) 
                   /(4*pi) 
                   * factorial(q+n-m) 
                   * factorial(q+n+m)
                   / factorial(n) 
                   / factorial(2*q+n)
                   )
###-------------------------------------------------------------------------###
###----------------------------- Functions ---------------------------------###
###-------------------------------------------------------------------------###

def Psi_CEL(positions,N, Vortex):
    #takes in positions, spits out wavefunction 
    #u,v coords of particles 
    #Vortex is True or False
    u_v_coords= np.zeros((N,2), dtype = complex) #e.g. u_v_coords[i]=(u_(i+1),v_(i+1)) 
    for j in range(N):
        theta_j=positions[j][0]
        phi_j=positions[j][1]
        u_v_coords[j][0]=cos(theta_j/2)*exp(-1j * phi_j/2) 
        u_v_coords[j][1]=sin(theta_j/2)*exp(+1j * phi_j/2)
    return Jastrow_factor(u_v_coords) * determinant_factor(u_v_coords) * vortex_factor(u_v_coords, Vortex)

def vortex_factor(u_v_coords, Vortex):
    N = len(u_v_coords)
    if Vortex==False:
        return 1
    else:
        mod_z_values = np.zeros(N) 
        for i in range(N):
            mod_z_values[i]=np.abs(u_v_coords[i][1]/u_v_coords[i][0])
        return np.prod(mod_z_values**2/(1+mod_z_values**2))**2
        
        

def Jastrow_factor(u_v_coords):
    N = len(u_v_coords)
    total=1
    for j in range(N):
        for i in range(j):
            u_i = u_v_coords[i][0]
            u_j = u_v_coords[j][0]
            v_i = u_v_coords[i][1]
            v_j = u_v_coords[j][1]
            e_to_i_phi_i = v_i**2 + u_i.conjugate()**2
            e_to_i_phi_j = v_j**2 + u_j.conjugate()**2
            
            total *= (u_i * v_j - u_j* v_i)**2 * e_to_i_phi_i * e_to_i_phi_j
    return total

def determinant_factor(u_v_coords):
    N = len(u_v_coords)
    matrix = np.zeros((N,N), dtype=complex)
    for i in range(N):
        for j in range(N):
            #print('calculating Y for i = ' + str(i) + ', j = ' + str(j))
            matrix[i,j]=tilde_Y(i,j,u_v_coords)
    #print('matrix = ' + str(matrix))
    return np.linalg.det(matrix)

def tilde_Y(i,j,u_v_coords):
    N = len(u_v_coords)
    #convert i into n and m
    nm_values = [(0,1/2),(0,-1/2),(1,3/2),(1,1/2), (1,-1/2), (1,-3/2), (2,5/2), (2,3/2), (2,1/2), (2,-1/2), (2,-3/2),  (2,-5/2)]
    n, m = nm_values[i]
    #print('n = ' + str(n) + ', m = ' + str(m))
    total =1
    ###
    ###normalisation
    ###
    q=N-1/2
    total *= (Normalisation(Q,n,m) 
        * (-1)**(Q+n-m) 
        * factorial(2*q+1) 
        / factorial(2*q+n+1))
    #print('total = ' + str(total))
    ###
    u_j = u_v_coords[j][0]
    v_j = u_v_coords[j][1]
    total *= u_j**(Q+m) * v_j**(Q-m)
    e_to_i_phi_j = v_j**2 + u_j.conjugate()**2 
    total *= (e_to_i_phi_j)**Q
    #print('total = ' + str(total))
    ###
    ###sum
    ###
    s_sum = 0
    for s in range(n+1):
        #s=0,1, ..., n
        s_sum += (  (-1)**s 
                * nCr(n, s) 
                * nCr(2*Q+n,Q+ n-m-s) 
                * v_j**(n-s) 
                * u_j**(s)
                * R(j,s,n-s, u_v_coords)  )
    
    total *= s_sum
    #print('total = ' + str(total))
    return total
    
def R(j,a,b, u_v_coords): #need to rewrite this bit to define the u,v outside the loop - cleaner
    N = len(u_v_coords)
    #6 different cases
    if a==0 and b==0:
        return 1
    if a==1 and b==0:
        k_sum=0
        for k in range(N):
            u_k = u_v_coords[k][0]
            u_j = u_v_coords[j][0]
            v_k = u_v_coords[k][1]
            v_j = u_v_coords[j][1]
            if k != j:
                k_sum += v_k / (u_j * v_k - v_j * u_k)
            else:
                k_sum += 0
        return k_sum
    if a==0 and b==1:
        k_sum=0
        for k in range(N):
            u_k = u_v_coords[k][0]
            u_j = u_v_coords[j][0]
            v_k = u_v_coords[k][1]
            v_j = u_v_coords[j][1]
            if k != j:
                k_sum += -u_k / (u_j * v_k - v_j * u_k)
            else:
                k_sum += 0
        return k_sum
    if a==2 and b==0:
        sum_1=0 #the sum in the first term
        for k in range(N):
            u_k = u_v_coords[k][0]
            u_j = u_v_coords[j][0]
            v_k = u_v_coords[k][1]
            v_j = u_v_coords[j][1]
            if k != j:
                sum_1 += v_k / (u_j * v_k - v_j * u_k)
            else:
                sum_1 += 0
        sum_2=0 # the sum in the 2nd term
        for k in range(N):
            u_k = u_v_coords[k][0]
            u_j = u_v_coords[j][0]
            v_k = u_v_coords[k][1]
            v_j = u_v_coords[j][1]
            if k != j:
                sum_2 += v_k**2 / (u_j * v_k - v_j * u_k)**2
            else:
                sum_2 += 0
        return sum_1**2 - sum_2
    if a==1 and b==1:
        sum_1=0 #the 1st sum in the first term
        for k in range(N):
            u_k = u_v_coords[k][0]
            u_j = u_v_coords[j][0]
            v_k = u_v_coords[k][1]
            v_j = u_v_coords[j][1]
            if k != j:
                sum_1 += v_k / (u_j * v_k - v_j * u_k)
            else:
                sum_1 += 0
        sum_2=0 #the 2nd sum in the first term
        for k in range(N):
            u_k = u_v_coords[k][0]
            u_j = u_v_coords[j][0]
            v_k = u_v_coords[k][1]
            v_j = u_v_coords[j][1]
            if k != j:
                sum_2 += u_k / (u_j * v_k - v_j * u_k)
            else:
                sum_2 += 0
        sum_3=0 # the sum in the 3rd term
        for k in range(N):
            u_k = u_v_coords[k][0]
            u_j = u_v_coords[j][0]
            v_k = u_v_coords[k][1]
            v_j = u_v_coords[j][1]
            if k != j:
                sum_3 += v_k * u_k / (u_j * v_k - v_j * u_k)**2
            else:
                sum_3 += 0
        return -sum_1*sum_2+ sum_3
    if a==0 and b==2:
        sum_1=0 #the sum in the first term
        for k in range(N):
            u_k = u_v_coords[k][0]
            u_j = u_v_coords[j][0]
            v_k = u_v_coords[k][1]
            v_j = u_v_coords[j][1]
            if k != j:
                sum_1 += u_k / (u_j * v_k - v_j * u_k)
            else:
                sum_1 += 0
        sum_2=0 # the sum in the 2nd term
        for k in range(N):
            u_k = u_v_coords[k][0]
            u_j = u_v_coords[j][0]
            v_k = u_v_coords[k][1]
            v_j = u_v_coords[j][1]
            if k != j:
                sum_2 += u_k**2 / (u_j * v_k - v_j * u_k)**2
            else:
                sum_2 += 0
        return sum_1**2 - sum_2


#N=3
#positions =np.random.random((N,2)) #e.g. Positions[i]=(theta_(i+1),phi_(i+1)) 
#print(Psi(positions, N))