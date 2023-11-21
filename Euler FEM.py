import numpy as np
from scipy.linalg import eigh
from matplotlib import pyplot as plt
from prettytable import PrettyTable
import time


num_elems = 50
tleng = 7
l = tleng / num_elems
E = 210e9
I = 2003e-8
rho = 61.3
A = 1.5
Cm = rho * A   # rho.A
Ck = E * I  # E.I
# Uncomment desired Boundary Conditions
#bc = 'c-f'            # clamped-free
bc = 'c-c'           # clamped-clamped
#bc = 'c-s'           # clamped-supported
#bc = 's-s'           # supported-supported

def beam(num_elems, l, Cm, Ck, bc):
    
        # element mass and stiffness matrices
    m = np.array([[156, 22 * l, 54, -13 * l],
                   [22 * l, 4 * l * l, 13 * l, -3 * l * l],
                   [54, 13 * l, 156, -22 * l],
                   [-13 * l, -3 * l * l, -22 * l, 4 * l * l]]) * Cm * l / 630

    k = np.array([[12, 6 * l, -12, 6 * l],
                   [6 * l, 4 * l * l, -6 * l, 2 * l * l],
                   [-12, -6 * l, 12, -6 * l],
                   [6 * l, 2 * l * l, -6 * l, 4 * l * l]]) * Ck / l ** 3

        # construct global mass and stiffness matrices
    M = np.zeros((2 * num_elems + 2 , 2 * num_elems + 2))
    K = np.zeros((2 * num_elems + 2, 2 * num_elems + 2))

        # for each element, change to global coordinates
    for i in range(num_elems):
        M_temp = np.zeros((2*num_elems+2,2*num_elems+2))
        K_temp = np.zeros((2*num_elems+2,2*num_elems+2))
        M_temp[2*i:2*i+4, 2*i:2*i+4] = m
        K_temp[2*i:2*i+4, 2*i:2*i+4] = k
        M += M_temp
        K += K_temp
        restrained_dofs = BoundaryConditions(bc)
        # remove the fixed degrees of freedom
    for dof in restrained_dofs:
        for i in [0,1]:
            M = np.delete(M, dof, axis=i)
            K = np.delete(K, dof, axis=i)

    eval, evec = eigh(K,M)
    frequencies = np.sqrt(eval)
    return M, K, frequencies, evec , eval

def BoundaryConditions(bc):
    if bc == 'c-c':    # clamped-clamped beam
        restrained_dofs = [1, 0, -2, -1] 
        
    elif bc == 'c-f': # clamped-free beam
       restrained_dofs = [1, 0] 
               
    elif bc == 'c-s':   # clamped-supported beam
        restrained_dofs = [1, 0, -1] 
                  
    elif bc == 's-s':     # supported-supported beam
        restrained_dofs = [0, -2] 
         
    return restrained_dofs

def theory(bc,E1,I1,m1,L1):
    E = E1 
    I = I1 
    m = m1 
    L = L1
    
    if bc == 'c-c':
        kn = np.array([22.4, 61.7, 121, 200, 299])
   
    elif bc == 'c-s':
        kn = np.array([15.4, 50.0, 104, 178, 272])
        
    elif bc == 'c-f':
        kn = np.array([3.52, 22.0, 61.7, 121, 200]) 
       
    elif bc == 's-s':
        kn = np.array([9.87, 39.5, 88.8, 158, 247])
 
    thfreq = kn/L**2*np.sqrt((E*I)/m)
    thfreqHz = (kn/(2*np.pi*L**2))*np.sqrt((E*I)/m) 
    
    return thfreq, thfreqHz
# beam element
print('Beam element')

errors = []
start = time.time()
M, K, frequencies, evec, eval = beam(num_elems, l, Cm, Ck, bc)
thfreq, thfreqHz = theory(bc,E,I,rho,tleng)
time_taken = time.time() - start
error = (frequencies[0:3] - thfreq[0:3]) / thfreq[0:3] * 100.0

#Plot shape modes
v = evec[::2,:]
for i in range(v.shape[1]):
    # Standardize each column
    v[:, i] = v[:, i] / np.max(np.abs(v[:, i]))

# Store the standardized array in 'V'
V = v

#matplotlib
for i in range(4):
    plt.subplot(2, 2, i+1 )
    plt.tight_layout()
    plt.plot(v[:,i])
    plt.xlabel('Normalized Response')
    plt.ylabel('X cordination')
    plt.title('Mode shapes')


t = PrettyTable()
t.add_column("FEM Frequency",np.round(frequencies[0:3], 3))
t.add_column("Theorical Frequency",np.round(thfreq[0:3], 3))
t.add_column("Error",np.round(error, 3))

print(t)
print('Num Elems: {} \tShape: {} \tTime: {}'.format(num_elems, K.shape, round(time_taken*1000, 3) ))




