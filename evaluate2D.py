import numpy as np
import pickle

def evaluate(dataset, func):
    H_score = []
    for data in dataset:
        N, h, J = pull_data(data)
        spins = assign_spins(N, h, J, func)
        H = evaluate_Hamiltonian(N, h, J, spins)
        H_score.append(H/N**2)
    return(np.mean(H_score))

def evaluate_Hamiltonian(N, h, J, spins):
    spin_left = np.roll(spins, -1, axis = 1)
    spin_down = np.roll(spins, -1, axis = 0)
    interacting_spins = np.array((spin_left, spin_down))
    temp = np.einsum('ij,kij -> ijk', spins, J)
    H = np.einsum('ij,ij', h, spins) + np.einsum('ijk,kij', temp, interacting_spins)
    return(H)

def assign_spins(N, h, J, func): 
    spins = np.ones(N**2)
    J_aug = np.empty((4,N,N))
    for i in range(2):
       J_aug[i] = J[i]
       J_aug[i+2] = np.roll(J[i], 1, axis = 1-i)
    priorities = np.array(func(h,J_aug))
    if priorities.shape == (N**2,2):  # verify priority functions dimensions are correct. If not, just leave spins as 1
        for i in range(N**2):
            if priorities[i,0] >= priorities[i,1]:
                spins[i] =  -1
            else:
                spins[i] = 1
    else:
        print(priorities.shape)
        raise IndexError("Priority matrix must be N^2 by 2")
    spins = spins.reshape((N,N))
    return(spins) # should return as an N x N array

def pull_data(data):
    h = data["h"]
    J = data["J"]
    N = len(h)
    return(N, h, J)
