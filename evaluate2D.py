import numpy as np
import pickle

def evaluate(dataset, func):
    H_score = []
    for data in dataset:
        N, h, J = pull_data(data)
        spins = assign_spins(N, h, J)
        H = evaluate_Hamiltonian(N, h, J, spins)
        H_score.append(H/N**2)
    return(np.mean(H_score))

def evaluate_Hamiltonian(N, h, J, spins):
    spin_left = np.roll(spins, -1, axis = 1)
    spin_down = np.roll(spins, -1, axis = 0)
    interacting_spins = np.array((spin_left, spin_down))
    temp = np.einsum('ij,ijk -> ijk', spins, J)
    H = np.einsum('ij,ij', h, spins) + np.einsum('ijk,kij', temp, interacting_spins)
    return(H)

def assign_spins(N, h, J, func): 
    spins = np.ones(N**2)
    priorities = np.array(func(h,J))
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

## sample priority functions for testing or for LLM promptin
'''
def priority(h, J):  # formula written by LLM
    N = len(h)
     should take in h, J
        and return list of length N of lists of length 2
        first item is assigning probability of -1, second item is assigning probabi1ity if 1
    
    score = np.zeros((N**2,2))   
    return(score)
def priority_random(h,J):
    N = len(h)
    score = np.random.rand(N**2,2)
    return(score)

def priority_h(h,J):  # 2D
    N = len(h)
    score_h = np.zeros((N**2,2))
    for i in range(N):
      for j in range(N):
        if h[i,j] > 0:
            score_h[(i*N+j),0] = h[i,j]
        else:
            score_h[(i*N+j),1] = -1*h[i,j]
    return(score_h)

with open('data2D.txt', 'rb') as handle:
    dataset = pickle.loads(handle.read())

score = evaluate()
print(score)
'''
