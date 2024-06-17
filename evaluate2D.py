import numpy as np
import pickle


def evaluate():
    H_score = []
    for data in dataset:
        h, J = pull_data(data)
        '''if len(h) != len(J[0]): ## if h and J don't line up we just leave out that data
            print("Error - Data Matrix Dimensions are wonky")
            continue'''
        spins = assign_spins(h, J)
        H = evaluate_Hamiltonian(h, J, spins)
        H_score.append(H)
    return(np.mean(H_score)) # This should work

def priority(h, J):  # formula written by LLM
    N = len(h)
    ''' should take in h, J
        and return list of length N of lists of length 2
        first item is assigning probability of -1, second item is assigning probabi1ity if 1
    ''' 
    score = np.zeros((N**2,2))   
    return(score) 

def evaluate_Hamiltonian(h, J, spins):
    N = len(spins)
    spin_left = np.roll(spins, -1, axis = 1)
    spin_down = np.roll(spins, -1, axis = 0)
    interacting_spins = np.array((spin_left, spin_down))
    temp = np.einsum('ij,ijk -> ijk', spins, J)
    H = np.einsum('ij,ij', h, spins) + np.einsum('ijk,kij', temp, interacting_spins)
    return(H)

def assign_spins(h, J): 
    N = len(h)
    spins = np.ones(N**2)
    priorities = np.array(priority(h,J)) ## TODO: Change this line to call LLM function
    if priorities.shape == (N**2,2):  # verify priority functions dimensions are correct. If not, just leave spins as 1
        while np.any(priorities != -np.inf):  # assign spins via a greedy algorithm
            i, j = np.unravel_index(np.argmax(priorities), shape = priorities.shape)
            if j == 0:
                spins[i] = -1
            else:
                spins[i] = 1
            priorities[i] = [-np.inf, -np.inf]
    else:
        raise IndexError("Priority matrix has wrong size")
    spins = spins.reshape((N,N))
    return(spins) # should return as an N x N array

def pull_data(data):   # returns the matrices for the calculation out of the dictionary of data
    h = data["h"]
    J = data["J"]
    return(h, J)    

## sample priority functions for testing or for LLM prompting
def priority_random(h,J):
    N = len(h)
    score = np.random.rand(N,2)
    return(score)

def priority_h(h,J):  # 2D
    N = len(h)
    score_h = np.zeros((N**3,2))
    for i in range(N):
      for j in range(N):
        if h[i,j] > 0:
            score_h[(i*N+j),0] = h[i,j]
        else:
            score_h[(i*N+j),1] = -1*h[i,j]
    return(score_h)

with open('funsearch_isingmodel/data2D.txt', 'rb') as handle:
    dataset = pickle.loads(handle.read())

score = evaluate()
print(score)

