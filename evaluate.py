import numpy as np
import pickle


def evaluate(dataset: dict, func):
    '''Scores LLM written `priority()` function on a given dataset of magnetism and interaction arrays \n
    
    Inputs: \n
    \t`dataset` -- list of dictionaries of data to use\n
    \t`func` -- priority function

    Output:\n
    \t`score` -- average energy per spin site over dataset
    '''
    H_score = []
    for data in dataset:
        N, D, h, J = pull_data(data)  # N - grid size, D - dimensions
        spins = assign_spins(N, D, h, J, func)
        H = evaluate_Hamiltonian(N, D, h, J, spins)
        H_score.append(H/N**D)  # average energy per site
    return(np.mean(H_score))


def evaluate_Hamiltonian(N: int, D: int, h, J, spins) -> float:
    """ Calculates energy of a set of spins for given interaction and magnetism matrices \n

    Inputs:
    \tN -- grid size\n
    \tD -- dimensions\n
    \th -- magnetism matrix (N^D ndarray)\n
    \tJ -- interaction matrix (2D x N^D ndarray)\n
    \tspins -- matrix of -1 and 1 assigning spins at each state (N^D ndarray) \n

    Outputs:
    \tH -- calculated energy 
    """
    """" How this code calculates energy:
    h is N^D tensor, and h[i,j,...,k]: magnetism at site i,j,...,k
    J is a D x N^D tensor. J[i] is an N^D matrices giving relations between the corresponding site and the site next to it along the ith axis.
        Ex:
        - J[0,j,k,...] is the interaction between sites j,k,... and j+1,k,... where addition is modulo N (periodic boundaries)
        - J[D-1,j,...,k] is interaction between sites j,...,k and j,...,k+1 where addition is modulo N (periodic boundaries)
    H_mag: sum over all sites i of h_i*spin_i
        - Einstien summation of two N^D arrays. Corresponding sites are multiplied together, sum of all of these is taken
    H_interact: sum over all neighbor interactions J_{ij}*s_i*s_j
        - interacting spins is a D x N^D tensor created by rolling the spins tensor along different axis
            - interacting_spins[0,j,...,k] = spins[j+1, ...,k]; interacting spins[D-1,j,...,k] = spins[j,...k,+1]; addition modulo N (periodic boundaries)
        - temp: D x N^D ndarray that gives an intermediate step: [J{ij}*s_i]
        - Einstien summation of temp and interacting_spins. Corresponding sites are multiplied together, sum over all of these is taken
    """
    H_mag = np.einsum(h,np.arange(D), spins, np.arange(D), [])  # energy from magnetism
    interacting_spins = np.empty(tuple([N if i!= 0 else D for i in range(D+1)]))  # D X N^D matrix of neighboring spins along each axis
    for i in range(D):
        interacting_spins[i] = np.roll(spins, -1, axis = i)
    temp = np.einsum('...,l... -> l...', spins, J)  # D x N^D matrix
    H_interact = np.einsum(temp, np.arange(D+1), interacting_spins, np.arange(D+1), [])
    H = H_mag + H_interact
    return(H)  

def assign_spins(N: int, D:int, h, J, func):
    """Uses LLM `priority()` function to create a N^D tensor of spin states which are either -1 or 1\n
    
    Inputs:
    \tN -- grid size\n
    \tD -- dimension\n
    \th -- magnetization matrix (N^D ndarray)\n
    \tJ -- interaction matrix (D x N^D ndarray)\n
    \t`func` -- function named `priority` that takes N, D, h, and a 2D x N^D ndarray J_aug\n

    Outputs:
    \tspins -- spin states (N^D ndarray containing only the values -1 and 1)
    """ 
    spins = np.ones(N**D)
    J_aug = np.empty(tuple([N if i!= 0 else 2*D for i in range(D+1)])) # input to LLM func is augmented J matrix, has all interaction values for corresponding site
    for i in range(D):
       J_aug[i] = J[i]
       J_aug[i+D] = np.roll(J[i], 1, axis = D-1-i)
    priorities = np.array(func(N, D, h, J_aug)) # calls LLM priority function
    if priorities.shape == (N**D,2):  # verify priority functions dimensions are correct
        for i in range(N**D):
            if priorities[i,0] >= priorities[i,1]:
                spins[i] =  -1
            else:
                spins[i] = 1
    else:
       raise IndexError("Priority matrix must be N^"+str(D)+" by 2")
    spins = spins.reshape(tuple(N for i in range(D))) 
    return(spins) # should return as an nd array

def pull_data(data: dict):
    """ Returns matrices and values for calculations out of dictionary of data\n
        
        Outputs:\n
        \tN -- grid size \n
        \tD -- grid dimesion \n
        \t`h` -- magnetization matrix (N^D ndarray)\n
        \t`J` -- interaction matrix  (2DxN^D ndarray)\n
    """
    h = data["h"]
    J = data["J"]
    N = len(h)
    D = len(h.shape)
    return(N, D, h, J)    

'''## sample priority functions for LLM to use

def priority(N, D, h, J):  # formula written by LLM
    score = np.zeros((N**D,2))   
    return(score) 

    def priority_random(N, D, h,J):
    N = len(h)
    score = np.random.rand(N**D,2)
    return(score)

with open('data2D.txt', 'rb') as handle:
    dataset = pickle.loads(handle.read())
    
score = evaluate(dataset, priority)
print(score)'''