import numpy as np
import pickle


def evaluate(dataset: dict, func):
    '''Scores LLM written `priority()` function on a given dataset of magnetism and interaction arrays \n
    
    Inputs: \n
    \t`dataset` -- list of dictionaries of data to use\n
    \t`func` -- priority function

    Output:\n
    `score` -- average energy per spin site over dataset
    '''
    H_score = []
    for data in dataset:
        N, D, h, J = pull_data(data)  # N - grid size, D - # dimensions
        spins = assign_spins(N, D, h, J, func)
        H = evaluate_Hamiltonian(N, D, h, J, spins)
        H_score.append(H/N**D)
    return(np.mean(H_score))

def priority(N, D, h, J):  # formula written by LLM
    score = np.zeros((N**D,2))   
    return(score) 

def evaluate_Hamiltonian(N: int, D: int, h, J, spins) -> float:
    interacting_spins = np.empty(tuple([N if i!= 0 else D for i in range(D+1)]))  # Not the prettiest way to do this but it works
    for i in range(D):
        interacting_spins[i] = np.roll(spins, -1, axis = D-1-i)
    temp = np.einsum('...,l... -> l...', spins, J)  # New formulation is identical to old formulation - works IF old one also works
    H = np.einsum(h,np.arange(D), spins, np.arange(D), []) + np.einsum(temp, np.arange(D+1), interacting_spins, np.arange(D+1), [])
    return(H)  

def assign_spins(N: int, D:int, h, J, func): 
    spins = np.ones(N**D)
    J_aug = np.empty(tuple([N if i!= 0 else 2*D for i in range(D+1)]))
    for i in range(D):
       J_aug[i] = J[i]
       J_aug[i+D] = np.roll(J[i], 1, axis = D-1-i)
    priorities = np.array(func(N, D, h, J_aug)) # Should call priority function from LLM
    if priorities.shape == (N**D,2):  # verify priority functions dimensions are correct. If not, just leave spins as 1
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
def priority_random(N, D, h,J):
    N = len(h)
    score = np.random.rand(N**D,2)
    return(score)

with open('data2D.txt', 'rb') as handle:
    dataset = pickle.loads(handle.read())
    
score = evaluate(dataset, priority)
print(score)'''