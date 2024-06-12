import numpy as np
import pickle


def evaluate():
    H_score = []
    for data in dataset:
        h, J = pull_data(data)
        if len(h) != len(J[0]): ## if h and J don't line up we just leave out that data
            print("Error - Data Matrix Dimensions are wonky")
            continue
        spins = assign_spins(h, J)
        H = evaluate_Hamiltonian(h, J, spins)
        H_score.append(H)
        if H == 0:
            print(h, spins)
    return(np.mean(H_score)) # This should work

def priority(h, J):  # formula written by LLM
    ''' should take in h, J
        and return list of length N of lists of length 2
        first item is assigning probability of -1, second item is assigning probabi1ity if 1
    ''' 
    score = np.zeros(len(h),2)   
    return(score) 

def evaluate_Hamiltonian(h, J, spins):
    N = len(spins)
    temp = np.matmul(J, spins)  # order does not appear to matter for this calculation
    H = spins.dot(h)+ spins.dot(temp)  # Hamiltonian
    return(H)  

def assign_spins(h, J): 
    N = len(h)
    spins = np.ones(N)
    priorities = np.array(priority(h,J))
    if priorities.shape == (N,2):  # verify priority functions dimensions are correct. If not, just leave spins as 1
        while np.any(priorities != -np.inf):  # assign spins via a greedy algorithm
            i, j = argmax_dim(priorities)
            if j == 0:
                spins[i] = -1
            else:
                spins[i] = 1
            priorities[i] = [-np.inf, -np.inf]
    return(spins) # should return as an nd array

def pull_data(data):   # returns the matrices for the calculation out of the dictionary of data
    h = data["h"]
    J = data["J"]
    return(h, J)    

def argmax_dim(priority_mat):  # np.argmax unflattened - returns both indexes of matrix
    n = len(priority_mat[0])
    argmax = np.argmax(priority_mat)
    return(argmax//n, argmax%n)

def priority_random(h,J):
    N = len(h)
    score = np.random.rand(N,2)
    return(score)


with open('data2D.txt', 'rb') as handle:
    dataset = pickle.loads(handle.read())
    score = evaluate()
    print(score)