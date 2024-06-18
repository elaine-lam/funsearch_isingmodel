import numpy as np
import pickle


def evaluate():
    H_score = []
    for data in dataset:
        N, h, J = pull_data(data)
        '''if len(h) != len(J[0]): ## if h and J don't line up we just leave out that data
            print("Error - Data Matrix Dimensions are wonky")
            continue'''
        spins = assign_spins(N, h, J)
        H = evaluate_Hamiltonian(N, h, J, spins)
        H_score.append(H/N**3)
    return(np.mean(H_score)) # This should work

def priority(h, J):  # formula written by LLM
    N = len(h)
    ''' should take in h, J
        and return list of length N of lists of length 2
        first item is assigning probability of -1, second item is assigning probabi1ity if 1
    ''' 
    score = np.zeros((N**3,2))   
    return(score) 

def evaluate_Hamiltonian(N, h, J, spins):
    spin_left = np.roll(spins, -1, axis = 2)
    spin_down = np.roll(spins, -1, axis = 1)
    spin_backward = np.roll(spins, -1, axis = 0)
    interacting_spins = np.array((spin_left, spin_down, spin_backward))
    temp = np.einsum('ijk,ijkl -> ijkl', spins, J)
    H = np.einsum('ijk,ijk', h, spins) + np.einsum('ijkl,lijk', temp, interacting_spins)
    return(H)  

def assign_spins(N, h, J): 
    spins = np.ones(N**3)
    priorities = np.array(priority_h(h,J)) # TODO: Should call priority function from LLM
    if priorities.shape == (N**3,2):  # verify priority functions dimensions are correct. If not, just leave spins as 1
        for i in range(N**3):
            if priorities[i,0] >= priorities[i,1]:
                spins[i] =  -1
            else:
                spins[i] = 1
    else:
       raise IndexError("Priority has wrong shape")
    spins = spins.reshape((N,N,N))
    return(spins) # should return as an nd array

def pull_data(data):   # returns the matrices for the calculation out of the dictionary of data
    h = data["h"]
    J = data["J"]
    N = len(h)
    return(N, h, J)    

## sample priority functions for LLM to use
def priority_random(h,J):
    N = len(h)
    score = np.random.rand(N**3,2)
    return(score)

def priority_h(h,J):  # 3D - Decent-ish function, only uses h
    N = len(h)
    score_h = np.zeros((N**3,2))
    for i in range(N):
      for j in range(N):
        for k in range(N):
          if h[i,j,k] > 0:
            score_h[(i*N**2+N*j+k),0] = h[i,j,k]
          else:
            score_h[(i*N**2+N*j+k),1] = -1*h[i,j,k]
    return(score_h)

with open('data3D.txt', 'rb') as handle:
    dataset = pickle.loads(handle.read())
    
score = evaluate()
print(score)