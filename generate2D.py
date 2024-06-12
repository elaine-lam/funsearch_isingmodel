import numpy as np
import pickle

def generate_J(N: int, prob_neg: float, prob_pos: float):
    if prob_pos == prob_neg == 0: ## use this for equal probability of 1, 0, -1 (uses more efficient sampling method)
        J = np.random.choice([-1,0,1], size = (N,N,2))
    else:
        J = np.random.choice([-1,0,1], size = (N,N,2), p = (prob_neg, 1-prob_pos-prob_neg, prob_pos))
    return(J)

def generate_h(N, min, max):
    h = np.random.randint(min, max, size = (N,N))
    return(h)


num_datasets = 100 
data = []
filetest = open("data2D.txt", "wb")
for i in range(num_datasets):
    N = np.random.randint(10,20)  ## TODO: Figure out good values for all of these
    U = 1 #np.random.uniform(-3,3)
    h = generate_h(N, 0, 4)
    J = U * generate_J(N, 0.35,0.35)
    dictionary = {"h": h, "J" : J}
    data.append(dictionary)
pickle.dump(data, filetest)
 