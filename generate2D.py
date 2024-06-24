import numpy as np
import pickle

def generate_J(N: int, prob_neg: float, prob_pos: float):
    if prob_pos == prob_neg == 0: ## use this for equal probability of 1, 0, -1 (uses more efficient sampling method)
        J = np.random.choice([-1,0,1], size = (2,N,N))
    else:
        J = np.random.choice([-1,0,1], size = (2,N,N), p = (prob_neg, 1-prob_pos-prob_neg, prob_pos))
    return(J)

def generate_h(N, min, max):
    h = np.random.randint(min, max, size = (N,N))
    return(h)


num_datasets = 1000 
data = []
file = open("data2D.txt", "wb")
N = 36   #  Any parameter here could be changed or scaled, but this is a good starting place
U = 1
mu = 0.2
for i in range(num_datasets):
    h = mu * generate_h(N, -3, 4)
    J = U * generate_J(N, 0, 0) 
    dictionary = {"h": h, "J" : J}
    data.append(dictionary)
pickle.dump(data, file) 