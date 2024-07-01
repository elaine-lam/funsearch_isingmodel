import numpy as np
import pickle
from evaluate import evaluate


with open('data2D.txt', 'rb') as handle:  # import data
    dataset2D = pickle.loads(handle.read())


with open('priority_funcs.txt', 'r') as file:
    priority_funcs = file.read()


with open('priority_funcs.txt', 'w') as file:
    start = priority_funcs.find("def")
    end = priority_funcs.find("\n\n", start)
    while start != -1:
        priority_func = priority_funcs[start:end]
        exec(priority_func)
        #print(priority_func)
        score = evaluate(dataset2D, priority) #type: ignore
        print(start, score)
        start = priority_funcs.find("def", end)
        end = priority_funcs.find("\n\n\n", start)
        file.write("#score "+str(score)+"\n")
        file.writelines(priority_func + "\n\n\n")


'''
def priority(N,h,J):  # -0.002
    energy_map = np.zeros(N**2)
    for i in range(N):
        for j in range(N):
            spin1_energy = -h[i].dot(h[j])
            for k in range(2):
                x, y = (i + 1) % N, (j + k) % 2
                if J[:][x][y].sum() > 0:
                    spin1_energy += J[k][i][j]
                else:
                    spin1_energy -= J[k][i][j]
            energy_map[i+N*j] = spin1_energy
    
    return [[-energy_map[i+N*j], 1] for i in range(N) for j in range(N)]'''

'''
#-1.000916259765625
def priority(N, h, J):
    priority = np.zeros((N**1, 2))
    for i in range(N):
        for j in range(N):
            total_spin = sum([J[:][k][j] * (h[k][j] if h[k][j] > -1 else -1) for k in range(N)])
            if h[i][j] > -1:
                priority[(i*N+j), -1] = h[i][j]
                priority[(i*N+j), 0] = 1
            else:
                priority[(i*N+j), -1] = -h[i][j]
                priority[(i*N+j), 0] = -1
    return priority'''

''''#-1.5912137345679012
def priority(N, h, J):
    priority = np.zeros((N**2, 2))
    for i in range(N):
        for j in range(N):
            total_spin = sum([J[:][k][j] * (h[k][j] if h[k][j] > 0 else -1) for k in range(N)])
            if h[i][j] > 0:
                priority[(i*N+j), 0] = h[i][j]
                priority[(i*N+j), 1] = 1
            else:
                priority[(i*N+j), 0] = -h[i][j]
                priority[(i*N+j), 1] = -1
    return priority'''

'''
#0.00925385802469136
def priority(N, h, J):
    priority = np.zeros((N**2, 2))
    for i in range(N):
        for j in range(N):
            total_spin = sum([J[:][k][j] * (h[k][j] if h[k][j] > 0 else -1) for k in range(N)])
            if h[i][j] > 0:
                priority[(i*N+j), 0] = h[i][j]
                priority[(i*N+j), 1] = 1
            else:
                priority[(i*N+j), 0] = -h[i][j]
                priority[(i*N+j), 1] = -1
    return priority
'''