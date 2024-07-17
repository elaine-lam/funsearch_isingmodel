import pickle
import evaluate
import numpy as np

def priority(N, h, J):
  priorities = np.zeros((N*N,2))
  for i in range(N**2):
    site_nbr = (i % N + ((i//N-1)%2 - 1)) % N
    total_spin = h[site_nbr][i%N] 
    if J[0,i%N,i//N] > 0:
      total_spin += sum([J[k,i%N,i//N]*h[(k+N-1)%N][i%N] for k in range(3)])
    else:
      total_spin -= sum([J[k,i%N,i//N]*h[(k+N-1)%N][i%N] for k in range(3)])

    if total_spin > 0:
      priorities[i][0] = total_spin
      priorities[i][1] = -priorities[i][0]
    else:
      priorities[i][0] = -total_spin
      priorities[i][1] = -priorities[i][0]

  return(priorities)


with open("data2D.txt", 'rb') as handle:  # import data
    test_data = pickle.loads(handle.read())

print(evaluate.evaluate(test_data,priority))

