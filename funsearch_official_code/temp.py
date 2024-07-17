import pickle
import evaluate
import numpy as np

def priority(N, h, J):
  total_spin = np.zeros((N*N, 2))
  for i in range(N):
    for j in range(N):
      site_nbr = (i + ((j-1)%2 - 1)) % N
      total_spin[i*N+j][0] += h[site_nbr][j]
      if h[i][j] > 0:
        total_spin[i*N+j][1] -= 1
      else:
        total_spin[i*N+j][1] += 1

  priority_total = np.zeros((N*N,2))
  for k in range(4):
    for i in range(N):
      site_nbr = (i + ((k-1)%2 - 1)) % N
      for j in range(N):
        if h[site_nbr][j] > 0:
          total_spin[i*N+j][0] += 1
          total_spin[i*N+j][1] -= 1
        else:
          total_spin[i*N+j][0] -= 1
          total_spin[i*N+j][1] += 1

  for i in range(N):
    for j in range(N):
      priority_total[i*N+j][0] = -total_spin[i*N+j][0]
      priority_total[i*N+j][1] = 1 - np.abs(total_spin[i*N+j][1])

  return priority_total.flatten().reshape(-1,2)

with open("data2D.txt", 'rb') as handle:  # import data
    test_data = pickle.loads(handle.read())

print(evaluate.evaluate(test_data,priority))

