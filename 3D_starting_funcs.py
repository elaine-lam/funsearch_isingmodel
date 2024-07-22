# score -0.4916790000000001
def priority(N, h, J):
  priorities = h
  interacting_spins = np.zeros((6,N,N,N))  # D X N^D matrix of neighboring spins along each axis
  for i in range(3):
    interacting_spins[i] = np.roll(h, -1, axis = i)
  for i in range(3):
    interacting_spins[i+3] = np.roll(h, 1, axis = i)
  for i in range(N):
    for j in range(N):
      for k in range(N):
        for l in range(6):
            priorities[i,j,k] += -J[l,i,j,k]*interacting_spins[l,i,j,k]
  priorities = np.array([priorities.flatten(), np.zeros(N**3)]).T
  return(priorities)


# score -0.34222139999999956
def priority(N, h, J):
  h_flat = h.flatten()
  priorities = np.array([h_flat, -h_flat]).T
  return(priorities)