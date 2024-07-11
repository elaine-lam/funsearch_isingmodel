# {score:-0.53650787037037}
def priority(N, h, J):
    priorities = h
    interacting_spins = np.zeros((4,N,N))  # D X N^D matrix of neighboring spins along each axis
    for i in range(2):
        interacting_spins[i] = np.roll(h, -1, axis = i)
    for i in range(2):
        interacting_spins[i+2] = np.roll(h, 1, axis = i)
    for i in range(N):
        for j in range(N):
            for k in range(4):
                priorities[i,j] += -0.5*J[k,i,j]*interacting_spins[k,i,j]
    priorities = np.array([priorities.flatten(), np.zeros(N**2)]).T
    return(priorities)