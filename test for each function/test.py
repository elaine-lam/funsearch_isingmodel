import numpy as np
from scipy.optimize import minimize

def ising_energy(config, J, h):
    N = len(config)
    E = 0.0
    for i in range(N):
        for j in range(i+1, N):
            E += -J * config[i] * config[j]
    E += np.sum(h * config)
    return E

def ising_model(L, J, h, T):
    N = L ** 2
    s = np.random.choice([-1, 1], size=(N,))
    while True:
        energy = ising_energy(s, J, h / (L**3))
        s_new = s.copy()
        for i in range(N):
            if np.random.rand() < np.exp(-deltaE(i, s, s_new)):
                s_new[i] *= -1
        if np.abs(energy - ising_energy(s_new, J, h / (L**3))) < 1e-6:
            return s

def deltaE(i, s, s_new):
    E = 2 * s[i] * s[i] + 4 * s[i] * sum([s[j] for j in neighbors(i, L)])
    E_new = 2 * s_new[i] * s_new[i] + 4 * s_new[i] * sum([s_new[j] for j in neighbors(i, L)])
    return (E_new - E) / T

def neighbors(i, L):
    N = L ** 2
    x = i // L
    y = i % L
    if x > 0:
        yield i - L
    if y < L-1:
        yield i + 1
    if x < L-1:
        yield i + L
    if y > 0:
        yield i - 1

L, J, h = 10, 1.0, 2.5
T = 2.0

s = ising_model(L, J, h, T)
print(f"Magnetization: {np.mean(s):.4f}")