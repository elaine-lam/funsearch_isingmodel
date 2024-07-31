'''
This file pull out the execution code we made for executing the algorithm code.
'''

import config as config_lib
from funsearch import main

if __name__ == '__main__':
  specification = '''import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import itertools
from evaluate import evaluate
import funsearch

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
'''
  inputstr = "data2D.txt" #name of the data set to test for given score
  inputs = inputstr.split(',')
  config = config_lib.Config()
  main(specification, inputs, config)