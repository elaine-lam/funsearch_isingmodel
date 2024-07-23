import pickle
import evaluate
import numpy as np

def priority(vector, n):
  """
  Assigns a priority to a vector based on its sum of elements and the number of trailing zeros.

  Args:
  vector (tuple): A vector in the cap set
  n (int): The size of the cap set

  Returns:
  float: The priority of the vector
  """
  # Calculate the sum of elements in the vector
  sum_elements = sum(vector)
  
  # Calculate the number of trailing zeros in the vector
  trailing_zeros = 0
  for elem in reversed(vector):
    if elem == 0:
      trailing_zeros += 1
    else:
      break
  
  # Assign a higher priority to vectors with lower sum of elements and more trailing zeros
  return -sum_elements - trailing_zeros / n


# with open("data2D.txt", 'rb') as handle:  # import data
#     test_data = pickle.loads(handle.read())

print(evaluate.evaluate(8,priority))

