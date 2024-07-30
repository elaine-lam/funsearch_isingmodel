from collections import Counter
import pickle
import evaluate
import numpy as np
import math

def priority(vector, n):
  """
  Assigns a priority to a vector based on its sum of elements and the number of trailing zeros.

  Args:
  vector (tuple): A vector in the cap set
  n (int): The size of the cap set

  Returns:
  float: The priority of the vector
  """
  sum_elements = sum(vector)
  leading_zeros = len(str(sum_elements).lstrip('0'))
  
  frequency = Counter(vector)
  max_freq = max(frequency.values())
  min_freq = min(frequency.values())

  if (sum_elements <= n/2 and sum(1 for x in vector if x) >= n//2):
    return math.sqrt(n)
  elif (sum_elements > n/2 and sum(1 for x in vector if x) < n//2):
    return -math.sqrt(n)

  variance = sum((x - sum_elements/n)**2 for x in vector) / n

  zero_ratio = len([i for i in range(len(vector)) if vector[i] == 0]) / len(vector)
  unique_elements = len(set(vector))
  median = np.median(vector)

  # Improved calculation of trailing zeros
  trailing_zeros = sum(1 for x in str(sum_elements).zfill(n) if x == '0')

  return -(sum_elements / n + (n - sum(1 for x in vector if x)) ** 2 / n) + abs(sum_elements - n/2) * (n - sum(1 for x in vector if x)) / n - min((x**2/n) for x in vector) + max(-sum(vector), 0) + (max(vector) - min(vector)) / n - sum(sorted(vector)[::-1]) / n + math.sqrt(trailing_zeros) * n / len(vector) - sum(x for x in set(vector)) / n - leading_zeros/n - max_freq/min_freq - (max_freq/min_freq)**2 + zero_ratio + (unique_elements - 1) / n + abs(median - sum_elements/len(vector)) - min(vector)**3/(n**2) + math.exp(-n/(sum_elements + (n - sum(1 for x in vector if x)))) - abs(sum(x**6/(n*n*n*n*n) for x in vector)) * len([i for i in range(len(vector)) if vector[i] == 0]) / n - max_freq**3/(min_freq**2) - math.sqrt(n)


# with open("data2D.txt", 'rb') as handle:  # import data
#     test_data = pickle.loads(handle.read())

print(evaluate.evaluate(8,priority))

