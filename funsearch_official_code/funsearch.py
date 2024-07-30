# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A single-threaded implementation of the FunSearch pipeline."""
from collections.abc import Sequence
from typing import Any

import code_manipulation
import config as config_lib
import evaluator
import programs_database
import sampler
from evaluate import evaluate
from collections import Counter


def _extract_function_names(specification: str) -> tuple[str, str]:
  """Returns the name of the function to evolve and of the function to run."""
  run_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
  if len(run_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
  evolve_functions = list(
      code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
  if len(evolve_functions) != 1:
    raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
  return evolve_functions[0], run_functions[0]


def main(specification: str, inputs: Sequence[Any], config: config_lib.Config):
  """Launches a FunSearch experiment."""
  function_to_evolve, function_to_run = _extract_function_names(specification)
  template = code_manipulation.text_to_program(specification)
  database = programs_database.ProgramsDatabase(
      config.programs_database, template, function_to_evolve)
  load_backup = "./data/backups/program_db_prioritycap.pickle"
  if load_backup:
    database.load(load_backup)
  evaluators = []
  for _ in range(config.num_evaluators):
    evaluators.append(evaluator.Evaluator(
        database,
        template,
        function_to_evolve,
        function_to_run,
        inputs,
    ))
  # We send the initial implementation to be analysed by one of the evaluators.
  initial = template.get_function(function_to_evolve).body
  evaluators[0].analyse(initial, island_id=None, version_generated=None)

  samplers = [sampler.Sampler(database, evaluators, config.samples_per_prompt)
              for _ in range(config.num_samplers)]


  # This loop can be executed in parallel on remote sampler machines. As each
  # sampler enters an infinite loop, without parallelization only the first
  # sampler will do any work.

  for s in samplers:
    s.sample()

def run(samplers, iterations: int = -1):
  """Launches a FunSearch experiment."""

  try:
    # This loop can be executed in parallel on remote sampler machines. As each
    # sampler enters an infinite loop, without parallelization only the first
    # sampler will do any work.
    while iterations != 0:
      for s in samplers:
        s.sample()
      if iterations > 0:
        iterations -= 1
  except Exception as e:
    print(e)

if __name__ == '__main__':
  specification = '''import numpy as np
import seaborn as sns
from scipy.optimize import minimize
import itertools
import math
from collections import Counter
from evaluate import evaluate
from scipy.stats import gmean
import funsearch

@funsearch.run
def evaluate(n: int, priority) -> int:
  """Returns the size of an `n`-dimensional cap set."""
  capset = None
  try:
    capset = solve(n, priority)
  except Exception as e:
    log(e)
  finally:
    return len(capset)
    
@funsearch.evolve
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

'''
  inputstr = "n8_size512.txt"
  inputs = inputstr.split(',')
  config = config_lib.Config()
  main(specification, inputs, config)