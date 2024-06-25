import ast

from collections.abc import Collection, Sequence
import pickle
from typing import Any

import numpy as np
import sampler
import evaluator

from evaluate2D import evaluate

from myExeception import TimeoutError, timeout_handler

class Llama3LLM(sampler.LLM):

    def __init__(self, samples_per_prompt: int) -> None:
        super().__init__(samples_per_prompt)
   
    

#llm = Llama3LLM(2)
#sample_code = llm.draw_samples("please generate me a model of ising model.")

class mySandbox(evaluator.Sandbox):
    @staticmethod
    def compile_code(program:str):
        programspace = {}
        parsed_functions = ast.parse(program)
        compiled_code = compile(parsed_functions, filename="<ast>", mode="exec")
        exec(compiled_code, programspace)
        return programspace

    @staticmethod
    def get_testdata(test_input:str):
        with open(test_input, 'rb') as handle:  # import data
            test_data = pickle.loads(handle.read())
        return test_data

    def run(self, program: str, function_to_run: str, test_input: str, timeout_seconds: int) -> tuple[Any, bool]:
        runable = False
        programspace = mySandbox.compile_code(program)
        test_data = mySandbox.get_testdata(test_input)
        test_output = evaluate(test_data, programspace[function_to_run])
        print(test_output)
        if isinstance(test_output, (int, float)):
            runable = True
        return test_output, runable
    

program = '''def priority(h,J):  #LLM written function - only one that actually works, no modification: 1.715
    N = len(h)
    state = [[-1 if h[i][j] > 0 else 1 for j in range(N)] for i in range(N)]
    priorities = []
    for i in range(N):
        for j in range(N):
            total_spin = 0
            for k in range(3):
                site = (i + ((k-1)%2 - 1)) % N
                total_spin += state[site][j]
            if h[i][j] > 0:
                priorities.append((total_spin, 1))
            else:
                priorities.append((total_spin, -1))
    return priorities'''
function_to_run = "priority"
test_inputs = ["data2D.txt"]
timeout_seconds = 100

sandbox = mySandbox()
scores_per_test = {}
for test_input in test_inputs:
    test_output, runs_ok = mySandbox.run(sandbox, program, function_to_run, test_input,timeout_seconds)
    scores_per_test[test_input] = test_output

print(scores_per_test)

