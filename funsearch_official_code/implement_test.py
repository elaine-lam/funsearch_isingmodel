#this is a file the test the usage of each class for the original code from Deepmind 

import ast

from collections.abc import Collection, Sequence
import pickle
from typing import Any

import numpy as np
import sampler
import evaluator

from evaluate import evaluate

from myExeception import TimeoutError, timeout_handler

# LLM test
# class Llama3LLM(sampler.LLM):

#     def __init__(self, samples_per_prompt: int) -> None:
#         super().__init__(samples_per_prompt)
   
    

#llm = Llama3LLM(2)
#sample_code = llm.draw_samples("please generate me a model of ising model.")

# Sandbox test
# class mySandbox(evaluator.Sandbox):
#     @staticmethod
#     def compile_code(program:str):
#         programspace = {}
#         parsed_functions = ast.parse(program)
#         compiled_code = compile(parsed_functions, filename="<ast>", mode="exec")
#         exec(compiled_code, programspace)
#         return programspace

#     @staticmethod
#     def get_testdata(test_input:str):
#         with open(test_input, 'rb') as handle:  # import data
#             test_data = pickle.loads(handle.read())
#         return test_data

#     def run(self, program: str, function_to_run: str, test_input: str, timeout_seconds: int) -> tuple[Any, bool]:
#         runable = False
#         programspace = mySandbox.compile_code(program)
#         test_data = mySandbox.get_testdata(test_input)
#         test_output = evaluate(test_data, programspace[function_to_run])
#         print(test_output)
#         if isinstance(test_output, (int, float)):
#             runable = True
#         return test_output, runable
    

program = '''import numpy as np
def priority(N, D, h, J):
    priority = np.zeros((N**D, D))
    
    for i in range(N**D):
        # Calculate the priority value based on N, D, h, and J
        # Replace this with your actual calculation
        priority[i][0] = (i % N) + (i // N) * D
        
        # Set the second element of the priority matrix to zero for now
        priority[i][1] = 0
    
    return(priority)
'''
function_to_run = "priority"
test_inputs = ["data2D.txt","data3D.txt"]
timeout_seconds = 100

sandbox = evaluator.Sandbox()
scores_per_test = {}
# for test_input in test_inputs:
#     test_output, runs_ok = evaluator.Sandbox.run(sandbox, program, function_to_run, test_input,timeout_seconds)
#     scores_per_test[test_input] = test_output
#     print(scores_per_test)

tree = ast.parse(program)
print(tree)
