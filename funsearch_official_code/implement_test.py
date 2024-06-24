import ast

from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
import sampler
import evaluator

class Llama3LLM(sampler.LLM):

    def __init__(self, samples_per_prompt: int) -> None:
        super().__init__(samples_per_prompt)
   
    

llm = Llama3LLM(2)
sample_code = llm.draw_samples("please generate me a model of ising model.")

class mySandbox(evaluator.Sandbox):
    def run(self, program: str, function_to_run: str, test_input: str, timeout_seconds: int) -> tuple[Any, bool]:
        score = 0
        runable = False
        return score, runable