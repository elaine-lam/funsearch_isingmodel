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

"""Class for sampling new programs."""
import ast
from collections.abc import Collection, Sequence

import numpy as np

import evaluator
import programs_database

#for constructing the  LLM
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

from error_log import log



class LLM:
  """Language model that predicts continuation of provided source code."""

  #construct LLM
  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt
    ollama_llm = Ollama(model = 'llama3')
    parser = StrOutputParser()
    loader = TextLoader('priority_funcs.txt',encoding = 'utf-8')
    document = loader.load()
    spliter = RecursiveCharacterTextSplitter(chunk_size = 250,chunk_overlap = 50)
    chunks = spliter.split_documents(document)
    vector_storage = FAISS.from_documents(chunks, OllamaEmbeddings(model='llama3'))
    retriever = vector_storage.as_retriever()

    template = ("""You are expert in Computer Science. You can only respond in python code and don't need to give usage examples. The function can be similar as the provided functions.
            You are going to provide creative input on building python code to minimize the ground state of an 2-dimensional Ising model of side length N by finding a deterministic, algorithm for assigning spins based on the site interactions and magnetism.
            Output a function called priority(N,h,J) that takes the grid size N, a N^2 matrix h of the magnetism at each site and a 4 x N^2 tensor J that gives the interaction between the corresponding site and its nearest neighbors. 
            The priority function should return a N^2 by 2 list which has priorities for assigning spins to -1 and 1. The indent-size for the code is 2 spacebar.

    Context:{context}      
    Input:{question}
  """)
    lprompt = PromptTemplate.from_template(template=template)
    lprompt.format(
        context = ' Here is a context to use',
        question = '',
    )
    result = RunnableParallel(context = retriever, question = RunnablePassthrough())
    chain = result |lprompt |ollama_llm |parser

    self.chain = chain
    if self.chain is None:
      print("can't construst chain!!")

  def _process(self, code: str) -> str:
    #remove the description part
    substr = "```"
    sub_location = code.find(substr)
    def_location = code.find("def")
    if sub_location >-1 and def_location > sub_location:
      code = code[int(def_location):]
    sub_location = code.find(substr)
    def_location = code.find("def")
    if sub_location >-1 and sub_location > def_location:
      code = code[:int(sub_location)]
    py_location = code.find("python")
    if py_location >-1:
      code = code[int(py_location)+7:] 
    return code
  
  def _try_parse(self, code:str):
    working = False
    msg = None
    tree = None
    try:
      tree = ast.parse(code)
      working = True
    except Exception as e:
      msg = str(e)
    finally:
      del tree
      return working, msg

  
  #generate response from prompt
  def _draw_sample(self, prompt: str) -> str:
    response = self.chain.invoke(prompt)
    p_response = self._process(response)
    working, msg = self._try_parse(p_response)
    error_count = 0
    while not working and error_count < 10:
      if not working:
        temp_msg = p_response + "\nThe program also have the following error, please help me to correct the entire function:\n" + msg
      response = self.chain.invoke(temp_msg)
      p_response = self._process(response)
      working, msg = self._try_parse(p_response)
      error_count += 1
    if error_count >= 10:
      p_response = "pass"
    else:
      p_response = '\n'.join(p_response.splitlines()[1:])
    return p_response

  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
  """Node that samples program continuations and sends them for analysis."""

  def __init__(
      self,
      database: programs_database.ProgramsDatabase,
      evaluators: Sequence[evaluator.Evaluator],
      samples_per_prompt: int,
  ) -> None:
    self._database = database
    self._evaluators = evaluators
    self._llm = LLM(samples_per_prompt)

  def sample(self):
    """Continuously gets prompts, samples programs, sends them for analysis."""
    while True:
      try:
        prompt = self._database.get_prompt()
        samples = self._llm.draw_samples(prompt.code)
        # This loop can be executed in parallel on remote evaluator machines.
        for sample in samples:
          chosen_evaluator = np.random.choice(self._evaluators)
          chosen_evaluator.analyse(
              sample, prompt.island_id, prompt.version_generated)
      except Exception as e:
        log(e)
