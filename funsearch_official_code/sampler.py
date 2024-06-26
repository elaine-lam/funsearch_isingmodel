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


class LLM:
  """Language model that predicts continuation of provided source code."""

  #construct LLM
  def __init__(self, samples_per_prompt: int) -> None:
    self._samples_per_prompt = samples_per_prompt
    ollama_llm = Ollama(model = 'llama3')
    memory = ConversationBufferMemory()
    parser = StrOutputParser()
    loader = TextLoader('data.txt',encoding = 'utf-8')
    document = loader.load()
    spliter = RecursiveCharacterTextSplitter(chunk_size = 250,chunk_overlap = 50)
    chunks = spliter.split_documents(document)
    vector_storage = FAISS.from_documents(chunks, OllamaEmbeddings(model='llama3'))
    retriever = vector_storage.as_retriever()

    template = ("""You are expert in Computer Science. You can only respond in python code and don't need to give usage examples. Any algorithm you create should be different from previous model version and not using random.
                You are going to provide creative input on building the python code to minimize the ground state of an NxN 2D Ising model by finding a deterministic algorithm for assigning spins based on the site interactions and magnetism. 
                Output an priority(h,J) function that takes NxN matrix h of the magnetism at each site and a NxNx3 tensor J that gives the interaction between the corresponding site and its nearest neighbors.
                The priority function should return a N^2 by 2 list which has priorities for assigning spins to -1 and 1, similar to the example functions.

    Context:{context}            
    Input:{question}
    History:{history}
    """)
    lprompt = PromptTemplate.from_template(template=template)
    lprompt.format(
        context = ' Here is a context to use',
        question = '',
        history = 'memory'
    )

    m_chain = ConversationChain(
        llm=ollama_llm,
        memory=memory,
    )
    result = RunnableParallel(context = retriever,question = RunnablePassthrough(), history = m_chain)
    chain = result |lprompt |ollama_llm |parser

    self.chain = chain
    if self.chain is None:
      print("can't construst chain!!")

  def _process(self, code: str) -> str:
    #remove descriptions on top
    def_location = code.find("```")
    code = code[int(def_location)+3:]
    return code
  
  #generate response from prompt
  def _draw_sample(self, prompt: str) -> str:
    response = self.chain.invoke(prompt)
    response = self._process(response)
    print(response)
    return response

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
      prompt = self._database.get_prompt()
      samples = self._llm.draw_samples(prompt.code)
      # This loop can be executed in parallel on remote evaluator machines.
      for sample in samples:
        chosen_evaluator = np.random.choice(self._evaluators)
        chosen_evaluator.analyse(
            sample, prompt.island_id, prompt.version_generated)
