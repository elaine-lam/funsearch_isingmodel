#for building the llama3
import os 
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

#pre import library
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import itertools
import pickle

#for timeout
import timeout
import signal

#from evaluate the score for the written code
from evaluate import evaluate


def process(code):
    #remove the description part
    substr = "```"
    sub_location = code.find(substr)
    code = code[int(sub_location)+3:]
    sub_location = code.find(substr)
    code = code[:int(sub_location)]

    #remove the import part
    def_location = code.find("def ")
    code = code[int(def_location):]

    #remove the sample usage part
    use_location = code.find("# Example usage")
    if use_location>-1:
        code = code[:int(use_location)]
    return code

def execute_code_with_timeout(codeStr, timeout_seconds):
    signal.signal(signal.SIGALRM, timeout.timeout_handler)
    signal.alarm(timeout_seconds)
    state = 0

    try:
        codeObject = compile(codeStr, 'sumstring', 'exec')
        # Execute the code object - make it a usable function
        exec(codeObject, globals())
        print("execution finished")
    except TimeoutError:
        print("Code execution timed out")
        state = 1
    except Exception as e:
        print("An error occurred during code execution:", e)
        state = 2
    finally:
        signal.alarm(0)  # Cancel the alarm
        return state
    
def usage():    #connect to the evaulate function to test the score
    state = 0
    msg = "nice execute"
    score = 10
    try:
        score = evaluate(dataset2D, priority) #type: ignore
        state = 0
        print("Here's the result for the current code:")
        print(score)
    except Exception as e:
        print("An error occurred during code execution:", e)
        state = 2
        msg = str(e)
    finally:
        return state, score, msg

with open('data2D.txt', 'rb') as handle:  # import data
    dataset2D = pickle.loads(handle.read())

#construct LLLM with different settings
ollama_llm = Ollama(model = 'llama3')   
memory = ConversationBufferMemory()
parser = StrOutputParser()
loader = TextLoader('data.txt',encoding = 'utf-8')
document = loader.load()
spliter = RecursiveCharacterTextSplitter(chunk_size = 200,chunk_overlap = 50)
chunks = spliter.split_documents(document)
vector_storage = FAISS.from_documents(chunks, OllamaEmbeddings(model='llama3'))
retriever = vector_storage.as_retriever()

#the instructions for the LLM to follow
template = ("""You are expert in Computer Science. You can only respond in python code and don't need to give usage examples. The function must be different than any previous functions.
            You are going to provide creative input on building python code to minimize the ground state of an 2-dimensional Ising model of side length N by finding a deterministic, algorithm for assigning spins based on the site interactions and magnetism.
            Output a function called priority(N,h,J) that takes the grid size N, a N^2 matrix h of the magnetism at each site and a 4 x N^2 tensor J that gives the interaction between the corresponding site and its nearest neighbors. 
            The priority function should return a N^2 by 2 list which has priorities for assigning spins to -1 and 1. 
Context:{context}            
Input:{question}
History:{history}
""")

#the format for the prompt and response
lprompt = PromptTemplate.from_template(template=template)
lprompt.format(
    context = 'Here is context to use:', 
    question = '',
    history = 'memory'
)

#chain all the settings into one, and use this as the following call for using LLM
m_chain = ConversationChain(
    llm=ollama_llm,
    memory=memory,
)
result = RunnableParallel(context = retriever,question = RunnablePassthrough(), history = m_chain)
chain = result |lprompt |ollama_llm |parser

count = 0   #number of loop, temp use for testing
while count<2:
    exec("def priority(N,h,J):\n\traise Exception('Function should have name priority(N,h,J)')")  # reset so if no priority function written by LLM then old one won't be called
    msg = """Write an algorithm using h and J to minimize the energy of the Ising model:"""   #code specification
    response = chain.invoke(msg)
    code = process(response)
    print(code)
    state = execute_code_with_timeout(code, 10)
    if state == 0:  # code compiles correctly
        #do the loop to make sure the generated code is working
        usage_state, u_sc, u_msg = usage() 
        error_count = 0
        while usage_state != 0 and error_count < 5 and state == 0:
            msg = "Correct this function called priority() according to the error message: \n" + u_msg
            response = chain.invoke(msg)
            code = process(response)
            print(code)
            state = execute_code_with_timeout(code, 10)
            if state != 0:
                break
            usage_state, u_sc, u_msg = usage()
            error_count += 1
        with open("priority_funcs.txt","a") as file:
            if usage_state == 0 and state == 0:
                file.writelines("\n\n#" + str(u_sc) + '\n'+ code)
        count += 1
