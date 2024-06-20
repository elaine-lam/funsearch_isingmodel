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

#from written code
from evaluate2D import evaluate


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
    
    # ensure that function has something with correct name and inputs to be called
'''    first_line_end = code.find('\n') ##TODO
    code = code[first_line_end:]        # if there is nothing to find with "find" than the output is -1
    code = "def priority(h,J):" + code'''

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
    
'''def usage():
    state = 0
    msg = "nice execute"
    score:int
    try:
        score = evaluate(dataset2D, priority) # type: ignore - priority function will come from LLM
        state = 0
        print("Here's the result for the current code:")
    except Exception as e:
        print("An error occurred during code execution:", e)
        state = 2
        msg = str(e)
    finally:
        return state, score, msg'''

with open('data2D.txt', 'rb') as handle:  # import data
    dataset2D = pickle.loads(handle.read())
with open("evaluation.txt", 'r') as file:
    evaluate_info = file.read()

ollama_llm = Ollama(model = 'llama3')
memory = ConversationBufferMemory()
parser = StrOutputParser()
loader = TextLoader('data.txt',encoding = 'utf-8')
document = loader.load()
spliter = RecursiveCharacterTextSplitter(chunk_size = 200,chunk_overlap = 50)
chunks = spliter.split_documents(document)
vector_storage = FAISS.from_documents(chunks, OllamaEmbeddings(model='llama3'))
retriever = vector_storage.as_retriever()

template = ("""You are expert in Computer Science. You can only respond in python code and don't need to give usage examples. Any algorithm you create should be different from previous model version.
            You are going to provide creative input on building the python code to minimize the ground state of an NxN 2D Ising model by finding a deterministic, algorithm for assigning spins based on the site interactions and magnetism.
            Output an priority(h,J) function that takes NxN matrix h of the magnetism at each site and a NxNx2 tensor J that gives the interaction between the corresponding site and its nearest neighbors. 
            The priority function should return a N^2 by 2 list which has priorities for assigning spins to -1 and 1, similar to the example functions. Any necessary variables should be defined within the function as nothing else is given 
Context:{context}            
Input:{question}
History:{history}
""")
lprompt = PromptTemplate.from_template(template=template)
lprompt.format(
    context = 'Here is context to use:', #This is the function that will be used to evaluate the output: \n' + evaluate_info,
    question = '',
    history = 'memory'
)

m_chain = ConversationChain(
    llm=ollama_llm,
    memory=memory,
)

result = RunnableParallel(context = retriever,question = RunnablePassthrough(), history = m_chain)
chain = result |lprompt |ollama_llm |parser

count = 0
while count<2:
    # msg = input("user: ")
    # if msg.lower() == "exit":
    #     break
    msg = "Please generate me a python code of finding a minimize ground state of the Ising model to solve the Ising problem base the the given dataset."
    response = chain.invoke(msg)
    code = process(response)
    with open("priority_funcs.txt","a") as file:
        file.writelines(code + '\n')
    print(code)
    count += 1
'''      state = execute_code_with_timeout(code, 10)
        if state == 0:
            usage_state, u_msg = usage()
            while usage_state != 0:
                msg = "Please help me to correct the model provided in the last conversation according to the error message: \n" + u_msg
                response = chain.invoke(msg)
                code = process(response)
                print(code)
                state = execute_code_with_timeout(code, 10)
                usage_state, u_sc, u_msg = usage()
            file1.writelines(str(count) + ': score' + str(u_sc) + '\n' + code + '\n')
            count += 1'''
