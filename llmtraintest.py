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
    
def usage():
    state = 0
    msg = "nice execute"
    score = 10
    try:
        score = evaluate(dataset2D, priority)
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

ollama_llm = Ollama(model = 'llama3')
memory = ConversationBufferMemory()
parser = StrOutputParser()
loader = TextLoader('data.txt',encoding = 'utf-8')
document = loader.load()
spliter = RecursiveCharacterTextSplitter(chunk_size = 200,chunk_overlap = 50)
chunks = spliter.split_documents(document)
vector_storage = FAISS.from_documents(chunks, OllamaEmbeddings(model='llama3'))
retriever = vector_storage.as_retriever()

template = ("""You are expert in Computer Science. You can only respond in python code and don't need to give usage examples. The function must be different than any previous functions.
            You are going to provide creative input on building python code to minimize the ground state of an NxN 2D Ising model by finding a deterministic, algorithm for assigning spins based on the site interactions and magnetism.
            Output a function called priority(h,J) that takes NxN matrix h of the magnetism at each site and a NxNx2 tensor J that gives the interaction between the corresponding site and its nearest neighbors. 
            The priority function should return a N^2 by 2 list which has priorities for assigning spins to -1 and 1, similar to the example functions
Context:{context}            
Input:{question}
History:{history}
""")
lprompt = PromptTemplate.from_template(template=template)
lprompt.format(
    context = 'Here is context to use:', 
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
    exec("def priority(h,J):\n\traise Exception('Function should have name priority(h,J)')")  # reset so if no priority function written by LLM then old one won't be called
    msg = """Write an algorithm that has the same sized inputs and same sized outputs as the given algorithms:
    
    def priority(h,J):  # uses h matrices: score is -1.717
    N = len(h)
    priority = np.zeros((N**2,2))
    for i in range(N):
      for j in range(N):
        if h[i,j] > 0:
            priority[(i*N+j),0] = h[i,j]
        else:
            priority[(i*N+j),1] = -1*h[i,j]
    return(priority)
    
    def priority(h,J):  #LLM written function - only one that actually works, no modification: 1.715
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
    return priorities
    """
    response = chain.invoke(msg)
    print(response)
    code = process(response)
    print(code)
    state = execute_code_with_timeout(code, 10)
    if state == 0:
        usage_state, u_sc, u_msg = usage()
        error_count = 0
        while usage_state != 0 and error_count < 5 and state == 0:
            msg = "Correct this function according to the error message: \n" + u_msg
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
