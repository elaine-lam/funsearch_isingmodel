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

#for timeout
import timeout
import signal

def process(code):
    #remove the describtion part
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
        # Execute the code object
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
    N = 5  # Number of spins
    J = 2  # Coupling constant
    state = 0
    msg = "nice execute"
    try:
        ground_state, min_energy = ising_ground_state(N, J)
        state = 0
        print("Here's the result for the current code:")
        print("Ground state:", ground_state)
        print("Minimum energy:", min_energy)
    except Exception as e:
        print("An error occurred during code execution:", e)
        state = 2
        msg = str(e)
    finally:
        return state, msg

ollama_llm = Ollama(model = 'llama3')
memory = ConversationBufferMemory()
parser = StrOutputParser()
loader = TextLoader('data.txt',encoding = 'utf-8')
document = loader.load()
spliter = RecursiveCharacterTextSplitter(chunk_size = 200,chunk_overlap = 50)
chunks = spliter.split_documents(document)
vector_storage = FAISS.from_documents(chunks, OllamaEmbeddings(model='llama3'))
retriever = vector_storage.as_retriever()

template = ("""You are expert in Computer Science. You are going to provide creative model on building the python code of finding a minimize ground state of the Ising model to solve the Ising problem base the the given dataset. 
You can only response by python code. The algorithm of the created model should be different from the models of the previous version and can not include random inside. Everytime, please give an ising_ground_state(N, J) function that return ground_state and also min_energy. No need to provide the example on usage.

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

with open("codeHistory.txt", "a") as file1:
    count = 0
    while count<2:
        # msg = input("user: ")
        # if msg.lower() == "exit":
        #     break
        msg = "Please generate me a python code of finding a minimize ground state of the Ising model to solve the Ising problem base the the given dataset."
        response = chain.invoke(msg)
        code = process(response)
        print(code)
        state = execute_code_with_timeout(code, 10)
        if state == 0:
            usage_state, u_msg = usage()
            while usage_state != 0:
                msg = "Please help me to correct the model provided in the last conversation according to the error message: \n" + u_msg
                response = chain.invoke(msg)
                code = process(response)
                print(code)
                state = execute_code_with_timeout(code, 10)
                usage_state, u_msg = usage()
            file1.writelines(str(count) + ': ' + code + '\n')
            count += 1
