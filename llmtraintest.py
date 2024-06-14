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

#for timeout
import timeout
import signal

def process(code):
    substr = "```"
    sub_location = code.find(substr)
    code = code[int(sub_location)+3:]
    sub_location = code.find(substr)
    code = code[:int(sub_location)]

    def_location = code.find("def ")
    code = code[int(def_location):]
    return code

def execute_code_with_timeout(codeStr, timeout_seconds):
    signal.signal(signal.SIGALRM, timeout.timeout_handler)
    signal.alarm(timeout_seconds)
    state = 0
    codeObject = compile(codeStr, 'sumstring', 'exec')

    try:
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
You can only response by python code. The model created should be different from the models of the previous version.

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
    while True:
        # msg = input("user: ")
        # if msg.lower() == "exit":
        #     break
        msg = "Please generate me a python code of finding a minimize ground state of the Ising model to solve the Ising problem base the the given dataset."
        response = chain.invoke(msg)
        code = process(response)
        file1.writelines(str(count) + ': ' + code)
        file1.write('\n')
        execute_code_with_timeout(code)
        print("executed")
        count += 1
