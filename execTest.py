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
import seaborn as sns
from scipy.optimize import minimize

#for timeout
import timeout

def process(code):
    substr = "```"
    sub_location = code.find(substr)
    code = code[int(sub_location)+3:]
    sub_location = code.find(substr)
    code = code[:int(sub_location)]

    def_location = code.find("def ")
    code = code[int(def_location):]
    return code

# def execute_code(code):
#     codeObject = compile(code, 'sumstring', 'exec')
#     exec(codeObject, globals())

def usage():
    N = 3  # Number of spins
    J = 1  # Coupling constant
    ground_state, min_energy = ising_ground_state(N, J)
    print("Ground state:", ground_state)
    print("Minimum energy:", min_energy)

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

with open("tempStore.txt", "a") as file1:
    count = 0
    while count == 0:
        # msg = input("user: ")
        # if msg.lower() == "exit":
        #     break
        msg = "can you generate me a model for ising problem"
        response = chain.invoke(msg)
        print(response)
        code = process(response)
        # file1.writelines(str(count) + ': ' + code)
        # file1.write('\n')
        print(code)
        state = timeout.execute_code_with_timeout(code, 10)
        if state == 0:
            N = 5
            J = 1.0
            ground_state, min_energy = ising_ground_state(N, J)
            print("Ground state:", ground_state)
            print("Minimum energy:", min_energy)
        # usage()
        count += 1
