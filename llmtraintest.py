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
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

ollama_llm = Ollama(model = 'llama3')
memory = ConversationBufferMemory()
parser = StrOutputParser()
loader = TextLoader('data.txt',encoding = 'utf-8')
document = loader.load()
spliter = RecursiveCharacterTextSplitter(chunk_size = 200,chunk_overlap = 50)
chunks = spliter.split_documents(document)
vector_storage = FAISS.from_documents(chunks, OllamaEmbeddings(model='llama3'))
retriever = vector_storage.as_retriever()

template = ("""You are expert in Computer Science. 
            
Input:{input}
History:{history}
""")
lprompt = PromptTemplate.from_template(template=template)
lprompt.format(
    input = ' Here is a context to use',
    history = ' '
)

# result = RunnableParallel(context = retriever,question = RunnablePassthrough())
# chain = result |lprompt |ollama_llm |memory |parser
m_chain = ConversationChain(
    llm=ollama_llm,
    memory=memory,
    output_parser = parser,
    prompt = lprompt
)

while True:
    msg = input("user: ")
    if msg.lower() == "exit":
        break
    response = m_chain.invoke(msg)
    print(response["response"])

#log the prompt