import os 
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

#pre import library
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

ollama_llm = Ollama(model = 'llama3')
parser = StrOutputParser()
loader = TextLoader('data.txt',encoding = 'utf-8')
document = loader.load()
spliter = RecursiveCharacterTextSplitter(chunk_size = 200,chunk_overlap = 50)
chunks = spliter.split_documents(document)
vector_storage = FAISS.from_documents(chunks, OllamaEmbeddings(model='llama3'))
retriever = vector_storage.as_retriever()

template = ("""You are expert in Computer Science. You need to provide creative model to slove Ising Problem in Python.
            
Context:{context}
Question:{question}
""")
prompt = PromptTemplate.from_template(template=template)
prompt.format(
    context = ' Here is a context to use',
    question = ' This is a question to answer'
)

result = RunnableParallel(context = retriever,question = RunnablePassthrough())
chain = result |prompt | ollama_llm | parser

print(chain.invoke("please generate a model for the ising problem in python"))