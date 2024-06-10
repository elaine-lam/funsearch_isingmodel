from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.llms import Ollama


ollama_llm = Ollama(model = 'llama3')
memory = ConversationBufferMemory()
chain = ConversationChain(
    llm=ollama_llm,
    memory=memory
)

while True:
    msg = input("user: ")
    if msg.lower() == "exit":
        break
    response = chain.invoke(msg)
    print(response["response"])