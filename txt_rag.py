import os
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

import dotenv

dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


file_path = "path to your txt file"

with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()
    docs = [content]

vectorstore = Chroma.from_texts(
    texts=docs, embedding=OpenAIEmbeddings(), persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    if isinstance(docs[0], str):
        return "\n\n".join(docs)
    else:
        return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("invoking...")
user_query = input("User: ")
while user_query.lower() != "exit":
    result = rag_chain.invoke(user_query)
    print("Chatbot:", result)
    user_query = input("User: ")
    print("invoking...")

print("Exiting...")
