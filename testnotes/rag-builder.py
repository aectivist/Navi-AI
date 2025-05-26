import os 
os.environ["PATH"] += os.pathsep + r"C:\poppler-24.08.0\Library\bin"

import ollama 
import os
url = "http://localhost:11434/api/generate"

#1: ingest PDF Files
#2: Extract Text from PDF Files and split into small chunks
#3: Send chunks to embedding model
#4: Save embeddings to a vector DB
#5: Perform similarity search on vector database to find similar documents
#6: retrieve similar documents and present to user


#====== 1, INGEST PDF
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

doc_path = r"testnotes\testdata\encryption.pdf"


#Local PDF file upload
if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("done loading...")
else: 
    print("Upload a PDF file")

#preview first page

content = data[0].page_content
#print(content[:100])


#====== 2, EXTRACT text from PDF + split into smaller chunks
from langchain_ollama import OllamaEmbeddings #get wrapper of ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter #recursively split text
from langchain_community.vectorstores import Chroma #DB

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=300
)

chunks = text_splitter.split_documents(data)
print("done splitting....")

#print(f"Number of chunks: {len(chunks)}")
#print(f"Example chunk: {chunks[0]}")

#====== 2, 
import ollama
ollama.pull("nomic-embed-text")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model = "nomic-embed-text"),collection_name="simple-rag", 
    )

print("done adding to vector database.....")

#====== 4, RETRIEVAL
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser 

from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
model = "llama 3.2:latest"
llm = ChatOllama(model = model)

QUERY_PROMPT = PromptTemplate(
    input_variables = ["question"],
    template="""You are an AI language model assistant. Your task is to generate different five different versions of the given users question to retrieve relevant documents from a vector database.  
    By generating multiple perspectives on the user's question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
)

#RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context":retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res = chain.invoke(input=("What is the document about?",))
print(res)