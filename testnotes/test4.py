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

doc_path = r"testdata\encryption.pdf"
model = "llama 3.2"

#Local PDF file upload
if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("done loading...")
else: 
    print("Upload a PDF file")

#preview first page

content = data[0].page_content
print(content[:100])

