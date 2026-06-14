from dotenv import load_dotenv
import glob
import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

folders = glob.glob("knowledge_base/*")
print(folders)
documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

print(f"Loaded {len(documents)} documents")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(f"Divided into {len(chunks)} chunks")
print(f"First chunk:\n\n{chunks[0]}")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
if os.path.exists(DB_NAME):
    Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_NAME)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")


# Debugging only
# collection = vectorstore._collection
# count = collection.count()

# sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
# dimensions = len(sample_embedding)
# print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")