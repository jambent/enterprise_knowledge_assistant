from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def retriever_agent(db_name, embedding_model):
    # Connect to existing vectorstore, ensuring embedding model is the same as used to create it
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 10})

if __name__ == "__main__":
    DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
    embedding_model = "all-MiniLM-L6-v2"
    retriever = retriever_agent(DB_NAME, embedding_model)
    print(f"Retriever created with {retriever.vectorstore._collection.count()} documents")