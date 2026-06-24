from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.load_config import load_config


config = load_config("./src/config/retrieval_config.yaml")
DB_NAME = config["paths"]["vector_db"]
EMBEDDING_MODEL_NAME = config["embedding"]["model_name"]
TOP_K_CHUNKS = config["chunks"]["top_k_chunks"]


def retriever_agent():
    # Connect to existing vectorstore, ensuring embedding model is the same as
    # used to create it
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db_path = str(Path(__file__).parent.parent.parent / DB_NAME)
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K_CHUNKS})

