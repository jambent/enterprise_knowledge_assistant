from dotenv import load_dotenv
import glob
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from src.load_config import load_config


load_dotenv()
config = load_config("./src/config/embedding_config.yaml")

KB_PATH = config["paths"]["knowledge_base"]
DB_NAME = config["paths"]["vector_db"]
EMBEDDING_MODEL_NAME = config["embedding"]["model_name"]
CHUNK_SIZE = config["chunking"]["chunk_size"]
CHUNK_OVERLAP = config["chunking"]["chunk_overlap"]
GLOB_PATTERN = config["file_loading"]["glob_pattern"]
ENCODING = config["file_loading"]["encoding"]


@tool
def build_vectorstore(_: str = "") -> str:
    """Loads markdown files, splits them, embeds them, and stores in Chroma"""
    print("Building vectorstore")
    documents = []
    folders = glob.glob(f"{KB_PATH}/*")

    for folder in folders:
        doc_type = os.path.basename(folder)
        md_files = glob.glob(
            os.path.join(
                folder,
                GLOB_PATTERN),
            recursive=True)

        for file_path in md_files:
            with open(file_path, "r", encoding=ENCODING) as f:
                content = f.read()

            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": file_path, "doc_type": doc_type}
                )
            )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    db_path = str(Path(__file__).parent.parent.parent / DB_NAME)

    if os.path.exists(db_path):
        Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        ).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

    return f"Vectorstore built with {
        vectorstore._collection.count()} documents."


def run_vectorstore_agent() -> str:
    """Creates a vectorstore agent that can build the vector database
      from the markdown knowledge base"""
    url = os.getenv("AGENT_URL")
    llm = ChatOllama(base_url=url, model="llama3.2", temperature=0)

    vectorstore_agent = create_agent(
        tools=[build_vectorstore],
        model=llm
    )

    response = vectorstore_agent.invoke(
        {"messages": [
            {"role": "user",
            "content": "Build the vector database from the Markdown knowledge base"
            }
        ]})
    return response["messages"][-1].content
