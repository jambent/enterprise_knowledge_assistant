from pathlib import Path
import os
import json
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.load_config import load_config


def _make_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj


config = load_config("./src/config/retrieval_config.yaml")
DB_NAME = config["paths"]["vector_db"]
EMBEDDING_MODEL_NAME = config["embedding"]["model_name"]

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
db_path = str(Path(__file__).parent.parent.parent / DB_NAME)
vectorstore = Chroma(
    persist_directory=db_path,
    embedding_function=embeddings
)
data = vectorstore.get(include=["embeddings", "documents", "metadatas"])

os.makedirs("vectorstore_artifacts", exist_ok=True)
with open("vectorstore_artifacts/vectorstore_export.jsonl",
          "w", encoding="utf-8") as f:
    for i in range(len(data["documents"])):
        record = {
            "id": str(data["ids"][i]),
            "document": data["documents"][i],
            "metadata": _make_serializable(data["metadatas"][i]),
            "embedding": _make_serializable(data["embeddings"][i])
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
