import sys
import types
import importlib
from pathlib import Path

def test_retriever_agent_calls_embeddings_and_chroma(monkeypatch):
    # 1) Provide a fake config
    fake_config = {
        "paths": {"vector_db": "fake_db_dir"},
        "embedding": {"model_name": "fake-embed-model"},
        "chunks": {"top_k_chunks": 3},
    }
    fake_load_mod = types.ModuleType("src.load_config")
    fake_load_mod.load_config = lambda path: fake_config
    monkeypatch.setitem(sys.modules, "src.load_config", fake_load_mod)

    # 2) Fake HuggingFaceEmbeddings to capture model_name and return a marker object
    captured = {}
    def FakeHuggingFaceEmbeddings(model_name=None):
        captured["model_name"] = model_name
        return "fake-emb-instance"
    fake_hf_mod = types.ModuleType("langchain_huggingface")
    fake_hf_mod.HuggingFaceEmbeddings = FakeHuggingFaceEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_huggingface", fake_hf_mod)

    # 3) Fake Chroma to capture init args and expose as_retriever
    class FakeChroma:
        def __init__(self, persist_directory=None, embedding_function=None):
            captured["persist_directory"] = persist_directory
            captured["embedding_function"] = embedding_function
        def as_retriever(self, search_kwargs):
            captured["search_kwargs"] = search_kwargs
            return "fake-retriever"
    fake_chroma_mod = types.ModuleType("langchain_chroma")
    fake_chroma_mod.Chroma = FakeChroma
    monkeypatch.setitem(sys.modules, "langchain_chroma", fake_chroma_mod)

    # 4) Import/reload the module under test so it uses the fakes above
    if "src.assistant_agents.retrieval" in sys.modules:
        del sys.modules["src.assistant_agents.retrieval"]
    retrieval = importlib.import_module("src.assistant_agents.retrieval")
    importlib.reload(retrieval)

    # 5) Call the factory and assert behavior
    retr = retrieval.retriever_agent()
    assert retr == "fake-retriever"
    assert captured["model_name"] == fake_config["embedding"]["model_name"]

    expected_db_path = str(Path(retrieval.__file__).parent.parent.parent / fake_config["paths"]["vector_db"])
    assert captured["persist_directory"] == expected_db_path
    assert captured["embedding_function"] == "fake-emb-instance"
    assert captured["search_kwargs"] == {"k": fake_config["chunks"]["top_k_chunks"]}