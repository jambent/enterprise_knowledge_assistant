import sys
import types
import importlib
from types import SimpleNamespace


def _import_with_fakes(monkeypatch, config):
    """
    Install lightweight fake modules into sys.modules so the target module can be
    imported without the real heavy dependencies.
    """
    # fake dotenv
    fake_dotenv = types.SimpleNamespace(load_dotenv=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    monkeypatch.setitem(sys.modules, "dotenv.load_dotenv", fake_dotenv.load_dotenv)

    # fake src.load_config module (so the module under test gets our config at import)
    fake_load_config_mod = types.SimpleNamespace(load_config=lambda p: config)
    monkeypatch.setitem(sys.modules, "src.load_config", fake_load_config_mod)

    # fake langchain_core.documents.Document
    fake_docs = types.ModuleType("langchain_core.documents")

    class FakeDocument:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    fake_docs.Document = FakeDocument
    monkeypatch.setitem(sys.modules, "langchain_core.documents", fake_docs)

    # fake langchain_text_splitters.RecursiveCharacterTextSplitter
    fake_splitters = types.ModuleType("langchain_text_splitters")

    class FakeSplitter:
        def __init__(self, chunk_size=None, chunk_overlap=None):
            pass

        def split_documents(self, documents):
            # keep documents as-is (one chunk per document)
            return documents

    fake_splitters.RecursiveCharacterTextSplitter = FakeSplitter
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", fake_splitters)

    # fake langchain_huggingface.HuggingFaceEmbeddings
    fake_hf = types.ModuleType("langchain_huggingface")

    class FakeEmbeddings:
        def __init__(self, model_name=None):
            self.model_name = model_name

    fake_hf.HuggingFaceEmbeddings = FakeEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_huggingface", fake_hf)

    # fake langchain_chroma.Chroma
    fake_chroma = types.ModuleType("langchain_chroma")

    class FakeCollection:
        def __init__(self, docs):
            self._docs = docs

        def count(self):
            return len(self._docs)

    class FakeChromaInstance:
        def __init__(self, persist_directory=None, embedding_function=None):
            self.persist_directory = persist_directory
            self.embedding_function = embedding_function
            # used only by delete_collection in source; no-op

        def delete_collection(self):
            self._deleted = True

    class FakeChroma:
        def __init__(self, persist_directory=None, embedding_function=None):
            self._inst = FakeChromaInstance(persist_directory, embedding_function)

        @staticmethod
        def from_documents(documents=None, embedding=None, persist_directory=None):
            obj = types.SimpleNamespace()
            obj._collection = FakeCollection(documents or [])
            return obj

    fake_chroma.Chroma = FakeChroma
    monkeypatch.setitem(sys.modules, "langchain_chroma", fake_chroma)

    # fake langchain.tools.tool decorator (no-op)
    fake_tools = types.ModuleType("langchain.tools")

    def tool_decorator(f=None, *args, **kwargs):
        if f is None:
            return lambda fn: fn
        return f

    fake_tools.tool = tool_decorator
    monkeypatch.setitem(sys.modules, "langchain.tools", fake_tools)

    # fake langchain.agents.create_agent (for run_vectorstore_agent test)
    fake_agents = types.ModuleType("langchain.agents")

    def create_agent(**k):
        # placeholder; individual tests will monkeypatch sys.modules["langchain.agents"].create_agent
        raise RuntimeError("create_agent not configured in fake; tests should override if needed")

    fake_agents.create_agent = create_agent
    monkeypatch.setitem(sys.modules, "langchain.agents", fake_agents)

    # fake langchain_ollama.ChatOllama
    fake_ollama = types.ModuleType("langchain_ollama")

    class FakeChatOllama:
        def __init__(self, base_url=None, model=None, temperature=None):
            self.base_url = base_url
            self.model = model
            self.temperature = temperature

    fake_ollama.ChatOllama = FakeChatOllama
    monkeypatch.setitem(sys.modules, "langchain_ollama", fake_ollama)

    # Ensure re-import of target module uses our fakes
    if "src.vectorstore_creation_agents.create_vectorstore" in sys.modules:
        del sys.modules["src.vectorstore_creation_agents.create_vectorstore"]
    mod = importlib.import_module("src.vectorstore_creation_agents.create_vectorstore")
    importlib.reload(mod)
    return mod


def test_build_vectorstore_reads_files_and_returns_count(tmp_path, monkeypatch):
    # prepare a tiny fake knowledge base on disk
    kb = tmp_path / "kb"
    folder = kb / "category"
    folder.mkdir(parents=True)
    file_path = folder / "doc1.md"
    file_path.write_text("hello world")

    config = {
        "paths": {"knowledge_base": str(kb), "vector_db": "vecdb"},
        "embedding": {"model_name": "fake-model"},
        "chunking": {"chunk_size": 1000, "chunk_overlap": 0},
        "file_loading": {"glob_pattern": "*.md", "encoding": "utf-8"},
    }

    mod = _import_with_fakes(monkeypatch, config)

    # call function under test
    result = mod.build_vectorstore()
    # ensure the returned message includes the number of documents (1 file -> 1 chunk)
    assert "1" in result and "Vectorstore" in result


def test_run_vectorstore_agent_invokes_agent(monkeypatch):
    config = {
        "paths": {"knowledge_base": "ignored", "vector_db": "vecdb"},
        "embedding": {"model_name": "fake-model"},
        "chunking": {"chunk_size": 1000, "chunk_overlap": 0},
        "file_loading": {"glob_pattern": "*.md", "encoding": "utf-8"},
    }

    # fake create_agent to return an object with invoke()
    def fake_create_agent(**k):
        class FakeAgent:
            def invoke(self, payload):
                return {"messages": [SimpleNamespace(content="ignored"), SimpleNamespace(content="agent-done")]}
        return FakeAgent()

    # prepare fakes and then override create_agent
    mod = _import_with_fakes(monkeypatch, config)
    # override the create_agent used by the module
    monkeypatch.setitem(sys.modules, "langchain.agents", types.ModuleType("langchain.agents"))
    importlib.reload(sys.modules["langchain.agents"])
    sys.modules["langchain.agents"].create_agent = fake_create_agent

    # ensure module sees the new create_agent
    importlib.reload(mod)

    # call run_vectorstore_agent and assert we get the agent message back
    out = mod.run_vectorstore_agent()
    assert "agent-done" in out