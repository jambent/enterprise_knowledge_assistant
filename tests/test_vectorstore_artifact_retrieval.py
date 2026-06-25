import sys
import os
import json
import types
import importlib
import builtins
from unittest.mock import mock_open


def _import_artifact_module_with_fakes(monkeypatch):
    fake_config = {"paths": {"vector_db": "fake_db"}, "embedding": {"model_name": "fake-model"}}

    fake_load_config = types.ModuleType("src.load_config")
    fake_load_config.load_config = lambda path: fake_config
    monkeypatch.setitem(sys.modules, "src.load_config", fake_load_config)

    fake_np = types.ModuleType("numpy")

    class FakeNdarray(list):
        def tolist(self):
            return list(self)

    class FakeInteger(int):
        def item(self):
            return int(self)

    class FakeFloating(float):
        def item(self):
            return float(self)

    fake_np.ndarray = FakeNdarray
    fake_np.integer = FakeInteger
    fake_np.floating = FakeFloating
    monkeypatch.setitem(sys.modules, "numpy", fake_np)

    fake_hf = types.ModuleType("langchain_huggingface")

    class FakeHuggingFaceEmbeddings:
        def __init__(self, model_name=None):
            self.model_name = model_name

    fake_hf.HuggingFaceEmbeddings = FakeHuggingFaceEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_huggingface", fake_hf)

    fake_chroma = types.ModuleType("langchain_chroma")

    class FakeChroma:
        def __init__(self, persist_directory=None, embedding_function=None):
            self.persist_directory = persist_directory
            self.embedding_function = embedding_function

        def get(self, include=None):
            return {
                "ids": [1, 2],
                "documents": ["doc1", "doc2"],
                "metadatas": [{"x": 1}, {"y": 2}],
                "embeddings": [FakeNdarray([1, 2]), FakeNdarray([3, 4])],
            }

        @staticmethod
        def from_documents(documents=None, embedding=None, persist_directory=None):
            return types.SimpleNamespace(_collection=types.SimpleNamespace(count=lambda: len(documents)))

    fake_chroma.Chroma = FakeChroma
    monkeypatch.setitem(sys.modules, "langchain_chroma", fake_chroma)

    monkeypatch.setattr(os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(os.path, "exists", lambda path: False)

    open_mock = mock_open()
    monkeypatch.setattr(builtins, "open", open_mock)

    if "src.utilities.vectorstore_artifact_retrieval" in sys.modules:
        del sys.modules["src.utilities.vectorstore_artifact_retrieval"]

    module = importlib.import_module("src.utilities.vectorstore_artifact_retrieval")
    return module, open_mock


def test_make_serializable_supports_numpy_like_objects(monkeypatch):
    module, _ = _import_artifact_module_with_fakes(monkeypatch)

    fake_array = module.np.ndarray([1, 2, 3])
    assert module._make_serializable(fake_array) == [1, 2, 3]

    assert module._make_serializable(module.np.integer(5)) == 5
    assert module._make_serializable(module.np.floating(3.14)) == 3.14
    assert module._make_serializable({"a": fake_array, "b": [module.np.integer(1)]}) == {"a": [1, 2, 3], "b": [1]}


def test_import_writes_jsonl_records_and_uses_config(monkeypatch):
    module, open_mock = _import_artifact_module_with_fakes(monkeypatch)

    assert module.embeddings.model_name == "fake-model"
    open_mock.assert_called_once_with("vectorstore_artifacts/vectorstore_export.jsonl", "w", encoding="utf-8")

    handle = open_mock()
    written = "".join(call.args[0] for call in handle.write.call_args_list)
    lines = [json.loads(line) for line in written.splitlines() if line.strip()]

    assert len(lines) == 2
    assert lines[0]["id"] == "1"
    assert lines[0]["document"] == "doc1"
    assert lines[0]["metadata"] == {"x": 1}
    assert lines[0]["embedding"] == [1, 2]
    assert lines[1]["id"] == "2"
    assert lines[1]["document"] == "doc2"