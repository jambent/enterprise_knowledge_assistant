import sys
import types
import importlib
from types import SimpleNamespace
from unittest.mock import mock_open
import builtins
import pytest
import os


def _import_loader_with_fakes(monkeypatch, pdf_texts=None, docx_texts=None):
    # Fake langchain_core.document_loaders.BaseLoader and langchain_core.documents.Document
    fake_doc_loaders = types.ModuleType("langchain_core.document_loaders")
    fake_doc_loaders.BaseLoader = object
    fake_documents = types.ModuleType("langchain_core.documents")

    class FakeDocument:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    fake_documents.Document = FakeDocument

    monkeypatch.setitem(sys.modules, "langchain_core.document_loaders", fake_doc_loaders)
    monkeypatch.setitem(sys.modules, "langchain_core.documents", fake_documents)

    # Fake pypdf.PdfReader
    fake_pypdf = types.ModuleType("pypdf")

    class FakePage:
        def __init__(self, text):
            self._text = text
        def extract_text(self):
            return self._text

    class FakePdfReader:
        def __init__(self, path):
            texts = pdf_texts or []
            self.pages = [FakePage(t) for t in texts]

    fake_pypdf.PdfReader = FakePdfReader
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    # Fake docx.Document
    fake_docx = types.ModuleType("docx")

    class FakeDocx:
        def __init__(self, path):
            texts = docx_texts or []
            self.paragraphs = [SimpleNamespace(text=t) for t in texts]

    fake_docx.Document = FakeDocx
    monkeypatch.setitem(sys.modules, "docx", fake_docx)

    # Now import/reload the module under test
    if "src.multi_format_loader" in sys.modules:
        del sys.modules["src.multi_format_loader"]
    loader_mod = importlib.import_module("src.multi_format_loader")
    importlib.reload(loader_mod)
    return loader_mod


def test_loads_txt_file(monkeypatch):
    loader_mod = _import_loader_with_fakes(monkeypatch)
    m = mock_open(read_data="line1\nline2")
    monkeypatch.setattr(builtins, "open", m)
    loader = loader_mod.MultiFormatLoader("some/path/file.txt")
    doc = next(loader.lazy_load())
    assert doc.page_content == "line1\nline2"
    assert os.path.normpath(doc.metadata["source"]).endswith(os.path.normpath("some/path/file.txt"))
    assert doc.metadata["file_type"] == ".txt"


def test_loads_pdf_file(monkeypatch):
    loader_mod = _import_loader_with_fakes(monkeypatch, pdf_texts=["p1", "p2"])
    loader = loader_mod.MultiFormatLoader("docs/sample.pdf")
    doc = next(loader.lazy_load())
    assert doc.page_content == "p1\np2"
    assert doc.metadata["file_type"] == ".pdf"
    assert os.path.normpath(doc.metadata["source"]).endswith(os.path.normpath("docs/sample.pdf"))


def test_loads_docx_file(monkeypatch):
    loader_mod = _import_loader_with_fakes(monkeypatch, docx_texts=["para1", "para2"])
    loader = loader_mod.MultiFormatLoader("docs/doc.docx")
    doc = next(loader.lazy_load())
    assert doc.page_content == "para1\npara2"
    assert doc.metadata["file_type"] == ".docx"
    assert os.path.normpath(doc.metadata["source"]).endswith(os.path.normpath("docs/doc.docx"))


def test_unsupported_extension_raises(monkeypatch):
    # minimal fakes to allow import
    loader_mod = _import_loader_with_fakes(monkeypatch)
    loader = loader_mod.MultiFormatLoader("weird/file.customext")
    with pytest.raises(ValueError):
        list(loader.lazy_load())