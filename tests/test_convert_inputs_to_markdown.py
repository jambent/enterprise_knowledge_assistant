import os
import builtins
from types import SimpleNamespace
from unittest.mock import mock_open
from src.vectorstore_creation_agents.convert_inputs_to_markdown import convert_input_files_to_markdown


class FakeLoader:
    def __init__(self, path):
        self.path = path

    def lazy_load(self):
        return [
            SimpleNamespace(page_content="chunk1"),
            SimpleNamespace(page_content="chunk2"),
        ]


class FakeAgent:
    def __init__(self):
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return {
            "messages": [
                SimpleNamespace(content="ignored"),
                SimpleNamespace(content="converted"),
            ]
        }


def _assert_open_called_with_path(m, expected_path, expected_mode="w", expected_encoding="utf-8"):
    # Accept if any of the open() calls match the expected path/mode/encoding.
    matches = []
    for call in m.call_args_list:
        call_args, call_kwargs = call
        if call_args:
            called_path = call_args[0]
            called_mode = call_args[1] if len(call_args) > 1 else call_kwargs.get("mode")
        else:
            called_path = call_kwargs.get("file") or call_kwargs.get("path") or ""
            called_mode = call_kwargs.get("mode")
        called_encoding = call_kwargs.get("encoding")
        if os.path.normpath(called_path) == os.path.normpath(expected_path) and called_mode == expected_mode and called_encoding == expected_encoding:
            matches.append(call)
    assert matches, f"open was not called with path {expected_path}. Calls: {m.call_args_list}"


def test_happy_path(monkeypatch):
    fake_agent = FakeAgent()
    m = mock_open()

    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.os.walk",
        lambda root: [("input_files", [], ["file1.txt"])],
    )
    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.os.makedirs",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(builtins, "open", m)

    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.ChatOllama",
        lambda **k: object(),
    )
    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.create_agent",
        lambda **k: fake_agent,
    )
    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.MultiFormatLoader",
        FakeLoader,
    )

    monkeypatch.setenv("AGENT_URL", "http://fake")

    convert_input_files_to_markdown()

    expected = os.path.join("knowledge_base", os.path.relpath("input_files", "input_files"), "file1.md")
    _assert_open_called_with_path(m, expected)

    handle = m()
    written = "".join(call.args[0] for call in handle.write.call_args_list)
    assert written == "converted\n\nconverted"
    assert len(fake_agent.calls) == 2


def test_nested_directories(monkeypatch):
    fake_agent = FakeAgent()
    m = mock_open()

    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.os.walk",
        lambda root: [("input_files/subdir", [], ["file.txt"])],
    )
    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.os.makedirs",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(builtins, "open", m)

    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.ChatOllama",
        lambda **k: object(),
    )
    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.create_agent",
        lambda **k: fake_agent,
    )
    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.MultiFormatLoader",
        FakeLoader,
    )

    monkeypatch.setenv("AGENT_URL", "http://fake")

    convert_input_files_to_markdown()

    expected = os.path.join("knowledge_base", "subdir", "file.md")
    _assert_open_called_with_path(m, expected)


def test_no_files(monkeypatch):
    fake_agent = FakeAgent()
    m = mock_open()

    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.os.walk",
        lambda root: [("input_files", [], [])],
    )
    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.os.makedirs",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(builtins, "open", m)

    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.ChatOllama",
        lambda **k: object(),
    )
    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.create_agent",
        lambda **k: fake_agent,
    )
    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.MultiFormatLoader",
        FakeLoader,
    )

    monkeypatch.setenv("AGENT_URL", "http://fake")

    convert_input_files_to_markdown()

    # Ensure no output .md file under knowledge_base was created
    def is_kb_call(call):
        call_args, call_kwargs = call
        if call_args:
            p = call_args[0]
        else:
            p = call_kwargs.get("file") or call_kwargs.get("path") or ""
        return os.path.normpath(p).startswith(os.path.normpath("knowledge_base"))
    assert not any(is_kb_call(call) for call in m.call_args_list)


def test_agent_input_format(monkeypatch):
    fake_agent = FakeAgent()
    m = mock_open()

    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.os.walk",
        lambda root: [("input_files", [], ["file.txt"])],
    )
    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.os.makedirs",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(builtins, "open", m)

    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.ChatOllama",
        lambda **k: object(),
    )
    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.create_agent",
        lambda **k: fake_agent,
    )
    monkeypatch.setattr(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown.MultiFormatLoader",
        FakeLoader,
    )

    monkeypatch.setenv("AGENT_URL", "http://fake")

    convert_input_files_to_markdown()

    payload = fake_agent.calls[0]
    assert "Convert this file content" in payload["messages"][0]["content"]
    assert "chunk1" in payload["messages"][0]["content"]