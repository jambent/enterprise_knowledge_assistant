import json
from types import SimpleNamespace
import src.assistant_agents.citation as citation


def test_get_citation_returns_agent_response_and_passes_context(monkeypatch):
    # no-op dotenv
    monkeypatch.setattr(citation, "load_dotenv", lambda *a, **k: None)

    # fake env url
    monkeypatch.setattr(citation.os, "getenv", lambda k, d=None: "http://fake")

    # fake ChatOllama that returns an identifiable object
    fake_llm = object()
    monkeypatch.setattr(citation, "ChatOllama", lambda **k: fake_llm)

    captured = {}

    # fake create_agent that captures system_prompt and model then returns a fake agent
    def fake_create_agent(model=None, system_prompt=None):
        captured["model"] = model
        captured["system_prompt"] = system_prompt

        class FakeAgent:
            def invoke(self, payload):
                captured["payload"] = payload
                # return last message content as the "citation"
                return {"messages": [SimpleNamespace(content="ignored"), SimpleNamespace(content="chosen/source/path")]}

        return FakeAgent()

    monkeypatch.setattr(citation, "create_agent", fake_create_agent)

    # build fake context: objects with page_content and metadata
    ctx = [
        SimpleNamespace(page_content="p1", metadata={"source": "input_files/doc1.txt"}),
        SimpleNamespace(page_content="p2", metadata={"source": "input_files/doc2.pdf"}),
    ]

    answer = "some answer text"

    result = citation.get_citation(answer, ctx)

    # returned value comes from the fake agent last message
    assert result == "chosen/source/path"

    # create_agent should have been called with the llm object we returned
    assert captured["model"] is fake_llm
    assert "Your role is to assess an answer" in captured["system_prompt"]

    # payload should be a dict with messages list whose first message content is JSON having answer and context
    payload = captured["payload"]
    assert isinstance(payload, dict)
    msgs = payload.get("messages")
    assert isinstance(msgs, list) and len(msgs) >= 1
    payload_json = json.loads(msgs[0]["content"])
    assert payload_json["answer"] == answer
    # context list items must have page_content and source keys with values from ctx
    assert isinstance(payload_json["context"], list)
    assert payload_json["context"][0]["page_content"] == "p1"
    assert payload_json["context"][0]["source"] == "input_files/doc1.txt"
    assert payload_json["context"][1]["source"] == "input_files/doc2.pdf"


def test_get_citation_handles_missing_metadata_source(monkeypatch):
    monkeypatch.setattr(citation, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(citation.os, "getenv", lambda k, d=None: "http://fake")
    monkeypatch.setattr(citation, "ChatOllama", lambda **k: object())

    captured = {}

    def fake_create_agent(model=None, system_prompt=None):
        class FakeAgent:
            def invoke(self, payload):
                captured["payload"] = payload
                return {"messages": [SimpleNamespace(content="ignored"), SimpleNamespace(content="")]}

        return FakeAgent()

    monkeypatch.setattr(citation, "create_agent", fake_create_agent)

    # item with missing 'source' in metadata
    ctx = [SimpleNamespace(page_content="p", metadata={})]
    ans = "irrelevant"

    out = citation.get_citation(ans, ctx)

    # returns whatever the fake agent last message was (empty string in this fake)
    assert out == ""
    payload_json = json.loads(captured["payload"]["messages"][0]["content"])
    # metadata.get('source') should produce None in the context list for that entry
    assert payload_json["context"][0]["source"] is None