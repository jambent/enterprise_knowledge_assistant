import sys
import types
import importlib
from types import SimpleNamespace


def _import_reasoning_with_fakes(monkeypatch):
    # Provide fake langchain_ollama and langchain.agents modules so importing reasoning is safe.
    fake_ll = types.ModuleType("langchain_ollama")
    fake_ll.ChatOllama = lambda *a, **k: object()
    fake_agents = types.ModuleType("langchain.agents")
    fake_agents.create_agent = lambda **k: object()
    # Ensure parent package exists
    monkeypatch.setitem(sys.modules, "langchain_ollama", fake_ll)
    monkeypatch.setitem(sys.modules, "langchain", types.ModuleType("langchain"))
    monkeypatch.setitem(sys.modules, "langchain.agents", fake_agents)

    # Now import/reload the target module
    import src.assistant_agents.reasoning as reasoning
    importlib.reload(reasoning)
    return reasoning


def test_make_rag_messages_builds_system_and_includes_history_and_question(monkeypatch):
    reasoning = _import_reasoning_with_fakes(monkeypatch)

    chunks = [
        SimpleNamespace(page_content="chunkA", metadata={"source": "input_files/doc1.txt"}),
        SimpleNamespace(page_content="chunkB", metadata={"source": "input_files/doc2.txt"}),
    ]
    history = [{"role": "assistant", "content": "previous"}]
    q = "What is Makara?"

    msgs = reasoning.make_rag_messages(q, history, chunks)

    assert isinstance(msgs, list)
    assert msgs[0]["role"] == "system"
    # system message includes both extracts
    assert "Extract from input_files/doc1.txt" in msgs[0]["content"]
    assert "chunkA" in msgs[0]["content"]
    assert "Extract from input_files/doc2.txt" in msgs[0]["content"]
    # history preserved
    assert history[0] in msgs
    # last message is the user question
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == q


def test_fetch_context_uses_retriever_agent(monkeypatch):
    reasoning = _import_reasoning_with_fakes(monkeypatch)

    class FakeRetriever:
        def invoke(self, question):
            return [
                SimpleNamespace(page_content="pc1", metadata={"source": "s1"}),
                SimpleNamespace(page_content="pc2", metadata={"source": "s2"}),
            ]

    # Replace the retriever_agent factory used by the module
    monkeypatch.setattr(reasoning, "retriever_agent", lambda: FakeRetriever())

    chunks = reasoning.fetch_context("anything")
    assert len(chunks) == 2
    assert chunks[0].page_content == "pc1"
    assert chunks[0].metadata["source"] == "s1"


def test_answer_question_calls_reasoning_agent_and_returns_answer_and_chunks(monkeypatch):
    reasoning = _import_reasoning_with_fakes(monkeypatch)

    # Provide retriever that returns one doc
    class FakeRetriever:
        def invoke(self, question):
            return [SimpleNamespace(page_content="doccontent", metadata={"source": "src"})]

    monkeypatch.setattr(reasoning, "retriever_agent", lambda: FakeRetriever())

    # Create a fake reasoning agent that captures the payload and returns a final message
    class FakeReasoningAgent:
        def __init__(self):
            self.last_payload = None

        def invoke(self, payload):
            self.last_payload = payload
            return {"messages": [SimpleNamespace(content="ignored"), SimpleNamespace(content="final answer")]}

    fake_agent = FakeReasoningAgent()
    monkeypatch.setattr(reasoning, "reasoning_agent", fake_agent, raising=False)

    answer, chunks = reasoning.answer_question("help me", history=[{"role": "assistant", "content": "h"}])

    assert answer == "final answer"
    assert len(chunks) == 1
    # Ensure the reasoning agent was invoked with messages and that the user question is present
    assert fake_agent.last_payload is not None
    messages = fake_agent.last_payload.get("messages") if isinstance(fake_agent.last_payload, dict) else fake_agent.last_payload
    assert any(m.get("content", "") == "help me" or m.get("content") == "help me" for m in messages)