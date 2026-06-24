import sys
import types
import importlib


def _fake_graph_and_nodes(monkeypatch):
    called = {
        "convert_called": False,
        "run_called": False,
        "graph_invoked": False,
        "state": None,
    }

    fake_convert_module = types.ModuleType(
        "src.vectorstore_creation_agents.convert_inputs_to_markdown"
    )

    def fake_convert_input_files_to_markdown():
        called["convert_called"] = True
        return "converted"

    fake_convert_module.convert_input_files_to_markdown = fake_convert_input_files_to_markdown
    monkeypatch.setitem(
        sys.modules,
        "src.vectorstore_creation_agents.convert_inputs_to_markdown",
        fake_convert_module,
    )

    fake_create_module = types.ModuleType("src.vectorstore_creation_agents.create_vectorstore")

    def fake_run_vectorstore_agent():
        called["run_called"] = True
        return "vectorstore-built"

    fake_create_module.run_vectorstore_agent = fake_run_vectorstore_agent
    monkeypatch.setitem(
        sys.modules,
        "src.vectorstore_creation_agents.create_vectorstore",
        fake_create_module,
    )

    fake_langgraph = types.ModuleType("langgraph")
    fake_graph_mod = types.ModuleType("langgraph.graph")

    class FakeStateGraph:
        def __init__(self, container_type):
            self.container_type = container_type
            self.nodes = {}
            self.entry = None
            self.edges = {}

        def add_node(self, name, fn):
            self.nodes[name] = fn

        def set_entry_point(self, name):
            self.entry = name

        def add_edge(self, src, dst):
            self.edges[src] = dst

        def compile(self):
            return self

        def invoke(self, state):
            state = self.nodes[self.entry](state)
            next_node_name = self.edges.get(self.entry)
            if next_node_name:
                state = self.nodes[next_node_name](state)
            called["graph_invoked"] = True
            called["state"] = state
            return state

    fake_graph_mod.StateGraph = FakeStateGraph
    monkeypatch.setitem(sys.modules, "langgraph", fake_langgraph)
    monkeypatch.setitem(sys.modules, "langgraph.graph", fake_graph_mod)

    return called


def test_import_runs_graph_and_node_functions(monkeypatch):
    called = _fake_graph_and_nodes(monkeypatch)

    if "src.create_vectorstore_from_enterprise_docs" in sys.modules:
        del sys.modules["src.create_vectorstore_from_enterprise_docs"]

    import src.create_vectorstore_from_enterprise_docs as module
    importlib.reload(module)

    assert called["graph_invoked"] is True
    assert called["convert_called"] is True
    assert called["run_called"] is True
    assert called["state"]["docs_to_markdown"] == "converted"
    assert called["state"]["create_vectorstore"] == "vectorstore-built"