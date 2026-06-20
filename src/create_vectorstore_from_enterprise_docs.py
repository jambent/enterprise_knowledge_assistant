from langgraph.graph import StateGraph
from src.vectorstore_creation_agents.convert_inputs_to_markdown \
    import convert_input_files_to_markdown
from src.vectorstore_creation_agents.create_vectorstore \
    import run_vectorstore_agent


def docs_to_markdown_node(state):
    state["docs_to_markdown"] = convert_input_files_to_markdown()
    return state

def create_vectorstore_node(state):
    state["create_vectorstore"] = run_vectorstore_agent()
    return state

graph = StateGraph(dict)

graph.add_node("docs_to_markdown", docs_to_markdown_node)
graph.add_node("create_vectorstore", create_vectorstore_node)

graph.set_entry_point("docs_to_markdown")
graph.add_edge("docs_to_markdown", "create_vectorstore")

app = graph.compile()
app.invoke({})

# for event in app.stream({}):
#     print(event)

