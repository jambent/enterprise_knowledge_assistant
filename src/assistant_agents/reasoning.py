import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from src.assistant_agents.retrieval import retriever_agent


load_dotenv(override=True)

url = os.getenv("AGENT_URL")
llm = ChatOllama(base_url=url, model="llama3.2", temperature=0)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Makara.
You are chatting with a user about Makara.
Your answer will be evaluated for accuracy, relevance and completeness,
 so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base
that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question.
Be accurate, relevant and complete.
"""
reasoning_agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT)


class Result:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {
            chunk.metadata['source']}:\n{
            chunk.page_content}" for chunk in chunks)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


def fetch_context(question):
    retriever = retriever_agent()
    docs = retriever.invoke(question)
    chunks = [
        Result(page_content=doc.page_content, metadata=doc.metadata)
        for doc in docs
    ]
    return chunks


def answer_question(
        question: str, history: list[dict] = []) -> tuple[str, list]:
    """
    Answer a question using RAG and return the answer and the retrieved context
    """
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    response = reasoning_agent.invoke({"messages": messages})
    return response["messages"][-1].content, chunks
