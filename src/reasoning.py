import os
#import logging
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from chromadb import PersistentClient
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from pathlib import Path
from src.retrieval import retriever_agent


load_dotenv(override=True)

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"

embedding_model = "all-MiniLM-L6-v2"
#retriever = retriever_agent(DB_NAME, embedding_model)
retriever = retriever_agent()
chroma = PersistentClient(path=DB_NAME)
collection_name = "docs"
collection = chroma.get_or_create_collection(collection_name)

#RETRIEVAL_K = 20
FINAL_K = 3


url = os.getenv("AGENT_URL")
llm = ChatOllama(base_url=url, model="llama3.2", temperature=0)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Makara.
You are chatting with a user about Makara.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""
reasoning_agent = create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT)


class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

class Result:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


def rewrite_query(question, history=[]):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    message = f"""
You are in a conversation with a user, answering questions about the company Makara.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    response = reasoning_agent.invoke({
                "messages":[{"role": "system", "content": message}]
            })
    return response["messages"][-1].content

def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


def fetch_context_unranked(question):
    docs = retriever.invoke(question)
    chunks = [
        Result(page_content=doc.page_content, metadata=doc.metadata)
        for doc in docs
    ]
    return chunks


def fetch_context(original_question):
    #rewritten_question = rewrite_query(original_question)
    chunks1 = fetch_context_unranked(original_question)
    #chunks2 = fetch_context_unranked(rewritten_question)
    #chunks = merge_chunks(chunks1, chunks2)
    #return chunks[:FINAL_K]
    return chunks1[:FINAL_K]

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """
    Answer a question using RAG and return the answer and the retrieved context
    """
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    response = reasoning_agent.invoke({"messages":messages})
    return response["messages"][-1].content, chunks

if __name__ == "__main__":
    #print(answer_question("What is the first stage of the Enterprise Knowledge Assistant Capstone Project?", []))
    #print(answer_question("What is in the test document?", []))
    from src.citation import get_citation
    answer, context = answer_question("Hello", [])
    #cited_document_name = get_citation(user_message, answer, context)
    cited_document_name = get_citation(answer, context)
    print(answer)
    print(context)
    print(cited_document_name)