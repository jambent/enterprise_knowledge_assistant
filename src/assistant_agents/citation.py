import os
import json
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama


def get_citation(answer, context):
    load_dotenv()

    url = os.getenv("AGENT_URL")
    llm = ChatOllama(base_url=url, model="llama3.2", temperature=0)

    SYSTEM_PROMPT = """
    Your role is to assess an answer to a question that has been provided by
    another agent and determine from the provided context which source
    document the agent has used to obtain the answer.
    You will be provided with the answer, and the
    context used to answer the question.
    The context will be a list of strings with the following format:
    {"page_content": page_content, "source": source}

    Your answer must be accurate, and must consist ONLY of the full filepath
    provided in the source value.
    The file extension of the document name in the source value MUST be
    excluded from the answer.
    If you don't know the answer, or the answer is not found in the context,
    YOU MUST respond with the empty string "", only.
    Do not guess, and do not make up a source document name.
    """
    context_list = [{"page_content": doc.page_content,
                     "source": doc.metadata.get("source")} for doc in context]
    context_dict = {}
    context_dict["answer"] = answer
    context_dict["context"] = context_list
    context_for_agent = json.dumps(context_dict)

    citation_agent = create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT)
    response = citation_agent.invoke({
        "messages": [
            {
                "role": "user",
                        "content": context_for_agent
            }
        ]
    })
    return response["messages"][-1].content
