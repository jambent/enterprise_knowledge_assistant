import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama


def get_citation(question, answer, context):
    load_dotenv()

    url = os.getenv("AGENT_URL")
    llm = ChatOllama(base_url=url, model="llama3.2", temperature=0)

    SYSTEM_PROMPT = """
    Your role is to assess an answer to a question that has been provided by another agent
    and determine from the context, that the agent has used to
    answer the question, which source document referenced in the context 
    the agent has used to obtain the answer.
    You will be provided with a list containingthe question, the answer, and the 
    context used to answer the question. 
    The context will be a list of pydantic Result objects, each object being defined as follows:
    class Result(BaseModel):
        page_content: str
        metadata: dict
    The source document name is stored in the metadata dictionary under the key 'source'.
    Your answer must be accurate, and must be a valid string.
    The file extension of the source document name must be excluded from the answer.
    If you don't know the answer, or the answer is not found in the context, respond with 
    the empty string "", only.  
    Do not guess, and do not make up a source document name.
    """

    citation_agent = create_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT)
    response = citation_agent.invoke({
                "messages":[
                    {
                        "role": "user",
                        "content": f"[{question}, {answer}, {context}]"
                    }
                ]
            })
    return response["messages"][-1].content