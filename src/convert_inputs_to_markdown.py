
from pypdf import PdfReader
import os
import glob
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from dotenv import load_dotenv
# from openai import OpenAI

# client = OpenAI()



def convert_input_files_to_markdown():
    load_dotenv()
    """Preprocess input files and convert them to Markdown format."""
    url = os.getenv("AGENT_URL")
    llm = ChatOllama(base_url=url, model="llama3.2", temperature=0)
    #print(llm)
    agent = create_agent(
        model=llm,
        system_prompt="You are a file converter. Convert input files to Markdown."
    )
    filenames = os.listdir("./input_files")

    for filename in filenames:
        file_extension = os.path.splitext(filename)[1]
        filepath = os.path.join("./input_files", filename)
        match file_extension:
            case ".pdf":
        #filepath = os.path.join("./input_files", filename)
                reader = PdfReader(filepath)
                pages = [p.extract_text() for p in reader.pages]
            case _:
                print(f"Unsupported file type: {file_extension}. Skipping {filename}.")
                continue

        markdown_output = []

        for page in pages:
            response = agent.invoke({
                "messages":[
                    {
                        "role": "user",
                        "content": f"""Convert this to Markdown:\n\n{page}.
                                        Do not include any text that is not
                                        in the original document."""
                    }
                ]
            })
            markdown_output.append(response["messages"][-1].content)

        final_md = "\n\n".join(markdown_output)
        file_extension = os.path.splitext(filename)[1]
        with open(f"knowledge_base/{os.path.basename(filename).replace(file_extension, '.md')}", "w", encoding="utf-8") as f:
            f.write(final_md)


if __name__ == "__main__":
    convert_input_files_to_markdown()
    