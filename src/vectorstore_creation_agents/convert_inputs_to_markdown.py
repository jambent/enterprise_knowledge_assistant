import os
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from dotenv import load_dotenv
from src.multi_format_loader import MultiFormatLoader


def convert_input_files_to_markdown():
    """Preprocess input files and convert them to Markdown format"""
    load_dotenv()

    url = os.getenv("AGENT_URL")
    llm = ChatOllama(base_url=url, model="llama3.2", temperature=0)
    agent = create_agent(
        model=llm,
        system_prompt="""You are an agent whose job is to convert files to
        Markdown format.
        Do not under any circumstances include any text in your response that
        is not in the original document.
        Do not include any headings or formatting that do not exist in the
        original file."""
    )

    input_root = "input_files"
    output_root = "knowledge_base"
    os.makedirs(output_root, exist_ok=True)

    for root, _, files in os.walk(input_root):
        rel_path = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, rel_path)
        os.makedirs(output_dir, exist_ok=True)

        for file in files:
            input_file_path = os.path.join(root, file)
            print(f"Processing {file}...")

            markdown_output = []
            loader = MultiFormatLoader(input_file_path)
            for doc in loader.lazy_load():
                response = agent.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": f"""Convert this file content
                                        to Markdown:\n\n{doc.page_content}"""
                        }
                    ]
                })
                markdown_output.append(response["messages"][-1].content)

            final_md = "\n\n".join(markdown_output)
            filename = os.path.splitext(file)[0]
            output_file_path = os.path.join(output_dir, filename + ".md")
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(final_md)
            print(f"{file} converted to Markdown")
