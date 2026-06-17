import gradio as gr
from dotenv import load_dotenv
from src.reasoning import answer_question
from src.citation import get_citation

load_dotenv(override=True)


def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result


def chat(history):
    last = history[-1]

    # ✅ FORCE STRING EXTRACTION
    if isinstance(last, dict):
        last_message = last["content"]
    elif isinstance(last, (list, tuple)):
        last_message = last[0]
    else:
        last_message = last

    # ✅ CRITICAL: ensure it's a string
    if isinstance(last_message, list):
        last_message = " ".join(map(str, last_message))
    else:
        last_message = str(last_message)

    prior = history[:-1]

    answer, context = answer_question(last_message, prior)
    cited_document_name = get_citation(last_message, answer, context)
    if cited_document_name != "":
        answer_with_citation = f"{answer}\n\n[Source: {cited_document_name}]"
    else:
        answer_with_citation = answer
    history.append({
        "role": "assistant",
        "content": answer_with_citation
    })

    return history, format_context(context)



def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Apexon Knowledge Assistant") as ui:
        gr.Markdown("# 🏢 Apexon Knowledge Assistant\nAsk me anything about Apexon!")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="💬 Conversation", height=600
                )
                message = gr.Textbox(
                    #label="Your Question",
                    placeholder="Ask anything about Apexon...",
                    show_label=False,
                    lines=1,
                    max_lines=1,
                    autofocus=True
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    ui.launch(inbrowser=True, theme=theme)


if __name__ == "__main__":
    main()