import json
import logging
import getpass
from logging.handlers import RotatingFileHandler
from datetime import datetime as dt, timezone
import time
import streamlit as st
from dotenv import load_dotenv

from src.reasoning import answer_question
from src.citation import get_citation


load_dotenv(override=True)
USER = getpass.getuser()

# -----------------------------
# Logging setup
# -----------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            #"timestamp": dt.fromtimestamp(record.created, tz=timezone.utc).isoformat() + "Z",
            "timestamp": dt.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "user": USER,
            "session_id": st.session_state.get("session_id", "unknown"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


@st.cache_resource
def setup_logger():
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = RotatingFileHandler(
            "app.log",
            maxBytes=50000000,
            backupCount=3
        )
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    return logger

logger = setup_logger()



# -----------------------------
# Helpers
# -----------------------------
# def format_context(context):
#     result = "## 📚 Relevant Context\n\n"
#     for doc in context:
#         result += f"**Source:** {doc.metadata['source']}\n\n"
#         result += doc.page_content + "\n\n---\n\n"
#     return result


def run_chat(user_message, history):
    logger.info(f"user: {user_message}")

    answer, context = answer_question(user_message, history)
    cited_document_name = get_citation(user_message, answer, context)

    if cited_document_name != "":
        final_answer = f"{answer}\n\n[Source: {cited_document_name}]"
    else:
        final_answer = answer

    logger.info(f"assistant: {final_answer}")

    return final_answer, context


# -----------------------------
# Streamlit UI
# -----------------------------
# def main():
#     st.set_page_config(page_title="Apexon Knowledge Assistant", layout="wide")

#     st.title("🏢 Apexon Knowledge Assistant")
#     st.write("Ask me anything about Apexon!")

#     # ✅ Initialize session state
#     if "history" not in st.session_state:
#         st.session_state.history = []

#     if "context" not in st.session_state:
#         st.session_state.context = ""

#     # ✅ Layout (2 columns)
#     col1, col2 = st.columns([1, 1])

#     # -----------------------------
#     # Chat column
#     # -----------------------------
#     with col1:
#         st.subheader("💬 Conversation")

#         for msg in st.session_state.history:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"])

#         # ✅ Chat input
#         user_input = st.chat_input("Ask anything about Apexon...")

#         if user_input:
#             # Add user message
#             st.session_state.history.append({
#                 "role": "user",
#                 "content": user_input
#             })

#             # Display user message immediately
#             with st.chat_message("user"):
#                 st.markdown(user_input)

#             # Run model
#             answer, context = run_chat(user_input, st.session_state.history[:-1])

#             # Add assistant message
#             st.session_state.history.append({
#                 "role": "assistant",
#                 "content": answer
#             })

#             # Save context
#             st.session_state.context = format_context(context)

#             # Display assistant message
#             with st.chat_message("assistant"):
#                 st.markdown(answer)

#     # -----------------------------
#     # Context column
#     # -----------------------------
#     with col2:
#         st.subheader("📚 Retrieved Context")

#         if st.session_state.context:
#             st.markdown(st.session_state.context)
#         else:
#             st.markdown("*Retrieved context will appear here*")


# def main():
#     st.set_page_config(page_title="Apexon Knowledge Assistant", layout="wide")

#     st.title("🏢 Apexon Knowledge Assistant")
#     #st.write("Ask me anything about Apexon!")
    
#     import uuid
#     if "session_id" not in st.session_state:
#         st.session_state.session_id = str(uuid.uuid4())

#     # ✅ Initialize session state
#     if "history" not in st.session_state:
#         st.session_state.history = []

#     # -----------------------------
#     # Chat UI (full width)
#     # -----------------------------
#     #st.subheader("💬 Conversation")

#     # for msg in st.session_state.history:
#     #     with st.chat_message(msg["role"]):
#     #         st.markdown(msg["content"])
    
#     center_col = st.columns([1, 3, 1])[1]

#     with center_col:
#         st.subheader("💬 Conversation")

#         for msg in st.session_state.history:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"])

#     # user_input = st.chat_input("Ask anything about Apexon...")

#     # ✅ Chat input
#         user_input = st.chat_input("Ask anything about Apexon...")

#         if user_input:
#             # Add user message
#             st.session_state.history.append({
#                 "role": "user",
#                 "content": user_input
#             })

#             # Display user message immediately
#             with st.chat_message("user"):
#                 st.markdown(user_input)

#             # Run model
#             answer, _ = run_chat(user_input, st.session_state.history[:-1])

#             # Add assistant message
#             st.session_state.history.append({
#                 "role": "assistant",
#                 "content": answer
#             })

#             # Display assistant response
#             with st.chat_message("assistant"):
#                 st.markdown(answer)



def main():
    st.set_page_config(page_title="Apexon Knowledge Assistant", layout="wide")

    st.title("🏢 Apexon Knowledge Assistant")

    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())

    if "history" not in st.session_state:
        st.session_state.history = []

    # ✅ Chat messages (centered)
    center_col = st.columns([1, 3, 1])[1]

    with center_col:
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ✅ Input (fixed bottom)
    user_input = st.chat_input("Ask anything about Apexon...")

    if user_input:
        # ✅ Add user message
        st.session_state.history.append({
            "role": "user",
            "content": user_input
        })

        with center_col:
            # Show user message immediately
            with st.chat_message("user"):
                st.markdown(user_input)

            # ✅ Assistant container
            with st.chat_message("assistant"):
                placeholder = st.empty()

                # ✅ Typing indicator
                placeholder.markdown("_Thinking..._")

                # Run model
                answer, _ = run_chat(user_input, st.session_state.history[:-1])

                # ✅ Streaming effect (character-by-character)
                streamed_text = ""
                for char in answer:
                    streamed_text += char
                    placeholder.markdown(streamed_text)
                    time.sleep(0.01)  # adjust speed here

        # ✅ Save final message AFTER streaming
        st.session_state.history.append({
            "role": "assistant",
            "content": answer
        })

        # ✅ Auto-scroll via rerun
        st.rerun()


# -----------------------------
if __name__ == "__main__":
    main()
