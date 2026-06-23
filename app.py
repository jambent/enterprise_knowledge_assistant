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

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
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


def run_chat(user_message, history):
    logger.info(f"user: {user_message}")
    answer, context = answer_question(user_message, history)
    
    cited_document_name = get_citation(answer, context) or ""
    clean_source = cited_document_name.strip().strip('"')
    if clean_source:
        final_answer = f"{answer}\n\n[Source: {clean_source}]"
    else:
        final_answer = answer
    logger.info(f"assistant: {final_answer}")
    return final_answer, context



def main():
    st.set_page_config(page_title="Makara Knowledge Assistant", layout="wide")

    st.title("Knowledge Assistant")

    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())

    if "history" not in st.session_state:
        st.session_state.history = []

    center_col = st.columns([1, 3, 1])[1]
    with center_col:
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Input fixed to bottom
    user_input = st.chat_input("Ask anything about Makara...")

    if user_input:
        # Add user message
        st.session_state.history.append({
            "role": "user",
            "content": user_input
        })

        with center_col:
            # Show user message immediately
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                placeholder = st.empty()

                placeholder.markdown("_Thinking..._")
                answer, _ = run_chat(user_input, st.session_state.history[:-1])

                # Streaming effect
                streamed_text = ""
                for char in answer:
                    streamed_text += char
                    placeholder.markdown(streamed_text)
                    time.sleep(0.01)  # adjust speed here

        st.session_state.history.append({
            "role": "assistant",
            "content": answer
        })

        # Auto-scroll via rerun
        st.rerun()


if __name__ == "__main__":
    main()
