import streamlit as st
import asyncio
from agent import execute_agent

st.set_page_config(page_title="Grünenthal AI Agent", layout="centered")

st.title("🧠 Grünenthal AI Chatbot")
st.caption("Ask questions about the Neo4j Healthcare Analytics graph database, FDA adverse events, or Grünenthal's financial data according to the 2023 report.")


if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask a question...")

if user_input:
    with st.spinner("Thinking..."):
        try:
            try:
                result = asyncio.run(execute_agent(user_input))
            except Exception as e:
                st.error(f"**System**: Error - {e}")
                pass
        except RuntimeError:
            # Workaround for: "RuntimeError: There is no current event loop"
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(execute_agent(user_input))
        
        st.session_state.history.append((user_input, result))

for q, a in reversed(st.session_state.history):
    st.markdown(f"**You**: {q}")
    st.markdown(f"**AI**: {a}")