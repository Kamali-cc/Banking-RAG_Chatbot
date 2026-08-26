"""
STEP 8: Streamlit UI for the Banking RAG Chatbot.

Run locally: streamlit run app.py
Deploy free: push to GitHub, then deploy via Streamlit Community Cloud
             (share.streamlit.io) or Hugging Face Spaces (Streamlit SDK).
"""

import streamlit as st
from rag_chain import load_retriever, build_chain, answer_question

st.set_page_config(page_title="Banking FAQ Assistant", page_icon="🏦")

st.title("🏦 Banking FAQ Assistant")
st.caption("RAG-powered chatbot grounded in real bank FAQ data — built with LangChain, "
           "Hugging Face embeddings, FAISS, and an open-source LLM.")


@st.cache_resource
def load_resources():
    vectorstore = load_retriever()
    chain = build_chain(backend="groq")  # switch to "huggingface" if using HF Inference API
    return vectorstore, chain


vectorstore, chain = load_resources()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about accounts, loans, cards, KYC, or online banking."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a banking question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            result = answer_question(query, vectorstore, chain)
            st.markdown(result["answer"])
            if result["sources"]:
                st.caption(f"📂 Source categories: {', '.join(result['sources'])}")
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
        **Stack:**
        - LangChain (orchestration)
        - Hugging Face `all-MiniLM-L6-v2` (embeddings, local & free)
        - FAISS (vector store)
        - GPT-OSS 20B via Groq (open-source LLM, fast inference)
        - Streamlit (UI)

        **Guardrail:** Refuses to answer if the query is outside
        the FAQ knowledge base, instead of hallucinating.
        """
    )
