"""
app.py
------
Streamlit front-end: a clean, ChatGPT-style chat interface backed by a
LlamaParse -> Markdown -> Chunk -> Embed -> Retrieve -> Claude RAG pipeline.

Run with: streamlit run app.py
"""

import os
import time
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import parser as pdf_parser
import chunker
import vectorstore
import rag

load_dotenv()

st.set_page_config(page_title="PDF Chat — RAG Assistant", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Styling — clean, minimal, ChatGPT-ish
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { .stApp{
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
} }
    section[data-testid="stSidebar"] { background-color: #171922; border-right: 1px solid #2a2d3a; }
    .chat-bubble-user {
        background:#2563eb22; border:1px solid #2563eb55; padding:12px 16px;
        border-radius:14px; margin:8px 0; max-width:90%; margin-left:auto;
    }
    .chat-bubble-assistant {
        background:#1c1f2a; border:1px solid #2a2d3a; padding:12px 16px;
        border-radius:14px; margin:8px 0; max-width:90%;
    }
    .source-pill {
        display:inline-block; background:#2a2d3a; color:#9ca3af; font-size:11px;
        padding:3px 8px; border-radius:999px; margin:2px 4px 0 0;
    }
    #MainMenu, footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sources"}]
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

UPLOAD_DIR = Path(tempfile.gettempdir()) / "pdf_rag_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Sidebar — keys, upload, processing, document library, controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📄 PDF Chat")
    st.caption("Multi-PDF RAG assistant powered by LlamaParse + Claude")

    llama_key = os.getenv("LLAMA_PARSE_API_KEY")

    with st.expander("🖥️ Local model (Ollama)", expanded=False):
        if rag.is_ollama_running():
            st.success("Ollama is running ✅")
            models = rag.list_ollama_models()
            if models:
                selected_model = st.selectbox("Model", models)
            else:
                st.warning("No models pulled yet. Run e.g. `ollama pull llama3.2:3b`")
                selected_model = rag.DEFAULT_MODEL
        else:
            st.error("Ollama isn't running.")
            st.caption("Install from ollama.com, then run `ollama serve` and `ollama pull llama3.2:3b` in a terminal.")
            selected_model = rag.DEFAULT_MODEL

    st.markdown("### 📤 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button("⚙️ Process documents", use_container_width=True, type="primary"):
        if not llama_key:
            st.error("Please add your LlamaParse API key above first.")
        elif not uploaded_files:
            st.error("Please upload at least one PDF.")
        else:
            progress = st.progress(0, text="Saving uploads…")
            saved_paths = []
            for f in uploaded_files:
                p = UPLOAD_DIR / f.name
                p.write_bytes(f.getbuffer())
                saved_paths.append(str(p))

            progress.progress(20, text="Parsing PDFs with LlamaParse…")
            try:
                parsed = pdf_parser.parse_pdfs(saved_paths, api_key=llama_key)
            except Exception as e:
                st.error(f"Parsing failed: {e}")
                st.stop()

            progress.progress(60, text="Chunking markdown…")
            total_chunks = 0
            for filename, md_path in parsed.items():
                md_text = Path(md_path).read_text(encoding="utf-8")
                doc_name = Path(md_path).stem
                records = chunker.chunk_and_save(doc_name, md_text)
                vectorstore.add_chunks(records)
                total_chunks += len(records)
                if doc_name not in st.session_state.processed_files:
                    st.session_state.processed_files.append(doc_name)

            progress.progress(100, text="Done!")
            time.sleep(0.4)
            progress.empty()
            st.success(f"Processed {len(parsed)} PDF(s) into {total_chunks} chunks ✅")

    st.markdown("### 📚 Document library")
    sources = vectorstore.list_sources()
    if sources:
        for s in sources:
            st.markdown(f"<span class='source-pill'>📄 {s}</span>", unsafe_allow_html=True)
    else:
        st.caption("No documents processed yet.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("♻️ Reset all data", use_container_width=True):
            try:
                vectorstore.reset_collection()
            except Exception:
                pass
            st.session_state.messages = []
            st.session_state.processed_files = []
            st.rerun()

    n_results = st.slider("Chunks retrieved per question", 2, 10, 5)

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.markdown("## 💬 Ask your documents")

if not vectorstore.list_sources():
    st.info("👈 Upload and process at least one PDF from the sidebar to get started.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📄"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            pills = " ".join(
                f"<span class='source-pill'>{s['source']} · chunk {s['chunk_index']}</span>"
                for s in msg["sources"]
            )
            st.markdown(pills, unsafe_allow_html=True)

# Chat input
question = st.chat_input("Ask a question about your PDFs…")

if question:
    if not vectorstore.list_sources():
        st.error("Please process at least one PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant", avatar="📄"):
        placeholder = st.empty()
        final_text, final_sources = "", []
        for partial_text, matches in rag.answer_question(
            question, chat_history=history, n_results=n_results, model=selected_model
        ):
            final_text, final_sources = partial_text, matches
            placeholder.markdown(final_text + "▌")
        placeholder.markdown(final_text)

        if final_sources:
            pills = " ".join(
                f"<span class='source-pill'>{s['source']} · chunk {s['chunk_index']}</span>"
                for s in final_sources
            )
            st.markdown(pills, unsafe_allow_html=True)

            # NEW: Show the retrieved chunks
            st.divider()
            st.subheader("📄 Retrieved Chunks")

            for i, chunk in enumerate(final_sources):
                with st.expander(
                    f"Chunk {chunk['chunk_index']} | "
                    f"{chunk['source']} | "
                    f"Score: {chunk['score']:.2f}"
                ):
                    st.markdown(chunk["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": final_text, "sources": final_sources}
    )
