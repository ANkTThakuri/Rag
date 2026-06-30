# 📄 PDF Chat — Multi-PDF RAG Assistant (100% free, one API key)

A multi-PDF parser + RAG chat app that only needs a **free LlamaParse key**.
Everything else — embeddings and answering — runs locally on your machine.

**Pipeline:** Upload PDFs → parse to Markdown with **LlamaParse** → chunk the
markdown (header-aware + token-limited, with overlap) → embed chunks locally
with `sentence-transformers` into a persistent **Chroma** vector DB → retrieve
relevant chunks per question → answer with a **local model via Ollama**,
streamed into a clean ChatGPT-style chat UI built with Streamlit.

## Folder structure (created automatically as you use the app)

```
pdf-rag-app/
├── app.py              # Streamlit chat UI (run this)
├── parser.py            # LlamaParse: PDF -> Markdown
├── chunker.py            # Markdown -> chunks (saved as individual .md files)
├── vectorstore.py        # Chroma vector DB (local embeddings)
├── rag.py                 # Retrieval + local Ollama answer generation
├── parsed_markdown/       # One .md file per parsed PDF
├── chunks/<doc_name>/     # chunk_0001.md, chunk_0002.md, ... per document
└── vectorstore/chroma_db/ # Persisted embeddings (local, no cloud DB needed)
```

## 1. Install Ollama (for local answering — free, no key)

1. Download from **https://ollama.com** and install it (Windows/Mac/Linux all supported).
2. Open a terminal and pull a small model that'll run fine on a laptop:
   ```bash
   ollama pull llama3.2:3b
   ```
   (Other options: `phi3:mini` is smaller/faster, `llama3.1:8b` is better quality but slower —
   pull whichever fits your RAM. 8GB+ RAM recommended for the 3B model.)
3. Start the Ollama server (it may already auto-start after install):
   ```bash
   ollama serve
   ```
   Leave this running in the background while you use the app.

## 2. Install app dependencies

```bash
cd pdf-rag-app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Add your LlamaParse key

Copy `.env.example` to `.env` and fill in:

```
LLAMA_CLOUD_API_KEY=...   # free tier at https://cloud.llamaindex.ai
```

(You can also paste the key directly into the sidebar at runtime instead.)

## 4. Run it

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).
The sidebar will show "Ollama is running ✅" if step 1 worked correctly —
if it shows an error instead, make sure `ollama serve` is running in a
terminal.

## How it works

1. **Upload** one or more PDFs in the sidebar and click **Process documents**.
2. Each PDF is sent to **LlamaParse**, which returns clean Markdown
   (tables, headings, etc. preserved) — saved to `parsed_markdown/`.
3. Each markdown file is **chunked**: first split on headers, then any
   section longer than ~400 tokens is split further with a 60-token overlap
   so context isn't lost across chunk boundaries. Every chunk is saved as its
   own file under `chunks/<document>/chunk_000N.md` for inspection/debugging.
4. Chunks are embedded with a local `all-MiniLM-L6-v2` sentence-transformer
   model (no API cost) and stored in a persistent local Chroma collection.
5. When you ask a question, the top-N most relevant chunks are retrieved and
   passed to your **local Ollama model** along with the question and recent
   chat history — it answers using only that context and cites which
   document/chunk it used.

## UI features

- ChatGPT-style message bubbles with streaming responses
- Persistent chat history within a session
- Source citation pills under each answer (which PDF + chunk)
- Document library showing everything you've processed
- Local model picker (swap between any models you've pulled in Ollama)
- Adjustable retrieval depth (chunks per question)
- Clear chat / full reset controls
- Multi-file drag-and-drop upload with a progress bar

## Notes / things you may want to extend

- Answer quality depends entirely on which local model you run — small
  models (3B) are decent for direct factual Q&A but can struggle with
  nuanced or multi-document comparison questions. Try a bigger model
  (`llama3.1:8b`) if your laptop can handle it and quality matters more
  than speed.
- Chat history resets if you restart the app (in-memory only) — swap in
  `st.session_state` persistence to disk if you want it to survive restarts.
- The vector DB persists across restarts (it's written to
  `vectorstore/chroma_db/`), so re-processing the same PDFs isn't required
  every time — only re-run "Process documents" for *new* PDFs.
- If you ever do want stronger answers and don't mind a key + small cost,
  `rag.py` is small and easy to swap back to a hosted API (Claude, Gemini,
  etc.) — the retrieval/chunking pipeline stays exactly the same either way.
