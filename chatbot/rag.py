"""
rag.py
------
Retrieval-augmented generation: pulls relevant chunks from the vector store
and asks a LOCAL model (via Ollama) to answer the user's question using only
that retrieved context, with citations back to the source PDF.

No API key required for this step — Ollama runs entirely on your machine.
"""

import json
import requests
import vectorstore

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a set \
of PDF documents that have been parsed and chunked for you. You will be given \
retrieved excerpts (with source filename and chunk number) relevant to the \
user's question.

Rules:
- Answer ONLY using the provided context. If the answer isn't in the context, \
say you couldn't find that information in the uploaded documents.
- Always cite which document(s) you used, e.g. (Source: report.pdf, chunk 3).
- Be concise and direct. Use markdown formatting (lists, bold) only when it \
genuinely improves clarity.
- If multiple documents disagree, point that out explicitly.
"""


def build_context(matches: list[dict]) -> str:
    blocks = []
    for m in matches:
        blocks.append(
            f"[Source: {m['source']}, chunk {m['chunk_index']}, relevance {m['score']:.2f}]\n{m['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def is_ollama_running() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def answer_question(
    question: str,
    chat_history: list[dict] = None,
    n_results: int = 5,
    model: str = DEFAULT_MODEL,
):
    """
    Retrieve relevant chunks and stream an answer from a local Ollama model.
    chat_history: list of {"role": "user"/"assistant", "content": str}
    Yields (partial_text, matches) tuples for streaming display.
    """
    matches = vectorstore.query(question, n_results=n_results)

    if not matches:
        yield "I couldn't find any relevant content. Make sure you've uploaded and processed at least one PDF first.", []
        return

    if not is_ollama_running():
        yield (
            "⚠️ Ollama doesn't seem to be running. Start it with `ollama serve` "
            f"(and make sure you've pulled a model, e.g. `ollama pull {model}`), then try again.",
            [],
        )
        return

    context = build_context(matches)
    history = (chat_history or [])[-6:]
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [
            {
                "role": "user",
                "content": f"Context from documents:\n\n{context}\n\nQuestion: {question}",
            }
        ]
    )

    payload = {"model": model, "messages": messages, "stream": True}

    full_text = ""
    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                full_text += piece
                yield full_text, matches
                if chunk.get("done"):
                    break
                
    except requests.exceptions.RequestException as e:
        yield f"⚠️ Error talking to Ollama: {e}", matches
