"""
chunker.py
----------
Splits parsed Markdown documents into overlapping chunks suitable for
embedding + retrieval. Each chunk is saved as its own .md file inside
chunks/<document_name>/chunk_XXXX.md so you can inspect them individually.
"""

import re
from pathlib import Path

CHUNKS_DIR = Path(__file__).parent / "chunks"
CHUNKS_DIR.mkdir(exist_ok=True)

CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 60

# Try to use tiktoken for accurate token counts. If the encoding file can't
# be downloaded (e.g. no internet on first run), fall back to a simple
# whitespace-based approximate tokenizer so the app still works offline.
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def _encode(text: str):
        return _ENC.encode(text)

    def _decode(tokens):
        return _ENC.decode(tokens)

except Exception:

    def _encode(text: str):
        return text.split(" ")

    def _decode(tokens):
        return " ".join(tokens)


def _token_len(text: str) -> int:
    return len(_encode(text))


def _split_by_headers(md_text: str) -> list[str]:
    """First pass split: break on markdown headers so we don't cut mid-section."""
    pattern = r"(?=^#{1,3}\s)"
    sections = re.split(pattern, md_text, flags=re.MULTILINE)
    return [s.strip() for s in sections if s.strip()]


def _split_section_to_chunks(section: str) -> list[str]:
    """Second pass: enforce token-size limit with overlap on long sections."""
    tokens = _encode(section)
    if len(tokens) <= CHUNK_SIZE_TOKENS:
        return [section]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_decode(chunk_tokens))
        if end == len(tokens):
            break
        start = end - CHUNK_OVERLAP_TOKENS
    return chunks


def chunk_markdown(md_text: str) -> list[str]:
    """Full chunking pipeline: header split -> token-size split."""
    sections = _split_by_headers(md_text)
    if not sections:
        sections = [md_text]

    final_chunks = []
    for section in sections:
        final_chunks.extend(_split_section_to_chunks(section))
    return [c for c in final_chunks if c.strip()]


def chunk_and_save(doc_name: str, md_text: str) -> list[dict]:
    """
    Chunk a document's markdown and save each chunk as its own file
    in chunks/<doc_name>/chunk_0001.md

    Returns a list of dicts: {"id", "text", "source", "chunk_index", "path"}
    """
    doc_dir = CHUNKS_DIR / doc_name
    doc_dir.mkdir(exist_ok=True, parents=True)

    chunks = chunk_markdown(md_text)
    chunk_records = []

    for i, chunk_text in enumerate(chunks):
        chunk_path = doc_dir / f"chunk_{i+1:04d}.md"
        chunk_path.write_text(chunk_text, encoding="utf-8")
        chunk_records.append(
            {
                "id": f"{doc_name}::chunk_{i+1:04d}",
                "text": chunk_text,
                "source": doc_name,
                "chunk_index": i + 1,
                "path": str(chunk_path),
            }
        )

    return chunk_records
