"""
parser.py
---------
Handles parsing of one or more PDF files into Markdown using LlamaParse.
Each parsed PDF is saved as a .md file inside `parsed_markdown/`.
"""

import os
import asyncio
from pathlib import Path
from llama_parse import LlamaParse

PARSED_DIR = Path(__file__).parent / "parsed_markdown"
PARSED_DIR.mkdir(exist_ok=True)


def _safe_name(filename: str) -> str:
    """Turn 'My Report.pdf' -> 'My_Report'."""
    stem = Path(filename).stem
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem)


async def _parse_single_pdf(parser: LlamaParse, file_path: str) -> str:
    """Parse one PDF and return its combined markdown content."""
    documents = await parser.aload_data(file_path)
    # LlamaParse can return multiple "pages" as separate Document objects
    md_parts = [doc.text for doc in documents]
    return "\n\n---\n\n".join(md_parts)


async def parse_pdfs_async(file_paths: list[str], api_key: str) -> dict[str, str]:
    """
    Parse multiple PDFs concurrently with LlamaParse and save the markdown
    output for each into parsed_markdown/<filename>.md

    Returns: dict mapping original filename -> path of saved markdown file
    """
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",  # "markdown" or "text"
        verbose=False,
        num_workers=4,
    )

    results = {}
    tasks = {fp: _parse_single_pdf(parser, fp) for fp in file_paths}

    for file_path, task in tasks.items():
        md_text = await task
        filename = os.path.basename(file_path)
        out_path = PARSED_DIR / f"{_safe_name(filename)}.md"
        out_path.write_text(md_text, encoding="utf-8")
        results[filename] = str(out_path)

    return results


def parse_pdfs(file_paths: list[str], api_key: str) -> dict[str, str]:
    """Synchronous wrapper around parse_pdfs_async (for use in Streamlit)."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(parse_pdfs_async(file_paths, api_key))
