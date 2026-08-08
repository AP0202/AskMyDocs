"""
Ingestion: load raw documents from disk and split them into retrieval-sized
chunks.

Documents in this demo are plain .txt files structured with a small header
(TITLE / DOCUMENT TYPE / ...) followed by SECTION: blocks. We chunk on
section boundaries (falling back to a word-count split for long sections)
because sections are natural, citation-friendly units in clinical documents.

In a production system this module would also handle PDF/HL7/FHIR ingestion,
OCR for scanned documents, and de-identification -- see README for notes.
"""

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.config import DATA_DIR, CHUNK_MAX_WORDS

SECTION_RE = re.compile(r"^SECTION:\s*(.+)$", re.MULTILINE)
HEADER_FIELD_RE = re.compile(r"^([A-Z][A-Z /_-]+):\s*(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    section: str
    text: str
    order: int


@dataclass
class Document:
    doc_id: str
    title: str
    doc_type: str
    date: Optional[str]
    content: str
    chunks: List[Chunk] = field(default_factory=list)


def _parse_header(raw: str) -> dict:
    """Pull the simple KEY: value header lines at the top of the file."""
    fields = {}
    for line in raw.splitlines():
        if line.startswith("SECTION:"):
            break
        m = HEADER_FIELD_RE.match(line.strip())
        if m:
            fields[m.group(1).strip()] = m.group(2).strip()
    return fields


def _split_into_sections(raw: str):
    """Yield (section_title, section_body) tuples."""
    matches = list(SECTION_RE.finditer(raw))
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        yield title, body


def _word_chunks(text: str, max_words: int) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    out = []
    for i in range(0, len(words), max_words):
        out.append(" ".join(words[i : i + max_words]))
    return out


def load_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    return load_document_from_text(doc_id=path.stem, raw=raw)


def load_document_from_text(
    doc_id: str,
    raw: str,
    *,
    title: Optional[str] = None,
    doc_type: Optional[str] = None,
    date: Optional[str] = None,
) -> Document:
    header = _parse_header(raw)
    resolved_title = title or header.get("TITLE", doc_id)
    resolved_doc_type = doc_type or header.get("DOCUMENT TYPE", "Document")
    resolved_date = date or (
        header.get("DATE COLLECTED")
        or header.get("DATE OF EXAM")
        or header.get("DATE PRESCRIBED")
        or header.get("DISCHARGE DATE")
    )

    doc = Document(
        doc_id=doc_id,
        title=resolved_title,
        doc_type=resolved_doc_type,
        date=resolved_date,
        content=raw,
    )

    order = 0
    for section_title, body in _split_into_sections(raw):
        if not body:
            continue
        for piece in _word_chunks(body, CHUNK_MAX_WORDS):
            chunk = Chunk(
                chunk_id=f"{doc_id}::{order}",
                doc_id=doc_id,
                doc_title=resolved_title,
                doc_type=resolved_doc_type,
                section=section_title,
                text=piece,
                order=order,
            )
            doc.chunks.append(chunk)
            order += 1

    # Uploaded plain-text files may not contain SECTION markers.
    if not doc.chunks:
        cleaned = " ".join(raw.split())
        for piece in _word_chunks(cleaned, CHUNK_MAX_WORDS):
            if not piece.strip():
                continue
            chunk = Chunk(
                chunk_id=f"{doc_id}::{order}",
                doc_id=doc_id,
                doc_title=resolved_title,
                doc_type=resolved_doc_type,
                section="Document Body",
                text=piece,
                order=order,
            )
            doc.chunks.append(chunk)
            order += 1

    return doc


def load_all_documents(data_dir: Path = DATA_DIR) -> List[Document]:
    docs = []
    for path in sorted(data_dir.glob("*.txt")):
        docs.append(load_document(path))
    return docs


def new_id() -> str:
    return str(uuid.uuid4())
