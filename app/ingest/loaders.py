from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from app.core.logging import logger


SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".docx", ".html", ".htm"}


def _loader_for(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path))
    if suffix in (".md", ".markdown"):
        return UnstructuredMarkdownLoader(str(path))
    if suffix == ".docx":
        return Docx2txtLoader(str(path))
    if suffix in (".html", ".htm"):
        return BSHTMLLoader(str(path))
    return TextLoader(str(path), encoding="utf-8")


def load_path(path: str | Path) -> List[Document]:
    """Load a single file or every supported file under a directory."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    paths: Iterable[Path] = [p] if p.is_file() else (
        q for q in p.rglob("*") if q.is_file() and q.suffix.lower() in SUPPORTED_SUFFIXES
    )
    docs: List[Document] = []
    for fp in paths:
        try:
            loaded = _loader_for(fp).load()
            for d in loaded:
                d.metadata.setdefault("source", str(fp))
                d.metadata.setdefault("filename", fp.name)
            docs.extend(loaded)
            logger.info("Loaded {} chunks from {}", len(loaded), fp)
        except Exception as e:
            logger.warning("Failed to load {}: {}", fp, e)
    return docs
