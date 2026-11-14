"""Text extraction for pdf / docx / txt / md uploads."""

from __future__ import annotations

from pathlib import Path


def extract_text(path: Path) -> str:
    """Extract plain text from a stored upload.

    Dispatches on file suffix: ``.pdf`` via pypdf page extraction,
    ``.docx`` via python-docx paragraph text, and ``.txt``/``.md``/
    ``.markdown`` read directly as UTF-8 with undecodable bytes replaced.
    Raises ``ValueError`` for unsupported suffixes — callers surface that
    as a failed document status.

    Author: venkateshwara doijode
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf(path)
    if suffix == ".docx":
        return _docx(path)
    if suffix in (".txt", ".md", ".markdown"):
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"unsupported file type: {suffix}")


def _pdf(path: Path) -> str:
    """Pull text out of a PDF via pypdf, one block per page.

    Pages with no extractable text (scanned images) contribute "".

    Author: venkateshwara doijode
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _docx(path: Path) -> str:
    """Pull paragraph text out of a DOCX via python-docx.

    Empty paragraphs are skipped so blank spacing doesn't inflate chunks.

    Author: venkateshwara doijode
    """
    import docx

    doc = docx.Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
