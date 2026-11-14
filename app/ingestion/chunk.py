"""Recursive text chunking (~800 chars target with overlap, paragraph-aware).

Char-based sizing keeps this tokenizer-free; 800 chars ≈ 200 tokens, and
`chunk_chars`/`overlap` are tunable per deployment.
"""

from __future__ import annotations

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_text(text: str, chunk_chars: int = 3200, overlap: int = 400) -> list[str]:
    """Split ``text`` into overlapping chunks respecting natural boundaries.

    First recursively splits on paragraph/line/sentence/word separators so
    no chunk exceeds ``chunk_chars``, then prepends a tail of the previous
    chunk to each subsequent one for retrieval continuity. Returns [] for
    empty/whitespace-only input.

    Author: venkateshwara doijode
    """
    text = text.strip()
    if not text:
        return []
    pieces = _split(text, chunk_chars)
    return _merge_with_overlap(pieces, chunk_chars, overlap)


def _split(text: str, max_chars: int, seps: list[str] | None = None) -> list[str]:
    """Recursively split ``text`` into pieces of at most ``max_chars``.

    Tries separators in order (paragraph, newline, sentence, word); a
    piece that still doesn't fit falls through to the next separator, and
    text with no usable separator is hard-sliced. Guarantees every piece
    fits ``max_chars`` while keeping natural boundaries when possible.

    Author: venkateshwara doijode
    """
    seps = seps if seps is not None else _SEPARATORS
    if len(text) <= max_chars:
        return [text]
    sep = seps[0]
    if sep not in text and len(seps) > 1:
        return _split(text, max_chars, seps[1:])
    if sep not in text:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    out: list[str] = []
    buf = ""
    for part in text.split(sep):
        candidate = buf + sep + part if buf else part
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            if len(part) > max_chars:
                out.extend(_split(part, max_chars, seps[1:] or seps))
                buf = ""
            else:
                buf = part
    if buf:
        out.append(buf)
    return out


def _merge_with_overlap(pieces: list[str], max_chars: int, overlap: int) -> list[str]:
    """Prepend ``overlap`` chars of each piece's predecessor to the next.

    Preserves context across chunk boundaries so a sentence split by the
    separator still appears whole inside at least one chunk. First piece
    is kept verbatim; chunks may grow to ``max_chars + overlap``.

    Author: venkateshwara doijode
    """
    if overlap <= 0 or len(pieces) <= 1:
        return pieces
    out: list[str] = []
    for i, piece in enumerate(pieces):
        if i == 0:
            out.append(piece)
            continue
        prev_tail = pieces[i - 1][-overlap:]
        out.append((prev_tail + " " + piece)[: max_chars + overlap])
    return out
