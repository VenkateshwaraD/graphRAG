from app.ingestion.chunk import chunk_text


def test_short_text_single_chunk():
    """Text under the limit comes back as one unchanged chunk.

    Author: venkateshwara doijode
    """
    assert chunk_text("hello world", chunk_chars=100) == ["hello world"]


def test_long_text_splits_on_boundaries():
    """Long text splits on paragraph boundaries without losing content.

    Author: venkateshwara doijode
    """
    text = "\n\n".join(f"Paragraph {i} with some content." for i in range(20))
    chunks = chunk_text(text, chunk_chars=120, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # no paragraph content lost
    joined = " ".join(chunks)
    for i in range(20):
        assert f"Paragraph {i}" in joined


def test_empty_text():
    """Empty/whitespace input produces no chunks.

    Author: venkateshwara doijode
    """
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_overlap_carries_tail():
    """Each chunk after the first carries the tail of its predecessor.

    Author: venkateshwara doijode
    """
    pieces = [f"Section {i} " + "x" * 90 for i in range(4)]
    text = "\n\n".join(pieces)
    chunks = chunk_text(text, chunk_chars=110, overlap=30)
    assert len(chunks) > 1
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        assert prev[-30:-1].strip()[:10] in cur or cur.startswith(prev[-30:][:10])
