"""Load GraphRAG output artifacts (parquet) for retrieval + browsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.ingestion.graph_index import artifacts_dir


@dataclass
class GraphArtifacts:
    """In-memory view of one GraphRAG index build's parquets.

    Any frame missing on disk is an empty DataFrame, so callers can use
    ``.empty`` checks without FileNotFoundError handling.

    Author: venkateshwara doijode
    """

    entities: pd.DataFrame = field(default_factory=pd.DataFrame)
    relationships: pd.DataFrame = field(default_factory=pd.DataFrame)
    text_units: pd.DataFrame = field(default_factory=pd.DataFrame)
    communities: pd.DataFrame = field(default_factory=pd.DataFrame)
    community_reports: pd.DataFrame = field(default_factory=pd.DataFrame)
    documents: pd.DataFrame = field(default_factory=pd.DataFrame)
    out_dir: Path | None = None

    @property
    def indexed(self) -> bool:
        """True when a graph build has produced entities.

        Author: venkateshwara doijode
        """
        return not self.entities.empty

    @property
    def doc_id_map(self) -> dict[str, str]:
        """Map GraphRAG document ids back to our ``Document.id``.

        Documents are staged into the GraphRAG input dir as
        ``{our_doc_id}.txt`` and GraphRAG keeps the filename in
        ``documents.title`` — stripping the extension recovers our id.
        Returns {} before any index build.

        Author: venkateshwara doijode
        """
        if self.documents.empty or "title" not in self.documents.columns:
            return {}
        return {
            str(r["id"]): Path(str(r["title"])).stem
            for _, r in self.documents.iterrows()
        }


def _read(out_dir: Path, name: str) -> pd.DataFrame:
    """Read ``{name}.parquet`` or return an empty DataFrame.

    Author: venkateshwara doijode
    """
    f = out_dir / f"{name}.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame()


_ARTIFACT_NAMES = (
    "entities",
    "relationships",
    "text_units",
    "communities",
    "community_reports",
    "documents",
)


def _signature(out_dir: Path) -> int:
    """Fingerprint of the current build on disk: max parquet mtime_ns.

    Cheap to compute per call, so the cache auto-invalidates when an
    index is rebuilt — including by another process — without callers
    remembering to pass ``refresh``.

    Author: venkateshwara doijode
    """
    newest = 0
    for name in _ARTIFACT_NAMES:
        try:
            newest = max(newest, (out_dir / f"{name}.parquet").stat().st_mtime_ns)
        except FileNotFoundError:
            pass
    return newest


@lru_cache(maxsize=4)
def _cached(out_dir_str: str, signature: int) -> GraphArtifacts:
    """Load all artifact frames once per build signature.

    Author: venkateshwara doijode
    """
    out_dir = Path(out_dir_str)
    return GraphArtifacts(
        entities=_read(out_dir, "entities"),
        relationships=_read(out_dir, "relationships"),
        text_units=_read(out_dir, "text_units"),
        communities=_read(out_dir, "communities"),
        community_reports=_read(out_dir, "community_reports"),
        documents=_read(out_dir, "documents"),
        out_dir=out_dir,
    )


def load_artifacts(refresh: bool = False) -> GraphArtifacts:
    """Return the artifact set for the current on-disk build.

    ``refresh=True`` still forces a reload; it is no longer required
    after reindexing because the mtime signature changes with the build.

    Author: venkateshwara doijode
    """
    if refresh:
        _cached.cache_clear()
    out_dir = artifacts_dir()
    return _cached(str(out_dir), _signature(out_dir))
