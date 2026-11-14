"""Load GraphRAG output artifacts (parquet) for retrieval + browsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

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
            str(row["id"]): Path(str(row["title"])).stem
            for _, row in self.documents.iterrows()
        }


def _read(out_dir: Path, name: str) -> pd.DataFrame:
    """Read ``{name}.parquet`` or return an empty DataFrame.

    Author: venkateshwara doijode
    """
    parquet_path = out_dir / f"{name}.parquet"
    return pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()


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


def _id_set(df: pd.DataFrame, col: str) -> set[str]:
    """Non-empty string ids from one column ([] when absent/empty).

    Author: venkateshwara doijode
    """
    if df.empty or col not in df.columns:
        return set()
    return {str(val) for val in df[col]}


def _all_allowed(ids: Any, allowed: set[str]) -> bool:
    """True only when every supporting id is allowed — an empty or
    missing provenance list can never be verified, so it fails closed.

    Author: venkateshwara doijode
    """
    if ids is None:
        return False
    try:
        vals = [str(item) for item in ids]
    except TypeError:
        return False
    return bool(vals) and all(item in allowed for item in vals)


def _intersect(ids: Any, allowed: set[str]) -> list[str]:
    """Allowed subset of an id list, preserving order.

    Author: venkateshwara doijode
    """
    if ids is None:
        return []
    try:
        return [item for item in ids if str(item) in allowed]
    except TypeError:
        return []


def scope_artifacts(arts: GraphArtifacts, doc_ids: set[str]) -> GraphArtifacts:
    """Caller-scoped view of the graph artifacts — strict provenance.

    An aggregate row is exposed only when its *complete* supporting
    provenance is authorized; otherwise it is dropped, never trimmed:

    - ``documents``/``text_units``: kept iff their doc id is allowed.
    - ``entities``/``relationships``: kept iff every ``text_unit_ids``
      entry survives (empty/missing provenance can never be verified).
    - ``communities``: kept iff every member (entity, relationship,
      text unit) is clean — one disallowed member drops the whole
      community and prunes dangling parent/children references.
    - ``community_reports``: kept iff the referenced community
      survived fully clean.

    ``doc_ids`` is a concrete set of *our* Document ids; an empty set
    yields empty frames (fail closed).

    Author: venkateshwara doijode
    """
    doc_map = arts.doc_id_map  # graphrag doc id -> our doc id
    allowed_gdocs = {
        graphrag_id
        for graphrag_id, our_id in doc_map.items()
        if our_id in doc_ids
    }

    documents = arts.documents
    if not documents.empty and "id" in documents.columns:
        documents = documents[
            documents["id"].astype(str).isin(allowed_gdocs)
        ].copy()

    text_units = arts.text_units
    if not text_units.empty and "document_id" in text_units.columns:
        text_units = text_units[
            text_units["document_id"].astype(str).isin(allowed_gdocs)
        ].copy()
    allowed_tu = _id_set(text_units, "id")

    entities = arts.entities
    if "text_unit_ids" in entities.columns:
        entities = entities[
            entities["text_unit_ids"].map(
                lambda ids: _all_allowed(ids, allowed_tu)
            )
        ].copy()
    else:
        entities = entities.iloc[0:0].copy()
    allowed_ent = _id_set(entities, "id")

    relationships = arts.relationships
    if "text_unit_ids" in relationships.columns:
        relationships = relationships[
            relationships["text_unit_ids"].map(
                lambda ids: _all_allowed(ids, allowed_tu)
            )
        ].copy()
    else:
        relationships = relationships.iloc[0:0].copy()
    allowed_rel = _id_set(relationships, "id")

    communities = arts.communities
    if not communities.empty:

        def _clean_community(row) -> bool:
            return (
                _all_allowed(row.get("entity_ids"), allowed_ent)
                and _all_allowed(row.get("relationship_ids"), allowed_rel)
                and _all_allowed(row.get("text_unit_ids"), allowed_tu)
            )

        communities = communities[
            communities.apply(_clean_community, axis=1)
        ].copy()
        clean_keys = set(
            zip(
                communities["community"].astype(str),
                communities["level"].astype(str),
                strict=False,
            )
        ) if "community" in communities.columns else set()
        # prune dangling parent/children references to dropped rows
        if "children" in communities.columns:
            communities["children"] = communities["children"].map(
                lambda kids: _intersect(kids, {key[0] for key in clean_keys})
                if isinstance(kids, list | tuple)
                else kids
            )

    community_reports = arts.community_reports
    if not community_reports.empty and not communities.empty:
        community_reports = community_reports[
            community_reports.apply(
                lambda row: (str(row.get("community")), str(row.get("level")))
                in clean_keys,
                axis=1,
            )
        ].copy()
    else:
        community_reports = community_reports.iloc[0:0].copy()

    return GraphArtifacts(
        entities=entities,
        relationships=relationships,
        text_units=text_units,
        communities=communities,
        community_reports=community_reports,
        documents=documents,
        out_dir=arts.out_dir,
    )


def stale_doc_ids(arts: GraphArtifacts, live_doc_ids: set[str]) -> set[str]:
    """Our-document ids present in the index but not in ``live_doc_ids``
    — documents deleted (or never committed) since the last build.

    Author: venkateshwara doijode
    """
    return set(arts.doc_id_map.values()) - live_doc_ids
