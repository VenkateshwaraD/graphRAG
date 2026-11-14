from datetime import datetime

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    title: str
    filename: str
    mime: str
    status: str
    error: str | None = None
    doc_group: str = "default"
    created_at: datetime

    model_config = {"from_attributes": True}


class JobOut(BaseModel):
    id: str
    status: str
    stats: dict | None = None
    error: str | None = None
    created_at: datetime
    finished_at: datetime | None = None

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    conversation_id: str | None = None
    mode: str | None = None  # optional: hybrid|traverse|local|global override
    k: int = Field(default=8, ge=1, le=30)
    as_of: int | None = None  # temporal scope year for traverse evidence


class Citation(BaseModel):
    n: int
    document_id: str | None = None
    document_title: str | None = None
    text_unit_id: str | None = None
    excerpt: str
    hop_path: list[str] | None = None
    edge_directions: list[str] | None = None
    alt_paths: list[list[str]] | None = None
    score: float | None = None
    years: list[int] | None = None
    supported: bool | None = None


class ChatResponse(BaseModel):
    answer: str
    route: str
    route_reason: str | None = None
    warning: str | None = None
    conversation_id: str
    citations: list[Citation]


class SourceOut(BaseModel):
    id: str
    title: str
    filename: str
    text: str


class EntityOut(BaseModel):
    id: str
    name: str
    type: str | None = None
    description: str | None = None
    degree: int | None = None


class RelationshipOut(BaseModel):
    source: str
    target: str
    description: str | None = None
    weight: float | None = None


class GraphStatsOut(BaseModel):
    entities: int
    relationships: int
    communities: int
    text_units: int
    community_reports: int
    indexed: bool


class ReviewIn(BaseModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    description: str = ""
    verdict: str = Field(pattern="^(approved|rejected)$")
    note: str | None = None


class ReviewOut(BaseModel):
    id: str
    rel_hash: str
    source: str
    target: str
    description: str
    verdict: str
    note: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ReviewItemOut(BaseModel):
    """One unreviewed extracted relationship in the review queue.

    Author: venkateshwara doijode
    """

    rel_hash: str
    source: str
    target: str
    description: str | None = None
    weight: float | None = None


class SubgraphNodeOut(BaseModel):
    name: str
    description: str | None = None


class SubgraphEdgeOut(BaseModel):
    source: str
    target: str
    direction: str
    description: str | None = None
    weight: float | None = None


class SubgraphOut(BaseModel):
    nodes: list[SubgraphNodeOut]
    edges: list[SubgraphEdgeOut]
