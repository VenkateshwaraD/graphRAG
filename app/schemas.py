from datetime import datetime

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    title: str
    filename: str
    mime: str
    status: str
    error: str | None = None
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


class Citation(BaseModel):
    n: int
    document_id: str | None = None
    document_title: str | None = None
    text_unit_id: str | None = None
    excerpt: str
    hop_path: list[str] | None = None


class ChatResponse(BaseModel):
    answer: str
    route: str
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
