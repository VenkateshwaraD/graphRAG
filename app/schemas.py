from datetime import datetime

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    tenant_id: str | None = None
    owner_id: str | None = None
    department_id: str | None = None
    course_id: str | None = None
    classification: str = "internal"
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
    requested_route: str | None = None  # set when a scoped caller was rerouted
    route_reason: str | None = None
    warning: str | None = None
    conversation_id: str
    citations: list[Citation]


class SourceOut(BaseModel):
    id: str
    title: str
    filename: str
    text: str


# --- multi-tenant administration -------------------------------------------


class TenantIn(BaseModel):
    slug: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=256)


class TenantOut(BaseModel):
    id: str
    slug: str
    name: str
    created_at: datetime

    model_config = {"from_attributes": True}


class UserIn(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    display_name: str = ""
    role: str = "student"
    tenant_id: str | None = None  # required unless actor is platform_admin w/ target tenant
    department_id: str | None = None


class UserPatch(BaseModel):
    role: str | None = None
    department_id: str | None = None
    is_active: bool | None = None


class UserOut(BaseModel):
    id: str
    tenant_id: str | None = None
    email: str
    display_name: str
    role: str
    department_id: str | None = None
    is_active: bool
    key_prefix: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class UserCreatedOut(UserOut):
    """User row plus the raw API key — returned exactly once."""

    api_key: str


class RotateKeyOut(BaseModel):
    api_key: str
    key_prefix: str


class DepartmentIn(BaseModel):
    name: str = Field(min_length=1, max_length=256)


class DepartmentOut(BaseModel):
    id: str
    tenant_id: str
    name: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CourseIn(BaseModel):
    code: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=512)
    department_id: str


class CourseOut(BaseModel):
    id: str
    tenant_id: str
    department_id: str
    code: str
    title: str
    created_at: datetime

    model_config = {"from_attributes": True}


class EnrollmentIn(BaseModel):
    user_id: str
    course_id: str
    kind: str = "student"  # student (enrollment) | instructor


class EnrollmentOut(BaseModel):
    id: str
    user_id: str
    course_id: str
    created_at: datetime

    model_config = {"from_attributes": True}


class GrantIn(BaseModel):
    grantee_type: str  # user | role | department | course
    grantee_id: str


class GrantOut(BaseModel):
    id: str
    document_id: str
    grantee_type: str
    grantee_id: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AuditEventOut(BaseModel):
    id: str
    tenant_id: str | None = None
    user_id: str | None = None
    action: str
    resource_type: str | None = None
    resource_id: str | None = None
    decision: str | None = None
    detail: dict | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


# --- graph ------------------------------------------------------------------


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
    stale: bool = False
    stale_documents: int = 0


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
