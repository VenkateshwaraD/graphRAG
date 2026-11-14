# Project Requirements Document

## University Knowledge Assistant -- Permission-Aware Multi-Tenant Agentic RAG

**Document type:** Project requirements and initial solution design\
**Status:** Initial draft\
**Recommended Python version:** Python 3.11+

------------------------------------------------------------------------

## 1. Project Overview

Build a University Knowledge Assistant that allows students, faculty
members, and university administrators to ask questions about university
information and receive answers grounded in authorized institutional
documents.

The platform is designed around a multi-tenant model. Each university is
treated as a tenant, and its users and documents must remain isolated
from those of other universities. Within a university, access to
documents is controlled by user roles, departments, courses, enrollment,
document classifications, and explicit permissions.

The application will use an Agentic Retrieval-Augmented Generation (RAG)
architecture. It will retrieve relevant documents, enforce access
controls, and provide answers with source references.

## 2. Objectives

1.  Build a conversational knowledge assistant for university
    information.
2.  Support multiple universities on a shared application platform.
3.  Enforce tenant isolation so one university cannot access another
    university's data.
4.  Implement role-based access control (RBAC) and attribute-based
    access control (ABAC).
5.  Restrict document retrieval according to user identity, role,
    department, course enrollment, and document permissions.
6.  Support document ingestion, processing, indexing, retrieval, and
    grounded answer generation.
7.  Provide source references for generated answers.
8.  Maintain audit records for authentication, authorization decisions,
    document access, and user queries.
9.  Evaluate answer quality, retrieval quality, security, and
    application performance.
10. Keep the RAG backend independent of the user-interface technology so
    the UI can be replaced later.

## 3. Scope

### 3.1 Initial in-scope capabilities

-   University/tenant registration and management.
-   User management for students, faculty, and university
    administrators.
-   Authentication and authorization.
-   Role and permission management.
-   Department and course metadata.
-   Document upload and ingestion.
-   Document metadata and access-control metadata.
-   Text extraction, chunking, and embedding generation.
-   Vector and keyword indexing.
-   Permission-aware hybrid retrieval.
-   LangGraph-based RAG workflow.
-   LLM-based answer generation.
-   Source citations or document references.
-   Audit logging and observability.
-   Automated functional, security, and evaluation tests.

### 3.2 Initial out-of-scope capabilities

These may be considered in later releases:

-   Native Android and iOS applications.
-   A React or Next.js frontend.
-   Student information system (SIS) integrations.
-   Learning management system (LMS) integrations.
-   Automated grading or exam evaluation.
-   Public production deployment for real student records before
    security review.
-   Advanced GraphRAG and semantic caching.

## 4. Primary Use Cases

### UC5.1 -- Student Knowledge Assistant

Students can: - Ask questions about their enrolled courses. - Search
authorized lecture notes, syllabus documents, and learning materials. -
Find academic calendars and publicly available university policies. -
Receive answers with references to permitted source documents.

Students must not: - Access documents for courses they are not
authorized to view. - Access faculty-only or administrator-only
documents. - Access restricted examination papers or answer keys.

### UC5.2 -- Faculty Knowledge Assistant

Faculty members can: - Search materials for courses assigned to them. -
Retrieve permitted teaching resources and academic policies. - Search
documents shared with their department or role. - Access other materials
only when explicitly authorized.

### UC5.3 -- University Administration Assistant

Authorized administrators can: - Search university policies and
administrative procedures. - Manage users, roles, departments, and
document permissions according to their administrative scope. - Access
administrative documents permitted by their role. - Review relevant
audit records.

### UC5.4 -- Examination Document Assistant

Authorized examination staff and faculty can: - Search
examination-related documents within their assigned scope. - Access
answer keys or restricted examination material only when explicitly
permitted.

Students and other unauthorized users must not receive restricted
examination content through search results, generated answers,
citations, or cached responses.

## 5. User Roles

The initial role model is:

  -----------------------------------------------------------------------
  Role                                Example permissions
  ----------------------------------- -----------------------------------
  Platform Super Admin                Manage university tenants and
                                      platform-level configuration

  University Admin                    Manage users and documents within
                                      their university

  Head of Department (HOD)            Access permitted department-level
                                      information

  Faculty                             Access assigned courses and
                                      explicitly shared faculty materials

  Student                             Access enrolled-course materials
                                      and permitted university
                                      information

  Guest                               Access only documents explicitly
                                      made public or shared with them
  -----------------------------------------------------------------------

Roles are starting points, not the only authorization condition. Actual
access must also consider tenant, department, course, enrollment,
document classification, ownership, and explicit grants.

## 6. Functional Requirements

### FR-01: Tenant Management

-   The system shall support multiple university tenants.
-   Every tenant-owned user, document, chunk, and audit event shall be
    associated with a tenant.
-   Tenant identity shall be derived from a trusted authenticated
    identity or server-side session, not blindly accepted from
    user-supplied query parameters.

### FR-02: Authentication

-   Users shall authenticate before accessing protected university
    information.
-   The system shall validate identity tokens and token expiry.
-   The system shall support role and tenant claims from the identity
    provider.

### FR-03: Authorization

-   The system shall support RBAC and ABAC.
-   Authorization shall be checked server-side.
-   Access decisions shall consider tenant, user, role, department,
    course, enrollment, document classification, and explicit
    permissions where applicable.
-   A denied document shall not be passed to the LLM as context.

### FR-04: Document Ingestion

-   Authorized users shall be able to upload supported documents.
-   The ingestion pipeline shall extract text, split content into
    chunks, generate embeddings, and index the chunks.
-   Each document and chunk shall carry required tenant and
    access-control metadata.
-   Ingestion shall validate document ownership and permissions.

### FR-05: Retrieval

-   The system shall support vector retrieval and keyword retrieval.
-   The system shall apply tenant and permission filters during
    retrieval.
-   The system shall re-check authorization before retrieved content is
    used to construct LLM context.
-   Retrieval shall not rely solely on prompt instructions to enforce
    security.

### FR-06: Answer Generation

-   The system shall generate answers using only authorized retrieved
    context and approved general knowledge behavior.
-   The system shall provide source references for document-grounded
    claims.
-   The system should state when available authorized documents do not
    contain enough information.

### FR-07: Audit and Observability

-   The system shall record relevant user, tenant, document, and
    authorization activity.
-   Audit records shall not expose sensitive document contents
    unnecessarily.
-   Operational traces shall support debugging and performance analysis
    while respecting privacy controls.

### FR-08: Administration

-   Authorized administrators shall manage users, roles, departments,
    courses, documents, and permissions within their scope.
-   Administrative actions shall be auditable.

## 7. Non-Functional Requirements

### Security

-   Enforce tenant isolation across APIs, database queries, indexes,
    caches, and logs.
-   Use least-privilege access.
-   Validate authorization on every protected request.
-   Protect secrets and credentials using environment configuration or a
    secrets manager.
-   Prevent insecure direct object references.
-   Re-check access when a user opens a source citation.
-   Ensure revoked permissions take effect for subsequent requests.
-   Do not treat frontend hiding or LLM prompt instructions as security
    controls.

### Reliability

-   Handle unavailable LLM, database, and search services gracefully.
-   Use timeouts and bounded retries for external services.
-   Avoid returning partial or unauthorized retrieval results as
    successful answers.

### Performance

-   Measure API latency, retrieval latency, LLM latency, time to first
    token, throughput, and error rate.
-   Establish performance targets after creating a baseline with
    representative test data.

### Maintainability

-   Separate API, identity, authorization, ingestion, retrieval,
    orchestration, and evaluation modules.
-   Use configuration rather than hard-coded tenant or provider
    settings.
-   Include automated tests and architecture documentation.

### Scalability

-   Keep application services stateless where practical.
-   Design tenant data access so the platform can later support
    dedicated infrastructure for selected universities if required.

## 8. Proposed High-Level Architecture

``` text
                 Students / Faculty / Admins
                              |
                              v
                    Streamlit User Interface
                              |
                              v
                         FastAPI
                              |
                 Authentication / Identity
                       (Keycloak)
                              |
                              v
                  Authorization Service
                     (RBAC + ABAC)
                              |
                              v
                    LangGraph RAG Workflow
                              |
               +--------------+--------------+
               |                             |
               v                             v
       Permission-Aware              Query / Workflow
       Hybrid Retrieval                 Management
               |
       +-------+--------+
       |                |
       v                v
  PostgreSQL         OpenSearch
  + pgvector         Keyword Search
       |
       v
 Authorized Chunks and Source Metadata
       |
       v
      LLM
       |
       v
 Answer with Authorized Source References
```

### Document Ingestion Flow

``` text
Authorized Document Upload
          |
          v
File Validation and Tenant Validation
          |
          v
Text Extraction and Parsing
          |
          v
Chunking and Metadata Enrichment
          |
          v
Embedding Generation
          |
          v
PostgreSQL / pgvector + OpenSearch Indexing
```

### Important security design rule

Authorization must be enforced before retrieved content reaches the LLM.
Tenant and permission filters must be applied in retrieval, followed by
an authorization check before context construction. The LLM must never
be the component responsible for deciding whether a user is allowed to
see a document.

## 9. Proposed Technology Stack

  -----------------------------------------------------------------------
  Layer                   Selected technology     Purpose
  ----------------------- ----------------------- -----------------------
  Programming language    Python 3.11+            Main development
                                                  language

  Frontend                Streamlit               Python-based user
                                                  interface

  Backend API             FastAPI                 REST APIs and service
                                                  boundary

  Agent orchestration     LangGraph               RAG workflow and agent
                                                  orchestration

  LLM providers           OpenAI and/or Anthropic Answer generation
                          Claude                  

  Embeddings              Sentence Transformers   Text embeddings
                          initially               

  Primary database        PostgreSQL              Tenant, user, course,
                                                  document, and metadata
                                                  storage

  Vector search           pgvector                Vector similarity
                                                  retrieval

  Keyword search          OpenSearch              Keyword and text
                                                  retrieval

  Identity provider       Keycloak                Authentication and
                                                  identity management

  Authorization           Custom RBAC + ABAC      Fine-grained access
                          service                 decisions

  Evaluation              RAGAS                   RAG quality evaluation

  Observability           Langfuse                LLM traces and
                                                  monitoring

  Testing                 Pytest and HTTPX        Unit, integration, and
                                                  API tests

  Code quality            Ruff                    Linting and formatting

  Containers              Docker / Docker Compose Local development and
                                                  service packaging

  Deployment              Kubernetes (later       Production
                          phase)                  orchestration when
                                                  needed
  -----------------------------------------------------------------------

These are proposed project choices. Exact versions and provider-specific
integrations will be pinned and verified during implementation.

## 10. Core Data Entities

Initial entities to design:

-   Tenant / University
-   User
-   Role
-   Permission
-   UserRole
-   Department
-   Course
-   CourseEnrollment
-   Document
-   DocumentChunk
-   DocumentPermission
-   AuditEvent
-   Conversation
-   Message

Every tenant-owned entity must have a tenant association. Database
constraints and server-side authorization should reinforce tenant
boundaries.

## 11. API Areas

Initial API groups:

  API area         Example responsibilities
  ---------------- ----------------------------------------------------
  Authentication   Login/session integration and identity validation
  Tenants          University administration
  Users            User lifecycle and role assignment
  Departments      Department management
  Courses          Course and enrollment management
  Documents        Upload, list, metadata, and permission management
  Ingestion        Process and index documents
  Chat             Submit questions and receive grounded answers
  Sources          Validate access before opening document references
  Audit            Authorized audit search and review
  Health           Service health and readiness checks

Exact routes and request/response schemas will be defined in the API
design stage.

## 12. Security and Acceptance Test Scenarios

  -----------------------------------------------------------------------
  ID                      Scenario                Expected result
  ----------------------- ----------------------- -----------------------
  SEC-01                  Student asks about an   Authorized course
                          enrolled course         information can be
                                                  retrieved

  SEC-02                  Student asks about an   Restricted content is
                          unassigned course       not retrieved

  SEC-03                  Student asks for an     Access is denied unless
                          examination answer key  explicitly authorized

  SEC-04                  Faculty searches an     Permitted course
                          assigned course         content is retrieved

  SEC-05                  Faculty requests        Access is denied unless
                          another department's    explicitly granted
                          restricted file         

  SEC-06                  University A user       Access is denied
                          attempts to access      
                          University B data       

  SEC-07                  User changes a tenant   Server-side identity
                          identifier in a request and authorization
                                                  prevent access

  SEC-08                  User's document         Subsequent retrieval
                          permission is revoked   and source access are
                                                  denied

  SEC-09                  Unauthorized content    It is filtered before
                          appears among search    LLM context
                          candidates              construction

  SEC-10                  A cached response is    No protected content is
                          requested by another    disclosed
                          unauthorized user       

  SEC-11                  User opens a previously Current authorization
                          issued source citation  is checked again

  SEC-12                  Administrative          Action is recorded in
                          permission changes      the audit log
  -----------------------------------------------------------------------

Security tests should include direct API calls, not only UI tests.

## 13. Evaluation and Benchmarking

The project shall be evaluated in four areas:

### RAG quality

-   Faithfulness
-   Answer relevancy
-   Context precision
-   Context recall

### Security

-   Cross-tenant isolation test pass rate
-   Unauthorized document retrieval count
-   Unauthorized context inclusion count
-   Permission revocation effectiveness
-   Citation access-control test pass rate

### Performance

-   API latency
-   Retrieval latency
-   Time to first token (TTFT)
-   Tokens per second / time per output token
-   P50, P95, and P99 latency
-   Throughput and error rate

### Operations

-   Ingestion success and failure rates
-   LLM and retrieval traces
-   Token usage and model cost, where available
-   Audit event completeness

## 14. Implementation Roadmap

  -----------------------------------------------------------------------
  Phase                               Deliverable
  ----------------------------------- -----------------------------------
  1                                   Requirements, architecture,
                                      repository structure, and local
                                      development environment

  2                                   Tenant, user, department, course,
                                      and enrollment data model

  3                                   Keycloak authentication and
                                      RBAC/ABAC authorization

  4                                   Document upload, parsing, chunking,
                                      and ingestion

  5                                   PostgreSQL/pgvector and OpenSearch
                                      retrieval

  6                                   LangGraph RAG workflow and grounded
                                      answers

  7                                   Streamlit student, faculty, and
                                      admin interfaces

  8                                   Audit logging, Langfuse tracing,
                                      and error handling

  9                                   Security, functional, and
                                      integration test suite

  10                                  RAGAS evaluation and performance
                                      benchmarking

  11                                  Docker-based deployment and
                                      operational documentation

  12                                  Production hardening and optional
                                      Kubernetes deployment
  -----------------------------------------------------------------------

## 15. Assumptions and Decisions to Confirm

The following are proposed defaults and should be confirmed before
implementation:

1.  The initial target is a multi-university platform.
2.  Initial user groups are students, faculty, and university
    administrators.
3.  The first deployment may use shared application infrastructure with
    strict tenant isolation.
4.  PostgreSQL with pgvector is the initial vector store.
5.  OpenSearch is used for keyword retrieval.
6.  Streamlit is the initial UI; the backend remains UI-independent.
7.  Real student or sensitive institutional data will not be used in
    development without appropriate authorization and safeguards.
8.  Production security, privacy, retention, and institutional
    compliance requirements must be reviewed before deployment.

------------------------------------------------------------------------

## 16. Definition of Done

The initial project is considered complete when:

-   Multiple university tenants can be configured.
-   Students, faculty, and administrators can authenticate.
-   Role and attribute-based access decisions are enforced server-side.
-   Documents can be ingested with tenant and permission metadata.
-   Hybrid retrieval returns only authorized content.
-   The LLM receives only authorized context.
-   Answers include accessible source references.
-   Cross-tenant and role-based security tests pass.
-   Audit and observability are available.
-   RAG quality and performance evaluation reports can be generated.
-   Setup, configuration, architecture, API, and testing documentation
    are provided.
