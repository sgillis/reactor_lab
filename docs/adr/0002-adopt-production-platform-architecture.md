# ADR 0002: Adopt a production-oriented web and compute platform

- Status: Accepted
- Date: 2026-11-01

## Context

The proof of concept provides reproducible simulation, surrogate training, evaluation, and optimization through a local CLI. A command executes synchronously in the caller's process and writes artifacts to a local directory. This is appropriate for a single user and makes the complete scientific workflow easy to inspect.

That execution model does not provide the boundaries needed by a multi-user or long-running service:

- Browser clients cannot invoke the packaged workflows or inspect their results through a stable remote contract.
- Long computations would occupy a web-request process and could be lost if that process restarted.
- Local artifact directories do not provide shared storage across independent containers or hosts.
- Files alone do not provide efficient run discovery, status transitions, relational queries, or concurrent metadata updates.
- A single process cannot scale API traffic and compute demand independently.
- Inspecting numerical results through saved files and static images does not provide an interactive operational interface.

The platform should introduce production-shaped boundaries without splitting the scientific code into microservices or adopting cloud orchestration before it is needed. Local development must remain reproducible and usable on one machine.

## Decision

Retain `reactor_lab` as the authoritative physics and machine-learning package. Place a modular web platform around it consisting of shared application services, a FastAPI control plane, asynchronous Dramatiq workers, Redis, PostgreSQL, S3-compatible artifact storage, and a React/TypeScript frontend.

Run these components locally with Docker Compose. Use the same container image and locked Python environment for the API and workers, with different process entry points. Initially deploy a single worker and local infrastructure containers; scale later by replacing local services with managed equivalents and adding stateless API or worker replicas.

All simulation, training, and optimization submissions will create a run record and return `202 Accepted` with a run identifier. Workers will execute the computation and publish status and results. Runs will use the lifecycle `queued`, `running`, `succeeded`, and `failed`. The frontend will initially poll for changes rather than maintain a persistent event connection.

## Component decisions and rationale

### Shared application services

**Proof-of-concept limitation:** The CLI currently combines argument parsing, workflow orchestration, and artifact writing. Calling it in a subprocess from a web service would make structured errors, progress reporting, resource control, and result persistence harder to manage.

**Decision and production role:** Extract reusable workflow orchestration into application services called by both CLI commands and worker handlers. Continue using the domain functions in `reactor_lab` directly rather than invoking the CLI as a subprocess.

This creates one implementation of each scientific workflow while allowing different delivery mechanisms. It also prevents the CLI and web platform from developing divergent validation, artifact, and error-handling behavior.

### FastAPI control plane

**Proof-of-concept limitation:** A local synchronous CLI has no remote interface, request validation, service-health contract, or machine-readable API description.

**Decision and production role:** Use FastAPI to expose versioned REST resources for runs, models, results, and artifact downloads. Pydantic schemas will define the external boundary and translate validated requests into the existing typed domain configurations.

FastAPI provides request validation and OpenAPI generation while remaining a thin adapter around the application services. Keeping the API stateless allows multiple replicas to serve traffic behind a load balancer. API processes will enqueue substantial work rather than execute it in request handlers or FastAPI background tasks.

### Asynchronous Dramatiq workers

**Proof-of-concept limitation:** CLI work inherits the lifetime and resources of the invoking terminal process. Moving that work directly into FastAPI would make web availability depend on JAX compilation, simulation time, and training duration.

**Decision and production role:** Use dedicated Dramatiq worker processes for simulation, training, and optimization. Workers will load the same application services and domain package, update run state, and store resulting artifacts.

Separating execution from request handling isolates resource-intensive computation and permits independent worker scaling. It also creates a path to separate CPU and GPU queues without changing the frontend contract. The API and worker will share an image so dependency and checkpoint behavior remain consistent.

### Redis queue broker

**Proof-of-concept limitation:** Direct function calls provide no buffered handoff between a submitter and an independently scaled executor.

**Decision and production role:** Use Redis as the Dramatiq broker. The API will publish job messages containing run identifiers; workers will load authoritative run configuration from PostgreSQL rather than relying on large artifact payloads in queue messages.

Redis decouples submission from execution, absorbs bursts of work, and provides the broker needed to distribute jobs across workers. It is an infrastructure dependency with explicit durability and delivery semantics, not the system of record for runs or results. Job handlers must tolerate redelivery by using run-scoped artifact keys and idempotent state/result handling.

### PostgreSQL metadata store and Alembic migrations

**Proof-of-concept limitation:** Directory traversal and JSON files cannot efficiently support concurrent run status updates, filtered run lists, model lineage, or relational integrity.

**Decision and production role:** Use PostgreSQL as the system of record for run configuration, type, status, timestamps, errors, summary metrics, model records, lineage, and artifact metadata. Store artifact payloads outside the database. Use JSONB for configuration and metric documents whose schema belongs to the scientific workflow, with relational columns for fields used in status and lineage queries.

Use Alembic for explicit, versioned schema migrations. PostgreSQL provides transactions and concurrent access for multiple API and worker processes, and a managed PostgreSQL service can replace the local container without changing the application boundary.

### MinIO and S3-compatible object storage

**Proof-of-concept limitation:** Local output directories tie artifacts to one filesystem and cannot be shared reliably as API and worker processes scale across hosts.

**Decision and production role:** Store datasets, trajectories, Equinox checkpoints, metrics documents, history, and plots in S3-compatible object storage. Run MinIO locally and access storage through a small application storage interface. PostgreSQL records will contain object keys and metadata, not binary payloads. Downloads will use authorized application endpoints or time-limited presigned URLs.

Object storage separates large immutable payloads from transactional metadata and lets independent workers and API replicas access the same artifacts. MinIO preserves the S3 interaction model locally; a hosted S3-compatible service can replace it later without redesigning artifact references.

### React, TypeScript, and Vite frontend

**Proof-of-concept limitation:** The CLI and static output files do not provide an accessible interface for launching work, monitoring state, comparing runs, or exploring trained models.

**Decision and production role:** Build a browser frontend with React, TypeScript, and Vite. Its initial views will cover run discovery, simulation results, training results, and the model catalog. Vite will provide the development server and produce static production assets.

React supplies component-based interactive views without coupling visualization to Python request processes. TypeScript makes API and visualization data contracts explicit, while static assets can be served by a lightweight web server or CDN independently of the API.

### TanStack Query and an OpenAPI-generated client

**Proof-of-concept limitation:** Manually maintained browser request types and ad hoc request state would allow the frontend to drift from the FastAPI contract and would duplicate loading, error, cache, and polling logic.

**Decision and production role:** Generate TypeScript API types and client operations from FastAPI's OpenAPI document. Use TanStack Query for server-state caching, request status, invalidation, and bounded polling of queued or running jobs.

The generated client makes API contract changes visible at the frontend boundary. TanStack Query centralizes remote-state behavior rather than treating server data as locally maintained application state.

### Plotly visualization

**Proof-of-concept limitation:** Saved PNG files support static inspection but not interactive exploration of trajectories, residuals, and training behavior.

**Decision and production role:** Use Plotly for interactive trajectory, loss, parity, and residual visualizations. The API will expose numerical result projections suitable for charting; canonical arrays will remain in NPZ artifacts. Existing plotting code will continue to produce static report artifacts.

Interactive charts provide zooming, inspection, and comparison while preserving reproducible underlying data. The browser will not become responsible for parsing Equinox checkpoints or scientific binary formats.

### Nginx frontend serving and request proxying

**Proof-of-concept limitation:** Vite's development server is not a production static-asset server, and separate frontend/API origins would otherwise require ad hoc browser routing and CORS configuration.

**Decision and production role:** Serve the compiled frontend with Nginx and proxy same-origin `/api` requests to FastAPI in the local production-shaped container setup.

Nginx keeps static delivery outside the Python API process and provides a single browser origin. A managed static host, CDN, ingress, or load balancer can assume this role in a deployed environment.

### Docker images and Docker Compose

**Proof-of-concept limitation:** A local `uv` environment reproduces Python dependencies but does not describe the processes, network, service dependencies, health checks, or stateful infrastructure required by the platform.

**Decision and production role:** Build reproducible container images and define the local frontend, API, worker, PostgreSQL, Redis, and MinIO topology in Docker Compose. Use named volumes for local service state and health checks for startup dependencies.

Compose makes the complete local system startable through one interface and exercises production-like process boundaries during development. It is not the production scaling or scheduling mechanism; horizontal scaling will come from stateless service design and externalized state.

## Alternatives considered

### Synchronous FastAPI requests or background tasks

This would minimize initial infrastructure, but long-running computations would remain coupled to API process lifetime and capacity. It would also make independent compute scaling and reliable queueing harder. It is suitable only for operations with a demonstrated short, bounded latency, not the primary workflow execution model.

### CLI subprocesses behind the API

Subprocesses would preserve command behavior without refactoring, but structured errors, run metadata, progress, cancellation, and resource management would become adapter-specific. Shared application services provide a cleaner boundary while retaining the CLI.

### SQLite and local artifact directories

SQLite and files would reduce the local container count and remain appropriate for a single-process CLI. They do not provide the concurrent metadata and shared artifact boundaries required by independently scaled API and worker processes. The additional PostgreSQL and object-storage components are accepted to avoid redesigning those boundaries during deployment.

### Celery or RQ instead of Dramatiq

Celery provides a broad feature set and ecosystem but more configuration and operational concepts than the initial platform requires. RQ is simple but would provide fewer messaging and worker-management features. Dramatiq is the selected middle ground; the queue boundary should remain small enough that another broker-backed executor can replace it if requirements change.

### Streamlit, notebooks, or server-rendered plots

These options would shorten the path to a basic demonstration but couple interactive state and rendering more closely to Python processes. A separate frontend provides a clearer deployment boundary, generated API contracts, and more control over a multi-view scientific application.

### MLflow or a dedicated model-registry service

A dedicated experiment platform could provide tracking and registry features, but current needs are limited to project-specific run lineage and self-contained checkpoints. PostgreSQL run/model records and object storage are sufficient initially. MLflow remains an option if cross-project experiment workflows or registry features become requirements.

### Microservices and Kubernetes

Splitting simulation, training, metadata, and API logic into independently deployed services would add network contracts, distributed operations, and deployment overhead before those boundaries have demonstrated value. The modular monolith keeps scientific workflows in one codebase while preserving independently scalable API and worker processes. Kubernetes is deferred until scheduling or deployment requirements justify it.

## Tradeoffs

### Accepted costs

- Local development gains several stateful services and therefore more resource use, configuration, backup considerations, and failure modes than the CLI-only project.
- Redis broker delivery and failure behavior must be understood and tested; enqueueing a job does not replace persistent run state or idempotent worker design.
- PostgreSQL schemas and object keys become compatibility boundaries that require migrations and lifecycle policies.
- Asynchronous execution requires status handling and makes results eventually available rather than immediately returned.
- Polling adds some status latency and repeated requests; it should use bounded intervals and stop for terminal runs.
- The frontend adds a second language, dependency lockfile, build pipeline, and generated-contract checks.
- MinIO is a local S3-compatible implementation, not proof that a particular hosted object store's availability, security, or consistency configuration is correct.
- Container sharing keeps API and worker dependencies consistent but can produce a larger API image than strictly necessary.
- Local-only deployment may initially omit authentication, but exposed or multi-user deployments require authentication, authorization, TLS, secret management, and artifact-access controls before use.

### Avoided costs

- No scientific workflow duplication across CLI, API, and worker adapters.
- No long-running JAX computation tied to web-request processes.
- No binary artifacts stored in PostgreSQL or run metadata reconstructed by scanning directories.
- No shared local-filesystem requirement between API and worker replicas.
- No premature microservice network boundaries, Kubernetes control plane, or dedicated experiment platform.
- No handwritten frontend API types or per-view request-state machinery.

## Scaling and deployment path

API replicas can scale independently because they hold no authoritative run or artifact state. Worker replicas can scale according to queue depth and resource requirements, with separate pools for CPU and GPU workloads when needed. Local PostgreSQL, Redis, and MinIO can be replaced by managed services, and static frontend assets can move to a CDN.

The first production deployment need not use Kubernetes. Virtual machines or managed container services are sufficient while the service count and scheduling requirements remain small.

## Assumptions and reversal triggers

Revisit this decision if any of the following assumptions stops holding:

- **The scientific workflows remain a coherent modular monolith.** Independently owned workloads or incompatible release cycles may justify separate services.
- **Redis and Dramatiq meet delivery and operational requirements.** Stronger durability, scheduling, routing, or observability needs may justify another broker or task system.
- **PostgreSQL and object storage meet scale requirements.** Substantially larger analytical queries or datasets may require dedicated analytical or data-lake infrastructure rather than overloading run metadata.
- **Polling is sufficient.** Live epoch updates or collaborative monitoring may justify server-sent events or WebSockets, backed by explicit progress events rather than frequent database writes.
- **A project-specific model catalog is sufficient.** Cross-project experiment management or approval workflows may justify MLflow or another registry.
- **One API/worker image remains practical.** Strong dependency isolation or image-size requirements may justify separately built images from the same locked source revision.
- **Local Compose remains representative enough.** Deployment requirements will require additional production configuration for security, backups, monitoring, resource limits, and disaster recovery.
- **The current frontend remains the primary interactive interface.** Embedding results in notebooks or another host application may change visualization and delivery choices without changing the backend run model.

Any reassessment should preserve reproducible typed configurations, authoritative run metadata, self-contained model artifacts, independent compute scaling, and parity between CLI and platform workflow behavior.
