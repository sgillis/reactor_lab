# ADR 0003: Use RustFS for local S3-compatible object storage

- Status: Accepted
- Date: 2026-11-01
- Supersedes: The MinIO container selection in ADR 0002

## Context

ADR 0002 selected MinIO as the local implementation of the platform's S3-compatible artifact-store boundary. The application stores datasets, trajectories, checkpoints, metrics, and plots through boto3 and treats PostgreSQL as the metadata system of record.

The public `minio/minio` community images were subsequently removed from Docker Hub, and the former community Quay images are also unavailable. Pulling the configured image therefore fails even after successful Docker Hub authentication. MinIO AIStor remains available but has different distribution and licensing requirements that are unnecessary for this local development deployment.

The platform still needs a locally runnable S3-compatible service. The scientific workflows, API schemas, metadata model, object keys, credentials, ports, and future managed-S3 migration path should not change because of the container supplier.

## Decision

Use the official **RustFS** container as the local S3-compatible object store. Pin the container by its multi-architecture image digest rather than relying on a mutable `latest` tag.

The Compose service and DNS name are `objectstore`; its named volume is `objectstore_data`. It exposes the existing localhost-only S3 API and console ports, 9000 and 9001. RustFS health checks use its `/health` endpoint. The API and worker continue to use the same `S3_*` environment variables and boto3 path-style S3 requests.

RustFS is an implementation detail behind the existing storage interface. This decision does not change the production preference for a managed S3-compatible object store, nor does it introduce a dependency on MinIO- or RustFS-specific application APIs.

The locally built API and worker images declare `pull_policy: build` so `docker compose pull` does not attempt to fetch the local `reactor-lab-platform` tag from a registry.

## Rationale

- The [official RustFS image](https://docs.rustfs.com/en/installation/container/docker) is publicly available, supports named volumes and health checks, and provides the required S3 API and browser console.
- RustFS is [Apache 2.0 licensed](https://rustfs.org/developer/license), avoiding a new license prerequisite for the local platform.
- Keeping the S3 boundary limits the change to Compose configuration and operator documentation.
- Pinning the image digest makes local setup reproducible across repeated pulls while retaining multi-architecture support.
- Building local service images explicitly distinguishes registry-hosted infrastructure from repository-built application images.

## Alternatives considered

### Continue using MinIO community images

The required public images are no longer available, so this is not a working deployment path. Depending on unofficial re-uploads would add an unverified supply-chain boundary.

### MinIO AIStor

AIStor is the maintained MinIO distribution but introduces licensing and operational requirements beyond this project's local development needs. The application does not require AIStor-specific capabilities.

### Garage or another S3-compatible service

Garage is a viable S3-compatible store, but its cluster layout, node initialization, and key/bucket administration model require more local configuration. RustFS is a smaller operational cutover from the existing single-node object-store service.

### Local filesystem or mocked S3

These would reduce infrastructure but would remove the production-shaped object-storage boundary that ADR 0002 intentionally introduced. They remain useful for isolated tests, not the Compose deployment.

## Tradeoffs

- RustFS is a different implementation with its own compatibility defects, release process, console, and operational behavior. Required S3 operations must be exercised in integration testing rather than assumed equivalent.
- Digest pinning prevents silent upstream changes but requires deliberate dependency updates to receive fixes.
- Existing MinIO named volumes are not migrated because there is no compatible existing deployment or data migration requirement in this project.
- Local development credentials remain intentionally insecure and localhost-only; production deployments still require managed credentials, authorization, TLS, backups, and lifecycle policies.

## Verification and reversal triggers

The Compose smoke workflow must verify bucket creation, artifact upload, result download, checkpoint restoration during optimization, and persistence across container recreation.

Reconsider this decision if RustFS cannot faithfully support the required S3 operations, its image availability or maintenance regresses, or the project adopts managed object storage as its only supported runtime. Any replacement must preserve the existing storage interface and artifact references rather than changing scientific workflow behavior.
