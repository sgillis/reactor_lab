"""Environment-based settings with development-only local defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str = "postgresql+psycopg://reactor:reactor@localhost:5432/reactor"
    redis_url: str = "redis://localhost:6379/0"
    s3_endpoint_url: str = "http://localhost:9000"
    s3_bucket: str = "reactor-lab"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_region: str = "us-east-1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    defaults = Settings()
    return Settings(
        database_url=os.getenv("DATABASE_URL", defaults.database_url),
        redis_url=os.getenv("REDIS_URL", defaults.redis_url),
        s3_endpoint_url=os.getenv("S3_ENDPOINT_URL", defaults.s3_endpoint_url),
        s3_bucket=os.getenv("S3_BUCKET", defaults.s3_bucket),
        s3_access_key=os.getenv("S3_ACCESS_KEY", defaults.s3_access_key),
        s3_secret_key=os.getenv("S3_SECRET_KEY", defaults.s3_secret_key),
        s3_region=os.getenv("S3_REGION", defaults.s3_region),
    )
