"""S3-compatible artifact storage with lazy clients and automatic bucket creation."""

from __future__ import annotations

import mimetypes
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from .settings import get_settings

_MEDIA_TYPES = {
    ".eqx": "application/octet-stream",
    ".json": "application/json",
    ".npz": "application/x-npz",
    ".png": "image/png",
}


def media_type_for(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    return _MEDIA_TYPES.get(suffix) or mimetypes.guess_type(path)[0] or "application/octet-stream"


class S3Storage:
    """Small artifact interface shared by API downloads and worker uploads."""

    def __init__(self) -> None:
        settings = get_settings()
        self.bucket = settings.s3_bucket
        self.region = settings.s3_region
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
            config=Config(s3={"addressing_style": "path"}),
        )
        self._bucket_ready = False
        self._bucket_lock = threading.Lock()

    def _ensure_bucket(self) -> None:
        if self._bucket_ready:
            return
        with self._bucket_lock:
            if self._bucket_ready:
                return
            try:
                self.client.head_bucket(Bucket=self.bucket)
            except ClientError as error:
                code = error.response.get("Error", {}).get("Code")
                status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if code not in {"404", "NoSuchBucket", "NotFound"} and status != 404:
                    raise
                parameters: dict[str, Any] = {"Bucket": self.bucket}
                if self.region != "us-east-1":
                    parameters["CreateBucketConfiguration"] = {
                        "LocationConstraint": self.region
                    }
                try:
                    self.client.create_bucket(**parameters)
                except ClientError as create_error:
                    create_code = create_error.response.get("Error", {}).get("Code")
                    if create_code not in {"BucketAlreadyExists", "BucketAlreadyOwnedByYou"}:
                        raise
            self._bucket_ready = True

    def put_file(self, key: str, path: str | Path, media_type: str | None = None) -> int:
        path = Path(path)
        self._ensure_bucket()
        with path.open("rb") as stream:
            self.client.upload_fileobj(
                stream,
                self.bucket,
                key,
                ExtraArgs={"ContentType": media_type or media_type_for(path)},
            )
        return path.stat().st_size

    def put_bytes(self, key: str, payload: bytes, media_type: str) -> int:
        self._ensure_bucket()
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=payload,
            ContentType=media_type,
        )
        return len(payload)

    def get_object(self, key: str) -> dict[str, Any]:
        self._ensure_bucket()
        return self.client.get_object(Bucket=self.bucket, Key=key)

    def download_file(self, key: str, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_bucket()
        self.client.download_file(self.bucket, key, str(path))

    def get_bytes(self, key: str) -> bytes:
        response = self.get_object(key)
        body = response["Body"]
        try:
            return body.read()
        finally:
            body.close()


@lru_cache(maxsize=1)
def get_storage() -> S3Storage:
    return S3Storage()
