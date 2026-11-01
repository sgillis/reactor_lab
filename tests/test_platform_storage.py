from __future__ import annotations

import io

from botocore.exceptions import ClientError

from reactor_lab.platform.settings import get_settings
from reactor_lab.platform.storage import S3Storage


class FakeS3Client:
    def __init__(self) -> None:
        self.created_buckets: list[str] = []
        self.objects: dict[tuple[str, str], bytes] = {}

    def head_bucket(self, *, Bucket: str) -> None:
        if Bucket not in self.created_buckets:
            raise ClientError(
                {
                    "Error": {"Code": "404"},
                    "ResponseMetadata": {"HTTPStatusCode": 404},
                },
                "HeadBucket",
            )

    def create_bucket(self, *, Bucket: str, **kwargs) -> None:
        self.created_buckets.append(Bucket)

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **kwargs) -> None:
        self.objects[(Bucket, Key)] = Body

    def upload_fileobj(self, stream, Bucket: str, Key: str, **kwargs) -> None:
        self.objects[(Bucket, Key)] = stream.read()

    def get_object(self, *, Bucket: str, Key: str):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


def test_storage_creates_missing_bucket_once_and_overwrites_scoped_keys(monkeypatch) -> None:
    client = FakeS3Client()
    monkeypatch.setattr("reactor_lab.platform.storage.boto3.client", lambda *args, **kwargs: client)
    get_settings.cache_clear()
    try:
        storage = S3Storage()
        storage.put_bytes("runs/one/results.json", b"first", "application/json")
        storage.put_bytes("runs/one/results.json", b"second", "application/json")

        assert client.created_buckets == ["reactor-lab"]
        assert storage.get_bytes("runs/one/results.json") == b"second"
    finally:
        get_settings.cache_clear()
