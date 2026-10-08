"""Optional Azure Blob Storage for uploaded CVs.

Disabled unless the deployment sets ``CV_STORAGE_ENABLED=true`` *and* a
connection string. The caller must additionally obtain explicit consent from
the visitor. Blob names are random; the original file name is never stored and
there is deliberately no function that lists stored CVs.
"""

from __future__ import annotations

import os
import uuid

CONTAINER = "cv-uploads"


def is_enabled() -> bool:
    return os.getenv("CV_STORAGE_ENABLED", "").lower() == "true" and bool(
        os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )


def new_blob_name() -> str:
    return f"{uuid.uuid4().hex}.pdf"


def upload_cv(data: bytes) -> bool:
    """Upload CV bytes under a random name. Returns ``True`` on success."""
    if not is_enabled():
        return False
    try:
        from azure.storage.blob import BlobServiceClient

        service = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])
        container = service.get_container_client(CONTAINER)
        if not container.exists():
            container.create_container()  # private by default
        container.upload_blob(new_blob_name(), data, overwrite=False)
    except Exception:
        return False
    return True
