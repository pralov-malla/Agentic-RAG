"""Stable application exceptions and error codes."""

from enum import StrEnum

from fastapi import HTTPException


class ErrorCode(StrEnum):
    """Machine-readable error codes from the API contract."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_DOCUMENT = "INVALID_DOCUMENT"
    INDEX_NOT_READY = "INDEX_NOT_READY"
    INGESTION_IN_PROGRESS = "INGESTION_IN_PROGRESS"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    THREAD_NOT_FOUND = "THREAD_NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class APIException(HTTPException):
    """HTTP exception that carries the API contract error code."""

    def __init__(
        self,
        *,
        status_code: int,
        message: str,
        error_code: ErrorCode | str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            status_code=status_code,
            detail={"message": message, "error_code": str(error_code)},
            headers=headers,
        )
