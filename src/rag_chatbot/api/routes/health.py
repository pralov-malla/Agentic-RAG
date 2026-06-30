"""Application liveness endpoint."""

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness():
    """Simple liveness check."""
    return {"status": "ok"}
