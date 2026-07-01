"""Web UI endpoints."""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["web"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))


@router.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse(request=request, name="index.html", context={})
