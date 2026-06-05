"""
FastAPI app for live Arlians stream (protocol v1).

Run from repo root:
  uvicorn stream.server.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Repo root on path (same pattern as scripts/demo.py)
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stream.server.config import StreamConfig
from stream.server.engine import SimEngine

_cfg = StreamConfig.from_env()
_engine = SimEngine(_cfg)
_clients: set[WebSocket] = set()
_step_task: asyncio.Task | None = None
_step_lock = asyncio.Lock()


app = FastAPI(title="Arlians Live Stream", version="1")


class ControlBody(BaseModel):
    action: str


@app.get("/api/v1/bootstrap")
def get_bootstrap():
    return _engine.bootstrap


@app.get("/api/v1/health")
def get_health():
    return _engine.health()


@app.get("/api/v1/delta")
def get_delta(after: int = -1):
    delta = _engine.get_delta_after(after)
    if delta is None:
        return JSONResponse(status_code=204, content=None)
    return delta


@app.post("/api/v1/control")
def post_control(body: ControlBody):
    action = body.action.lower()
    if action == "pause":
        _engine.set_paused(True)
    elif action == "play":
        _engine.set_paused(False)
    elif action == "reset":
        _engine.reset()
    else:
        return JSONResponse(status_code=400, content={"error": f"unknown action: {action}"})
    return _engine.health()


async def _broadcast(message: dict) -> None:
    dead: list[WebSocket] = []
    text = __import__("json").dumps(message)
    for ws in list(_clients):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


async def _step_loop() -> None:
    interval = 1.0 / max(_cfg.steps_per_sec, 0.1)
    while _clients:
        if not _engine.health()["paused"]:
            delta = await asyncio.to_thread(_engine.step_once)
            if delta:
                await _broadcast(delta)
        await asyncio.sleep(interval)


async def _ensure_step_loop() -> None:
    global _step_task
    async with _step_lock:
        if _step_task is None or _step_task.done():
            _step_task = asyncio.create_task(_step_loop())


@app.websocket("/api/v1/stream")
async def websocket_stream(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    await _ensure_step_loop()
    try:
        import json

        await ws.send_text(json.dumps({"type": "BootstrapRef", "bootstrap": _engine.bootstrap}))
        latest = _engine.latest_delta
        if latest:
            await ws.send_text(json.dumps(latest))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)


_client_dir = Path(__file__).resolve().parents[1] / "client"
if _client_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_client_dir)), name="static")


@app.get("/")
def index():
    index_path = _client_dir / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return JSONResponse({"message": "Arlians stream API", "bootstrap": "/api/v1/bootstrap"})
