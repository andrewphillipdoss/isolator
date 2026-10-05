"""Local web UI: upload a song, watch it process, audition and balance the layer kit.

Runs on 127.0.0.1 only by default. The heavy work happens in one background
worker thread, because the separation models want the whole GPU (or CPU)
to themselves.
"""

from __future__ import annotations

import io
import json
import queue
import re
import threading
import traceback
import uuid
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import hmac

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .layers import LayerTrack, load_session, mixdown, save_session, write_reaper_project
from .pipeline import PRESETS, run

WEB_DIR = Path(__file__).parent / "web"
_SAFE = re.compile(r"^[\w.\- ()\[\]]+$")


def _safe_name(name: str) -> str:
    if not _SAFE.match(name) or name in {".", ".."}:
        raise HTTPException(400, "bad name")
    return name


def create_app(out_root: Path, backend_factory: Callable, kits_root: Path | None = None, token: str | None = None) -> FastAPI:
    out_root.mkdir(parents=True, exist_ok=True)
    uploads = out_root / ".uploads"
    uploads.mkdir(exist_ok=True)
    kits_root = kits_root or (Path.home() / ".cache" / "iso" / "kits")
    jobs: dict[str, dict] = {}
    work: queue.Queue = queue.Queue()
    backends: dict = {}  # one per (tta, overlap) so switching presets doesn't reload needlessly

    def worker():
        while True:
            job_id, song, cfg = work.get()
            job = jobs[job_id]
            job["state"] = "running"
            try:
                key = (cfg.tta, cfg.overlap)
                if key not in backends:
                    job["message"] = "Loading models"
                    backends.clear()  # free the previous preset's models before loading new ones
                    backends[key] = backend_factory(cfg)

                def progress(msg: str, frac: float):
                    job["message"], job["progress"] = msg, frac

                job["report"] = run(song, out_root / job["name"], backends[key], cfg, progress)
                job["state"] = "done"
            except Exception as e:
                job["state"], job["message"] = "error", f"{e.__class__.__name__}: {e}"
                job["trace"] = traceback.format_exc()
            finally:
                work.task_done()

    threading.Thread(target=worker, daemon=True).start()
    app = FastAPI(title="Iso")

    if token:
        # Open the page once as /?token=...; a cookie carries it from then on.
        @app.middleware("http")
        async def require_token(request: Request, call_next):
            given = request.query_params.get("token") or request.cookies.get("iso_token") or ""
            if not hmac.compare_digest(given, token):
                return PlainTextResponse("Iso: add ?token=... to the URL", status_code=401)
            response = await call_next(request)
            if request.query_params.get("token"):
                response.set_cookie("iso_token", token, httponly=True, samesite="strict")
            return response

    @app.get("/api/presets")
    def presets():
        return {k: asdict(v) for k, v in PRESETS.items()}

    @app.get("/api/trigger-kits")
    def trigger_kits():
        if not kits_root.exists():
            return {"root": str(kits_root), "kits": []}
        return {"root": str(kits_root), "kits": sorted(p.name for p in kits_root.iterdir() if p.is_dir())}

    @app.post("/api/jobs")
    async def create_job(
        file: UploadFile = File(...),
        preset: str = Form("best"),
        kit: str = Form(""),
        gate: bool = Form(True),
    ):
        if preset not in PRESETS:
            raise HTTPException(400, f"unknown preset {preset}")
        stem = Path(file.filename or "song").stem
        name = re.sub(r"[^\w.\- ()]+", "_", stem)[:80] or "song"
        dest = uploads / f"{uuid.uuid4().hex[:8]}_{name}{Path(file.filename or '').suffix}"
        dest.write_bytes(await file.read())
        kit_dir = str(kits_root / _safe_name(kit)) if kit else None
        cfg = replace(PRESETS[preset], kit=kit_dir, gate=gate)
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {"id": job_id, "name": name, "state": "queued", "message": "Queued", "progress": 0.0}
        work.put((job_id, dest, cfg))
        return jobs[job_id]

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "no such job")
        return {k: v for k, v in jobs[job_id].items() if k != "trace"}

    @app.get("/api/kits")
    def list_kits():
        items = []
        for d in sorted(out_root.iterdir()):
            rep = d / "report.json"
            if d.is_dir() and rep.exists():
                r = json.loads(rep.read_text())
                items.append({"name": d.name, "seconds": r.get("seconds"), "bpm": r.get("bpm"), "hits": r.get("hits")})
        return items

    @app.get("/api/kits/{name}")
    def get_kit(name: str):
        d = out_root / _safe_name(name)
        if not (d / "session.json").exists():
            raise HTTPException(404, "no such kit")
        tracks, sr = load_session(d / "session.json")
        report = json.loads((d / "report.json").read_text())
        return {"name": name, "sr": sr, "tracks": [asdict(t) for t in tracks], "report": report}

    @app.put("/api/kits/{name}/session")
    def put_session(name: str, body: dict):
        d = out_root / _safe_name(name)
        old, sr = load_session(d / "session.json")
        by_name = {t.name: t for t in old}
        for t in body.get("tracks", []):
            if t.get("name") in by_name:
                cur = by_name[t["name"]]
                cur.gain_db = float(t.get("gain_db", cur.gain_db))
                cur.muted = bool(t.get("muted", cur.muted))
        tracks = list(by_name.values())
        save_session(d / "session.json", tracks, sr)
        report = json.loads((d / "report.json").read_text())
        write_reaper_project(d / "layer_kit.rpp", tracks, sr, report["seconds"], midi_file=report.get("midi"), bpm=report.get("bpm"))
        return {"ok": True}

    @app.post("/api/kits/{name}/bounce")
    def bounce(name: str, with_song: bool = False):
        from .audio import save

        d = out_root / _safe_name(name)
        tracks, _ = load_session(d / "session.json")
        out = save(d / ("bounce_with_song.wav" if with_song else "bounce_drums.wav"), mixdown(d, tracks, include_song=with_song))
        return {"file": out.name}

    @app.get("/api/kits/{name}/zip")
    def zip_kit(name: str):
        d = out_root / _safe_name(name)
        if not d.is_dir():
            raise HTTPException(404, "no such kit")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for f in sorted(d.rglob("*")):
                if f.is_file():
                    z.write(f, f"{name}/{f.relative_to(d)}")
        buf.seek(0)
        return StreamingResponse(
            buf, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}_layer_kit.zip"'}
        )

    app.mount("/files", StaticFiles(directory=str(out_root)), name="files")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    return app


def serve(host: str, port: int, out_root: Path, backend_factory: Callable, token: str | None = None) -> None:
    import uvicorn

    app = create_app(out_root, backend_factory, token=token)
    print(f"Iso is running at http://{host}:{port}" + (f"/?token={token}" if token else ""))
    uvicorn.run(app, host=host, port=port, log_level="warning")


__all__ = ["create_app", "serve", "LayerTrack"]
