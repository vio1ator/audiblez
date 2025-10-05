#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audiblez-web

FastAPI-powered web UI that mirrors the features of the TUI while providing a
responsive browser experience. It reuses the core audiobook pipeline and
exposes filesystem browsing, chapter selection, preview, and synthesis
controls over HTTP.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import core
from .pdf import extract_pages
from .tts_backends import has_mlx_audio
from .voices import flags as VOICE_FLAGS
from .voices import voices as VOICES_BY_LANG

try:
    from ebooklib import epub
except Exception as exc:  # pragma: no cover - surfaced via API response
    raise RuntimeError(
        "ebooklib is required for audiblez-web. Ensure optional dependencies are installed."
    ) from exc


# ------------------------------ Utilities ------------------------------


def _flatten_voices() -> List[str]:
    voices: List[str] = []
    for _, variants in VOICES_BY_LANG.items():
        voices.extend(variants)
    return voices


def _grouped_voices() -> List[Dict[str, Any]]:
    grouped: List[Dict[str, Any]] = []
    for key, variants in VOICES_BY_LANG.items():
        grouped.append(
            {
                "group": key,
                "flag": VOICE_FLAGS.get(key, ""),
                "voices": list(variants),
            }
        )
    return grouped


def _choose_device_auto() -> str:
    """Auto-select compute device without blocking the UI."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.set_default_device("cuda")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.set_default_device("mps")
            return "mps"
    except Exception:
        pass
    return "cpu"


def _sanitize_path(raw: str | None) -> Path:
    if not raw:
        raise HTTPException(status_code=400, detail="Path is required")
    try:
        return Path(raw).expanduser()
    except Exception as exc:  # pragma: no cover - path resolution failure
        raise HTTPException(status_code=400, detail=f"Invalid path: {raw}") from exc


def _compute_words(text: str) -> int:
    return len((text or "").split())


def _resolve_backend(preferred: str) -> str:
    if preferred == "auto":
        try:
            return "mlx" if has_mlx_audio() else "kokoro"
        except Exception:
            return "kokoro"
    return preferred


@dataclass
class Margins:
    header: float = 0.07
    footer: float = 0.07
    left: float = 0.07
    right: float = 0.07


def _margins_dict(margins: Margins) -> Dict[str, float]:
    return {
        "header": float(margins.header),
        "footer": float(margins.footer),
        "left": float(margins.left),
        "right": float(margins.right),
    }


# ------------------------------ Data Models ------------------------------


@dataclass
class ChapterInfo:
    index: int
    name: str
    preview: str
    characters: int
    words: int
    selected: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DirectoryEntry(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: Optional[int] = None


class SettingsPayload(BaseModel):
    voice: Optional[str] = Field(None, description="Voice identifier")
    speed: Optional[float] = Field(None, ge=0.5, le=2.0)
    backend: Optional[str] = Field(None, pattern=r"^(auto|mlx|kokoro)$")
    mlx_model: Optional[str] = None
    gold_min: Optional[int] = Field(None, ge=1)
    gold_ideal: Optional[int] = Field(None, ge=1)
    gold_max: Optional[int] = Field(None, ge=1)
    debug_text: Optional[bool] = None
    debug_text_file: Optional[str] = None


class MarginsPayload(BaseModel):
    header: float = Field(0.07, ge=0.0, le=0.3)
    footer: float = Field(0.07, ge=0.0, le=0.3)
    left: float = Field(0.07, ge=0.0, le=0.3)
    right: float = Field(0.07, ge=0.0, le=0.3)


class SelectionPayload(BaseModel):
    selected: List[int]


class MinCharsPayload(BaseModel):
    minimum: int = Field(..., ge=0)


class JobTriggerPayload(BaseModel):
    open_on_complete: bool = False


class FileLoadPayload(BaseModel):
    path: str
    margins: Optional[MarginsPayload] = None


# ------------------------------ State Containers ------------------------------


class WebState:
    """Shared state across API handlers."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.device = _choose_device_auto()
        self.voices = _flatten_voices()
        self.voice_groups = _grouped_voices()
        self.voice = self.voices[0] if self.voices else "af_sky"
        self.backend = "auto"
        self.mlx_model = "mlx-community/Kokoro-82M-8bit"
        self.speed = 1.0
        self.gold_min = 10
        self.gold_ideal = 25
        self.gold_max = 40
        self.debug_text = False
        self.debug_text_file: Optional[str] = None
        self.margins = Margins()
        self.file_path: Optional[Path] = None
        self.is_pdf = False
        self.output_dir = Path.cwd()
        self._chapters: List[Any] = []  # raw chapter objects
        self._chapter_info: List[ChapterInfo] = []
        self._selected: Set[int] = set()
        self.job: Optional[SynthesisJob] = None

    # ------- Chapter helpers
    def _build_chapter_info(self, raw: Sequence[Any], default_selected: Set[int]) -> List[ChapterInfo]:
        info: List[ChapterInfo] = []
        for chapter in raw:
            idx = int(getattr(chapter, "chapter_index", len(info)))
            name = getattr(chapter, "get_name", lambda: f"Chapter {idx + 1}")()
            text = getattr(chapter, "extracted_text", "") or ""
            preview = core.chapter_beginning_one_liner(chapter, 70)
            info.append(
                ChapterInfo(
                    index=idx,
                    name=name,
                    preview=preview,
                    characters=len(text),
                    words=_compute_words(text),
                    selected=idx in default_selected,
                )
            )
        return info

    def set_file(self, path: Path, margins: Optional[MarginsPayload] = None) -> None:
        with self.lock:
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"File not found: {path}")
            if not path.is_file():
                raise HTTPException(status_code=400, detail="Selected path is not a file")
            suffix = path.suffix.lower()
            if suffix not in {".epub", ".pdf"}:
                raise HTTPException(status_code=400, detail="Only EPUB and PDF files are supported")

            if margins:
                self.margins = Margins(
                    header=margins.header,
                    footer=margins.footer,
                    left=margins.left,
                    right=margins.right,
                )

            self.file_path = path.resolve()
            self.is_pdf = suffix == ".pdf"
            self._selected.clear()

            if self.is_pdf:
                raw = extract_pages(self.file_path, _margins_dict(self.margins))
                default_selected = {c.chapter_index for c in raw if (c.extracted_text or "").strip()}
            else:
                book = epub.read_epub(str(self.file_path))
                raw = core.find_document_chapters_and_extract_texts(book)
                good = set(core.find_good_chapters(raw))
                default_selected = {c.chapter_index for c in raw if c in good}

            self._chapters = list(raw)
            self._chapter_info = self._build_chapter_info(raw, default_selected)
            self._selected = set(default_selected)

    def ensure_file_loaded(self) -> None:
        if not self.file_path:
            raise HTTPException(status_code=400, detail="Select an EPUB or PDF file first")

    def get_chapter_text(self, index: int) -> str:
        with self.lock:
            self.ensure_file_loaded()
            try:
                chapter = self._chapters[index]
            except Exception:
                raise HTTPException(status_code=404, detail="Chapter not found")
            return getattr(chapter, "extracted_text", "") or ""

    def update_selection(self, selected: Iterable[int]) -> List[ChapterInfo]:
        with self.lock:
            self.ensure_file_loaded()
            indices = set(int(i) for i in selected)
            max_idx = len(self._chapters) - 1
            for idx in indices:
                if idx < 0 or idx > max_idx:
                    raise HTTPException(status_code=400, detail=f"Invalid chapter index: {idx}")
            self._selected = indices
            info = []
            for ci in self._chapter_info:
                ci.selected = ci.index in self._selected
                info.append(ci)
            return info

    def select_all(self, enabled: bool) -> List[ChapterInfo]:
        with self.lock:
            self.ensure_file_loaded()
            if enabled:
                self._selected = {ci.index for ci in self._chapter_info}
            else:
                self._selected.clear()
            for ci in self._chapter_info:
                ci.selected = ci.index in self._selected
            return list(self._chapter_info)

    def select_min_chars(self, minimum: int) -> List[ChapterInfo]:
        with self.lock:
            self.ensure_file_loaded()
            self._selected = {ci.index for ci in self._chapter_info if ci.characters >= minimum}
            for ci in self._chapter_info:
                ci.selected = ci.index in self._selected
            return list(self._chapter_info)

    def auto_select(self) -> List[ChapterInfo]:
        with self.lock:
            self.ensure_file_loaded()
            if self.is_pdf:
                self._selected = {ci.index for ci in self._chapter_info if ci.characters > 0}
            else:
                good = set(core.find_good_chapters(self._chapters))
                self._selected = {c.chapter_index for c in good}
            for ci in self._chapter_info:
                ci.selected = ci.index in self._selected
            return list(self._chapter_info)

    def refresh_pdf_with_margins(self, payload: MarginsPayload) -> List[ChapterInfo]:
        with self.lock:
            self.ensure_file_loaded()
            if not self.is_pdf:
                raise HTTPException(status_code=400, detail="PDF margins only apply to PDF documents")
            self.margins = Margins(
                header=payload.header,
                footer=payload.footer,
                left=payload.left,
                right=payload.right,
            )
            raw = extract_pages(self.file_path, _margins_dict(self.margins))
            self._chapters = list(raw)
            self._chapter_info = self._build_chapter_info(raw, {c.chapter_index for c in raw})
            self._selected = {ci.index for ci in self._chapter_info if ci.characters > 0}
            for ci in self._chapter_info:
                ci.selected = ci.index in self._selected
            return list(self._chapter_info)

    def to_dict(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "file": {
                    "path": str(self.file_path) if self.file_path else None,
                    "is_pdf": self.is_pdf,
                },
                "output_dir": str(self.output_dir),
                "device": self.device,
                "voices": self.voice_groups,
                "voice": self.voice,
                "backend": self.backend,
                "resolved_backend": _resolve_backend(self.backend),
                "mlx_model": self.mlx_model,
                "speed": self.speed,
                "gold_min": self.gold_min,
                "gold_ideal": self.gold_ideal,
                "gold_max": self.gold_max,
                "debug_text": self.debug_text,
                "debug_text_file": self.debug_text_file,
                "margins": _margins_dict(self.margins),
                "chapters": [ci.to_dict() for ci in self._chapter_info],
                "selected_count": len(self._selected),
                "job": self.job.to_dict() if self.job else SynthesisJob.idle_state(),
            }

    # ------- Settings
    def update_settings(self, payload: SettingsPayload) -> None:
        with self.lock:
            if payload.voice:
                if payload.voice not in self.voices:
                    raise HTTPException(status_code=400, detail=f"Unknown voice: {payload.voice}")
                self.voice = payload.voice
            if payload.speed is not None:
                self.speed = float(payload.speed)
            if payload.backend is not None:
                self.backend = payload.backend
            if payload.mlx_model is not None:
                self.mlx_model = payload.mlx_model.strip() or self.mlx_model
            if payload.gold_min is not None:
                self.gold_min = int(payload.gold_min)
            if payload.gold_ideal is not None:
                self.gold_ideal = int(payload.gold_ideal)
            if payload.gold_max is not None:
                self.gold_max = int(payload.gold_max)
            if payload.debug_text is not None:
                self.debug_text = bool(payload.debug_text)
            if payload.debug_text_file is not None:
                self.debug_text_file = payload.debug_text_file

    def set_output_dir(self, raw: str) -> None:
        path = _sanitize_path(raw)
        with self.lock:
            if not path.exists():
                raise HTTPException(status_code=404, detail=f"Directory not found: {path}")
            if not path.is_dir():
                raise HTTPException(status_code=400, detail="Output path must be a directory")
            self.output_dir = path.resolve()

    def start_job(self) -> "SynthesisJob":
        with self.lock:
            self.ensure_file_loaded()
            if not self._selected:
                raise HTTPException(status_code=400, detail="Select at least one chapter or page")
            if self.job and self.job.is_running:
                raise HTTPException(status_code=409, detail="A synthesis job is already running")

            selected = [self._chapters[idx] for idx in sorted(self._selected)]
            job = SynthesisJob(
                state=self,
                file_path=self.file_path,  # type: ignore[arg-type]
                output_dir=self.output_dir,
                selected_chapters=selected,
            )
            self.job = job
        job.start()
        return job


class SynthesisJob:
    """Background synthesis worker."""

    def __init__(
        self,
        state: WebState,
        file_path: Path,
        output_dir: Path,
        selected_chapters: Sequence[Any],
    ) -> None:
        self.id = str(uuid.uuid4())
        self.state = state
        self.file_path = file_path
        self.output_dir = output_dir
        self.selected = list(selected_chapters)
        self.voice: str
        self.speed: float
        self.backend: str
        self.mlx_model: str
        self.gold_min: int
        self.gold_ideal: int
        self.gold_max: int
        self.debug_text: bool
        self.debug_text_file: Optional[str]
        with self.state.lock:
            self.voice = self.state.voice
            self.speed = self.state.speed
            self.backend = _resolve_backend(self.state.backend)
            self.mlx_model = self.state.mlx_model
            self.gold_min = self.state.gold_min
            self.gold_ideal = self.state.gold_ideal
            self.gold_max = self.state.gold_max
            self.debug_text = self.state.debug_text
            self.debug_text_file = self.state.debug_text_file
            margins = self.state.margins
        base = Path(self.file_path).name
        self.expected_wavs = [
            (self.output_dir / Path(base).with_suffix(""))
            .with_name(
                f"{Path(base).with_suffix('').name}_chapter_{i+1}_{self.voice}_{self._safe_name(ch)}.wav"
            )
            for i, ch in enumerate(self.selected)
        ]
        self.expected_m4b = self.output_dir / f"{Path(base).stem}.m4b"
        self.margins = Margins(
            header=margins.header,
            footer=margins.footer,
            left=margins.left,
            right=margins.right,
        )
        self.status = "pending"
        self.progress = 0
        self.eta: Optional[str] = None
        self.current_chapter: Optional[int] = None
        self.events: List[Dict[str, Any]] = []
        self.error: Optional[str] = None
        self.started_at: Optional[dt.datetime] = None
        self.finished_at: Optional[dt.datetime] = None
        self._thread = threading.Thread(target=self._worker, name=f"audiblez-web-{self.id}", daemon=True)
        self._lock = threading.Lock()
        self.is_running = True

    @staticmethod
    def idle_state() -> Dict[str, Any]:
        return {
            "status": "idle",
            "progress": 0,
            "eta": None,
            "current_chapter": None,
            "started_at": None,
            "finished_at": None,
            "events": [],
            "error": None,
            "expected_outputs": [],
        }

    def _safe_name(self, chapter: Any) -> str:
        name = getattr(chapter, "get_name", lambda: "chapter")()
        safe = "".join(ch for ch in name if ch.isalnum() or ch in {"-", "_"}) or "chapter"
        return safe[:40]

    def start(self) -> None:
        self._thread.start()

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "progress": self.progress,
                "eta": self.eta,
                "current_chapter": self.current_chapter,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "finished_at": self.finished_at.isoformat() if self.finished_at else None,
                "events": list(self.events),
                "error": self.error,
                "expected_outputs": [str(p) for p in self.expected_outputs()],
            }

    def expected_outputs(self) -> List[Path]:
        outputs = list(self.expected_wavs)
        outputs.append(self.expected_m4b)
        return outputs

    def _record_event(self, event: str, **payload: Any) -> None:
        with self._lock:
            self.events.append(
                {
                    "timestamp": dt.datetime.utcnow().isoformat(),
                    "event": event,
                    "payload": payload,
                }
            )

    def _post_event(self, name: str, **payload: Any) -> None:
        self._record_event(name, **payload)
        if name == "CORE_CHAPTER_STARTED":
            self.current_chapter = payload.get("chapter_index")
        elif name == "CORE_CHAPTER_FINISHED":
            self.current_chapter = None
        elif name == "CORE_PROGRESS":
            stats = payload.get("stats")
            if stats is not None:
                self.progress = int(getattr(stats, "progress", self.progress))
                self.eta = getattr(stats, "eta", None)
        elif name == "CORE_FINISHED":
            self.progress = 100

    def _worker(self) -> None:
        self.started_at = dt.datetime.utcnow()
        self.status = "running"
        _choose_device_auto()

        def callback(name: str, **payload: Any) -> None:
            self._post_event(name, **payload)

        try:
            core.main(
                file_path=str(self.file_path),
                voice=self.voice,
                pick_manually=False,
                speed=self.speed,
                output_folder=str(self.output_dir),
                max_chapters=None,
                max_sentences=None,
                selected_chapters=self.selected,
                post_event=callback,
                backend=self.backend,
                mlx_model=self.mlx_model,
                header=self.margins.header,
                footer=self.margins.footer,
                left=self.margins.left,
                right=self.margins.right,
                debug_text=self.debug_text,
                debug_text_file=self.debug_text_file,
                gold_min=self.gold_min,
                gold_ideal=self.gold_ideal,
                gold_max=self.gold_max,
            )
            self.status = "success"
            self.progress = 100
        except Exception as exc:  # pragma: no cover - surface to UI
            self.error = str(exc)
            self.status = "error"
            self._record_event("ERROR", message=str(exc))
        finally:
            self.finished_at = dt.datetime.utcnow()
            self.is_running = False


# ------------------------------ FastAPI App ------------------------------


app = FastAPI(title="Audiblez Web UI", default_response_class=JSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = WebState()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/state")
def api_state() -> Dict[str, Any]:
    return STATE.to_dict()


@app.post("/api/settings")
def api_settings(payload: SettingsPayload) -> Dict[str, Any]:
    STATE.update_settings(payload)
    return STATE.to_dict()


@app.post("/api/file")
def api_file(payload: FileLoadPayload) -> Dict[str, Any]:
    path = _sanitize_path(payload.path)
    STATE.set_file(path, payload.margins)
    return STATE.to_dict()


@app.post("/api/output")
def api_output(data: Dict[str, str]) -> Dict[str, Any]:
    path = data.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="Path is required")
    STATE.set_output_dir(path)
    return STATE.to_dict()


@app.post("/api/margins")
def api_margins(payload: MarginsPayload) -> Dict[str, Any]:
    chapters = STATE.refresh_pdf_with_margins(payload)
    out = STATE.to_dict()
    out["chapters"] = [ci.to_dict() for ci in chapters]
    return out


@app.get("/api/chapters/{index}/text")
def api_chapter_text(index: int) -> Dict[str, Any]:
    text = STATE.get_chapter_text(index)
    return {"index": index, "text": text}


@app.post("/api/chapters/selection")
def api_chapter_selection(payload: SelectionPayload) -> Dict[str, Any]:
    chapters = STATE.update_selection(payload.selected)
    out = STATE.to_dict()
    out["chapters"] = [ci.to_dict() for ci in chapters]
    return out


@app.post("/api/chapters/select-all")
def api_select_all(enable: bool = Query(True)) -> Dict[str, Any]:
    chapters = STATE.select_all(enable)
    out = STATE.to_dict()
    out["chapters"] = [ci.to_dict() for ci in chapters]
    return out


@app.post("/api/chapters/select-auto")
def api_select_auto() -> Dict[str, Any]:
    chapters = STATE.auto_select()
    out = STATE.to_dict()
    out["chapters"] = [ci.to_dict() for ci in chapters]
    return out


@app.post("/api/chapters/select-min")
def api_select_min(payload: MinCharsPayload) -> Dict[str, Any]:
    chapters = STATE.select_min_chars(payload.minimum)
    out = STATE.to_dict()
    out["chapters"] = [ci.to_dict() for ci in chapters]
    return out


@app.post("/api/job")
def api_job(payload: JobTriggerPayload) -> Dict[str, Any]:
    job = STATE.start_job()
    return job.to_dict()


@app.get("/api/job")
def api_job_status() -> Dict[str, Any]:
    return STATE.job.to_dict() if STATE.job else SynthesisJob.idle_state()


@app.get("/api/fs")
def api_fs(
    path: Optional[str] = Query(None, description="Directory to list"),
    patterns: Optional[str] = Query(None, description="Comma-separated glob patterns for files"),
    include_dirs: bool = Query(True),
) -> Dict[str, Any]:
    base = _sanitize_path(path or Path.cwd().as_posix())
    if base.is_file():
        base = base.parent
    if not base.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {base}")
    if not base.is_dir():
        raise HTTPException(status_code=400, detail="Path must be a directory")
    patterns_list = None
    if patterns:
        patterns_list = [p.strip() for p in patterns.split(",") if p.strip()]

    entries: List[DirectoryEntry] = []
    try:
        for child in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name.startswith("."):
                continue
            if child.is_file() and patterns_list:
                if not any(fnmatch.fnmatch(child.name.lower(), pat.lower()) for pat in patterns_list):
                    continue
            if child.is_dir() and not include_dirs:
                continue
            stat_size = None
            if child.is_file():
                try:
                    stat_size = child.stat().st_size
                except Exception:
                    stat_size = None
            entries.append(
                DirectoryEntry(
                    name=child.name,
                    path=str(child.resolve()),
                    is_dir=child.is_dir(),
                    size=stat_size,
                )
            )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    resolved = base.resolve()
    parent_path = resolved.parent if resolved != resolved.parent else None
    return {
        "path": str(resolved),
        "parent": str(parent_path) if parent_path else None,
        "entries": [entry.model_dump() for entry in entries],
    }


@app.post("/api/preview/{index}")
def api_preview(index: int, background_tasks: BackgroundTasks) -> FileResponse:
    STATE.ensure_file_loaded()
    text = STATE.get_chapter_text(index)
    if not text or len(text.strip()) < 5:
        raise HTTPException(status_code=400, detail="Nothing to preview for this chapter")
    tmp_handle = NamedTemporaryFile(suffix=".wav", delete=False)
    tmp = Path(tmp_handle.name)
    tmp_handle.close()
    try:
        core.gen_text(
            text=text[:2000],  # limit preview length for responsiveness
            voice=STATE.voice,
            output_file=str(tmp),
            speed=STATE.speed,
            play=False,
            backend=_resolve_backend(STATE.backend),
            mlx_model=STATE.mlx_model,
        )
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise

    def cleanup() -> None:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

    background_tasks.add_task(cleanup)
    return FileResponse(str(tmp), media_type="audio/wav", filename=f"preview_{index}.wav", background=background_tasks)


# ------------------------------ HTML Front-End ------------------------------


INDEX_HTML = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Audiblez Web UI</title>
    <style>
        :root {
            color-scheme: light dark;
            --bg: #111;
            --fg: #f8f8f8;
            --accent: #38bdf8;
            --accent-dark: #0ea5e9;
            --muted: #64748b;
        }
        body {
            margin: 0;
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            padding: 1.5rem 1rem 1rem;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            box-shadow: 0 2px 12px rgba(15, 23, 42, 0.65);
        }
        header h1 {
            margin: 0;
            font-size: 1.75rem;
            letter-spacing: 0.02em;
        }
        header p {
            margin: 0.35rem 0 0;
            font-size: 0.95rem;
            color: var(--muted);
        }
        main {
            flex: 1;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1rem;
            padding: 1rem;
        }
        section {
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-radius: 14px;
            padding: 1rem 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.35);
        }
        section h2 {
            margin: 0;
            font-size: 1.1rem;
            color: #e0e7ff;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .field {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        label {
            font-size: 0.85rem;
            color: var(--muted);
        }
        input, select, button, textarea {
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            padding: 0.55rem 0.65rem;
            background: rgba(15, 23, 42, 0.6);
            color: #e2e8f0;
            font-size: 0.95rem;
        }
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.3);
        }
        button.primary {
            background: var(--accent);
            border: 1px solid transparent;
            color: #0f172a;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        button.primary:disabled {
            background: rgba(56, 189, 248, 0.35);
            cursor: not-allowed;
        }
        button.primary:hover:enabled {
            transform: translateY(-1px);
            box-shadow: 0 12px 20px rgba(14, 165, 233, 0.35);
        }
        button.secondary {
            background: rgba(148, 163, 184, 0.15);
            border-color: rgba(148, 163, 184, 0.2);
            color: #cbd5f5;
            cursor: pointer;
        }
        .grid-two {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.6rem;
        }
        .chapter-list {
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
            max-height: 55vh;
            overflow: auto;
            padding-right: 0.4rem;
        }
        .chapter-card {
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 12px;
            padding: 0.65rem 0.75rem;
            display: grid;
            grid-template-columns: auto 1fr auto;
            gap: 0.6rem;
            align-items: center;
            background: rgba(30, 41, 59, 0.65);
        }
        .chapter-card h3 {
            margin: 0;
            font-size: 1rem;
            color: #f8fafc;
        }
        .chapter-meta {
            font-size: 0.8rem;
            color: var(--muted);
        }
        .chapter-actions {
            display: flex;
            gap: 0.4rem;
        }
        .tag {
            background: rgba(15, 118, 110, 0.35);
            color: #5eead4;
            font-size: 0.75rem;
            padding: 0.2rem 0.4rem;
            border-radius: 999px;
        }
        .status {
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }
        progress {
            width: 100%;
            height: 1rem;
        }
        .event-log {
            max-height: 200px;
            overflow: auto;
            border-radius: 10px;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.15);
            font-size: 0.75rem;
            padding: 0.5rem 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }
        .event-log span {
            color: var(--muted);
        }
        dialog::backdrop {
            background: rgba(15, 23, 42, 0.8);
        }
        dialog {
            border: none;
            border-radius: 14px;
            padding: 0;
            max-width: min(800px, 90vw);
            max-height: 80vh;
        }
        dialog .dialog-content {
            display: flex;
            flex-direction: column;
            background: rgba(15, 23, 42, 0.96);
            color: #e2e8f0;
            padding: 1rem 1.5rem;
            gap: 0.75rem;
        }
        dialog textarea {
            min-height: 45vh;
            font-family: "SFMono-Regular", "JetBrains Mono", monospace;
            resize: vertical;
        }
        @media (max-width: 768px) {
            main {
                grid-template-columns: 1fr;
            }
            section {
                padding: 0.9rem;
            }
            .chapter-card {
                grid-template-columns: 1fr;
            }
            .chapter-actions {
                justify-content: flex-start;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>Audiblez Web Control</h1>
        <p>Convert EPUB/PDF books into audiobooks using your browser. Works great on desktop and mobile.</p>
    </header>
    <main>
        <section id=\"file-section\">
            <h2>📚 Source</h2>
            <div class=\"field\">
                <label>Document</label>
                <div style=\"display:flex;gap:0.5rem;align-items:center;\">
                    <input id=\"file-path\" placeholder=\"Select an EPUB or PDF...\" readonly />
                    <button class=\"secondary\" id=\"browse-file\">Browse</button>
                </div>
            </div>
            <div class=\"field\">
                <label>Output Folder</label>
                <div style=\"display:flex;gap:0.5rem;align-items:center;\">
                    <input id=\"output-path\" placeholder=\"Choose output directory...\" readonly />
                    <button class=\"secondary\" id=\"browse-output\">Browse</button>
                </div>
            </div>
            <div class=\"field\" id=\"pdf-margins\" style=\"display:none;\">
                <label>PDF Margins (0–0.3)</label>
                <div class=\"grid-two\">
                    <div><input type=\"number\" step=\"0.01\" id=\"margin-header\" min=\"0\" max=\"0.3\" /></div>
                    <div><input type=\"number\" step=\"0.01\" id=\"margin-footer\" min=\"0\" max=\"0.3\" /></div>
                    <div><input type=\"number\" step=\"0.01\" id=\"margin-left\" min=\"0\" max=\"0.3\" /></div>
                    <div><input type=\"number\" step=\"0.01\" id=\"margin-right\" min=\"0\" max=\"0.3\" /></div>
                </div>
                <button class=\"secondary\" id=\"apply-margins\">Re-extract with margins</button>
            </div>
            <div class=\"field\">
                <label>Detected Device</label>
                <input id=\"device\" readonly />
            </div>
        </section>

        <section>
            <h2>🎛️ Voice & Backend</h2>
            <div class=\"field\">
                <label>Voice</label>
                <select id=\"voice\"></select>
            </div>
            <div class=\"field\">
                <label>Backend</label>
                <div style=\"display:flex;gap:0.4rem;\">
                    <label><input type=\"radio\" name=\"backend\" value=\"auto\" checked /> Auto</label>
                    <label><input type=\"radio\" name=\"backend\" value=\"mlx\" /> MLX</label>
                    <label><input type=\"radio\" name=\"backend\" value=\"kokoro\" /> Kokoro</label>
                </div>
            </div>
            <div class=\"field\">
                <label>MLX Model</label>
                <input id=\"mlx-model\" />
            </div>
            <div class=\"field\">
                <label>Speed (0.5 – 2.0)</label>
                <input type=\"number\" step=\"0.05\" id=\"speed\" min=\"0.5\" max=\"2\" />
            </div>
            <div class=\"grid-two\">
                <div class=\"field\"><label>Min tokens</label><input type=\"number\" id=\"gold-min\" /></div>
                <div class=\"field\"><label>Ideal tokens</label><input type=\"number\" id=\"gold-ideal\" /></div>
                <div class=\"field\"><label>Max tokens</label><input type=\"number\" id=\"gold-max\" /></div>
            </div>
        </section>

        <section style=\"grid-column: 1 / -1;\">
            <h2>🗂️ Chapters & Pages</h2>
            <div style=\"display:flex;gap:0.5rem;flex-wrap:wrap;\">
                <button class=\"secondary\" id=\"select-all\">Select All</button>
                <button class=\"secondary\" id=\"select-none\">Select None</button>
                <button class=\"secondary\" id=\"auto-select\">Auto Select</button>
                <div style=\"display:flex;gap:0.4rem;align-items:center;\">
                    <input type=\"number\" id=\"min-chars\" placeholder=\"Min characters\" style=\"width:140px;\" />
                    <button class=\"secondary\" id=\"apply-min-chars\">Apply</button>
                </div>
                <span id=\"selection-count\" class=\"tag\">0 selected</span>
            </div>
            <div class=\"chapter-list\" id=\"chapter-list\"></div>
        </section>

        <section class=\"status\" style=\"grid-column: 1 / -1;\">
            <h2>🚀 Synthesis</h2>
            <button class=\"primary\" id=\"start-job\">Start Synthesis</button>
            <div>
                <label style=\"display:flex;justify-content:space-between;\">
                    <span>Status</span>
                    <span id=\"job-status\">Idle</span>
                </label>
                <progress id=\"job-progress\" value=\"0\" max=\"100\"></progress>
                <div style=\"display:flex;justify-content:space-between;font-size:0.8rem;color:var(--muted);\">
                    <span id=\"job-chapter\"></span>
                    <span id=\"job-eta\"></span>
                </div>
            </div>
            <div class=\"event-log\" id=\"event-log\"></div>
        </section>
    </main>

    <dialog id=\"dialog-browser\">
        <div class=\"dialog-content\">
            <h3 style=\"margin:0;\">Browse</h3>
            <input id=\"browser-path\" />
            <div id=\"browser-entries\" style=\"display:flex;flex-direction:column;gap:0.35rem;max-height:45vh;overflow:auto;\"></div>
            <div style=\"display:flex;justify-content:flex-end;gap:0.5rem;\">
                <button class=\"secondary\" id=\"browser-cancel\">Cancel</button>
                <button class=\"primary\" id=\"browser-choose\">Choose</button>
            </div>
        </div>
    </dialog>

    <dialog id=\"dialog-text\">
        <div class=\"dialog-content\">
            <h3 id=\"dialog-text-title\" style=\"margin:0;\"></h3>
            <textarea id=\"dialog-text-body\" readonly></textarea>
            <div style=\"display:flex;justify-content:flex-end;gap:0.5rem;\">
                <button class=\"secondary\" id=\"dialog-text-close\">Close</button>
                <audio id=\"dialog-audio\" controls style=\"display:none;\"></audio>
            </div>
        </div>
    </dialog>

    <script>
        const state = {
            voices: [],
            chapters: [],
            selection: new Set(),
            jobPoll: null,
            browsing: { mode: null, path: null }
        };

        function qs(id) { return document.getElementById(id); }

        async function fetchJSON(url, options = {}) {
            const res = await fetch(url, {
                headers: { 'Content-Type': 'application/json' },
                ...options
            });
            if (!res.ok) {
                const detail = await res.json().catch(() => ({}));
                throw new Error(detail.detail || res.statusText);
            }
            return res.json();
        }

        function renderVoices(voiceGroups, current) {
            const select = qs('voice');
            select.innerHTML = '';
            voiceGroups.forEach(group => {
                const optGroup = document.createElement('optgroup');
                optGroup.label = `${group.flag ? group.flag + ' ' : ''}${group.group.toUpperCase()}`;
                group.voices.forEach(v => {
                    const option = document.createElement('option');
                    option.value = v;
                    option.textContent = v;
                    if (v === current) option.selected = true;
                    optGroup.appendChild(option);
                });
                select.appendChild(optGroup);
            });
        }

        function renderChapters(chapters) {
            const list = qs('chapter-list');
            list.innerHTML = '';
            chapters.forEach(ch => {
                const card = document.createElement('div');
                card.className = 'chapter-card';

                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.checked = ch.selected;
                checkbox.addEventListener('change', () => handleSelectionChange(ch.index, checkbox.checked));
                card.appendChild(checkbox);

                const info = document.createElement('div');
                const title = document.createElement('h3');
                title.textContent = ch.name;
                info.appendChild(title);
                const meta = document.createElement('div');
                meta.className = 'chapter-meta';
                meta.textContent = `${ch.characters.toLocaleString()} chars • ${ch.words.toLocaleString()} words`;
                info.appendChild(meta);
                const preview = document.createElement('div');
                preview.className = 'chapter-meta';
                preview.textContent = ch.preview;
                info.appendChild(preview);
                card.appendChild(info);

                const actions = document.createElement('div');
                actions.className = 'chapter-actions';
                const viewBtn = document.createElement('button');
                viewBtn.className = 'secondary';
                viewBtn.textContent = 'View';
                viewBtn.addEventListener('click', () => openChapterText(ch));
                const previewBtn = document.createElement('button');
                previewBtn.className = 'secondary';
                previewBtn.textContent = 'Preview';
                previewBtn.addEventListener('click', () => previewChapterAudio(ch.index));
                actions.appendChild(viewBtn);
                actions.appendChild(previewBtn);
                card.appendChild(actions);

                list.appendChild(card);
            });
        }

        function updateSelectionBadge(count) {
            const badge = qs('selection-count');
            badge.textContent = `${count} selected`;
        }

        function updateJob(info) {
            qs('job-status').textContent = info.status.toUpperCase();
            qs('job-progress').value = info.progress || 0;
            qs('job-eta').textContent = info.eta ? `ETA: ${info.eta}` : '';
            qs('job-chapter').textContent = info.current_chapter !== null ? `Working on chapter #${info.current_chapter + 1}` : '';
            const log = qs('event-log');
            log.innerHTML = '';
            info.events.slice(-50).forEach(e => {
                const row = document.createElement('div');
                row.innerHTML = `<strong>${e.event}</strong> <span>${e.timestamp}</span>`;
                if (e.payload && Object.keys(e.payload).length) {
                    row.innerHTML += `<div>${JSON.stringify(e.payload)}</div>`;
                }
                log.appendChild(row);
            });
            qs('start-job').disabled = info.status === 'running';
        }

        async function refreshState() {
            try {
                const data = await fetchJSON('/api/state');
                state.voices = data.voices;
                state.chapters = data.chapters;
                state.selection = new Set(data.chapters.filter(ch => ch.selected).map(ch => ch.index));
                renderVoices(data.voices, data.voice);
                renderChapters(data.chapters);
                updateSelectionBadge(data.selected_count);
                qs('file-path').value = data.file.path || '';
                qs('output-path').value = data.output_dir || '';
                qs('device').value = data.device || '';
                document.querySelectorAll('input[name="backend"]').forEach(r => {
                    r.checked = r.value === data.backend;
                });
                qs('mlx-model').value = data.mlx_model || '';
                qs('speed').value = data.speed || 1.0;
                qs('gold-min').value = data.gold_min || 10;
                qs('gold-ideal').value = data.gold_ideal || 25;
                qs('gold-max').value = data.gold_max || 40;
                qs('job-progress').value = data.job.progress || 0;
                updateJob(data.job);

                const m = data.margins || {};
                qs('margin-header').value = m.header ?? 0.07;
                qs('margin-footer').value = m.footer ?? 0.07;
                qs('margin-left').value = m.left ?? 0.07;
                qs('margin-right').value = m.right ?? 0.07;
                qs('pdf-margins').style.display = data.file.is_pdf ? 'flex' : 'none';

                if (state.jobPoll) clearInterval(state.jobPoll);
                if (data.job.status === 'running') {
                    state.jobPoll = setInterval(async () => {
                        try {
                            const info = await fetchJSON('/api/job');
                            updateJob(info);
                            if (info.status !== 'running') {
                                clearInterval(state.jobPoll);
                                refreshState();
                            }
                        } catch (err) {
                            console.error(err);
                            clearInterval(state.jobPoll);
                        }
                    }, 2000);
                }
            } catch (err) {
                alert(err.message);
            }
        }

        async function handleSelectionChange(index, enabled) {
            const selected = new Set(state.selection);
            if (enabled) selected.add(index); else selected.delete(index);
            try {
                const data = await fetchJSON('/api/chapters/selection', {
                    method: 'POST',
                    body: JSON.stringify({ selected: Array.from(selected) })
                });
                state.selection = new Set(data.chapters.filter(ch => ch.selected).map(ch => ch.index));
                renderChapters(data.chapters);
                updateSelectionBadge(data.selected_count);
            } catch (err) {
                alert(err.message);
            }
        }

        async function openChapterText(chapter) {
            try {
                const data = await fetchJSON(`/api/chapters/${chapter.index}/text`);
                qs('dialog-text-title').textContent = chapter.name;
                qs('dialog-text-body').value = data.text;
                qs('dialog-audio').style.display = 'none';
                const dialog = qs('dialog-text');
                dialog.showModal();
            } catch (err) {
                alert(err.message);
            }
        }

        async function previewChapterAudio(index) {
            try {
                const dialog = qs('dialog-text');
                const audio = qs('dialog-audio');
                audio.style.display = 'none';
                const res = await fetch(`/api/preview/${index}`, { method: 'POST' });
                if (!res.ok) throw new Error('Preview failed');
                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                audio.src = url;
                audio.style.display = 'block';
                if (!dialog.open) dialog.showModal();
                audio.play();
            } catch (err) {
                alert(err.message);
            }
        }

        async function updateSettings(partial) {
            try {
                const data = await fetchJSON('/api/settings', { method: 'POST', body: JSON.stringify(partial) });
                state.selection = new Set(data.chapters.filter(ch => ch.selected).map(ch => ch.index));
                renderChapters(data.chapters);
                updateSelectionBadge(data.selected_count);
            } catch (err) {
                alert(err.message);
            }
        }

        async function triggerSelectAll(flag) {
            try {
                const data = await fetchJSON(`/api/chapters/select-all?enable=${flag}`, { method: 'POST' });
                state.selection = new Set(data.chapters.filter(ch => ch.selected).map(ch => ch.index));
                renderChapters(data.chapters);
                updateSelectionBadge(data.selected_count);
            } catch (err) {
                alert(err.message);
            }
        }

        async function triggerAutoSelect() {
            try {
                const data = await fetchJSON('/api/chapters/select-auto', { method: 'POST' });
                state.selection = new Set(data.chapters.filter(ch => ch.selected).map(ch => ch.index));
                renderChapters(data.chapters);
                updateSelectionBadge(data.selected_count);
            } catch (err) {
                alert(err.message);
            }
        }

        async function triggerMinChars() {
            const minimum = Number(qs('min-chars').value || 0);
            try {
                const data = await fetchJSON('/api/chapters/select-min', { method: 'POST', body: JSON.stringify({ minimum }) });
                state.selection = new Set(data.chapters.filter(ch => ch.selected).map(ch => ch.index));
                renderChapters(data.chapters);
                updateSelectionBadge(data.selected_count);
            } catch (err) {
                alert(err.message);
            }
        }

        async function startJob() {
            qs('start-job').disabled = true;
            try {
                const info = await fetchJSON('/api/job', { method: 'POST', body: JSON.stringify({}) });
                updateJob(info);
                state.jobPoll = setInterval(async () => {
                    try {
                        const update = await fetchJSON('/api/job');
                        updateJob(update);
                        if (update.status !== 'running') {
                            clearInterval(state.jobPoll);
                            refreshState();
                        }
                    } catch (err) {
                        console.error(err);
                        clearInterval(state.jobPoll);
                    }
                }, 2000);
            } catch (err) {
                alert(err.message);
                qs('start-job').disabled = false;
            }
        }

        function setupListeners() {
            qs('voice').addEventListener('change', e => updateSettings({ voice: e.target.value }));
            document.querySelectorAll('input[name="backend"]').forEach(radio => {
                radio.addEventListener('change', e => {
                    if (e.target.checked) updateSettings({ backend: e.target.value });
                });
            });
            ['mlx-model','speed','gold-min','gold-ideal','gold-max'].forEach(id => {
                qs(id).addEventListener('change', e => {
                    const payload = {};
                    const key = id.replace('-', '_');
                    const raw = e.target.value;
                    const numeric = Number(raw);
                    payload[key] = Number.isFinite(numeric) && raw !== '' ? numeric : raw;
                    updateSettings(payload);
                });
            });
            qs('select-all').addEventListener('click', () => triggerSelectAll(true));
            qs('select-none').addEventListener('click', () => triggerSelectAll(false));
            qs('auto-select').addEventListener('click', triggerAutoSelect);
            qs('apply-min-chars').addEventListener('click', triggerMinChars);
            qs('start-job').addEventListener('click', startJob);
            qs('dialog-text-close').addEventListener('click', () => qs('dialog-text').close());

            qs('browse-file').addEventListener('click', () => openBrowser('file'));
            qs('browse-output').addEventListener('click', () => openBrowser('output'));
            qs('browser-cancel').addEventListener('click', closeBrowser);
            qs('browser-choose').addEventListener('click', chooseBrowserSelection);
            qs('apply-margins').addEventListener('click', applyMargins);
        }

        async function openBrowser(mode) {
            state.browsing.mode = mode;
            const initial = mode === 'file' ? qs('file-path').value || '' : qs('output-path').value || '';
            await loadBrowser(initial || '/');
            qs('dialog-browser').showModal();
        }

        function closeBrowser() {
            qs('dialog-browser').close();
        }

        async function loadBrowser(path) {
            try {
                const params = new URLSearchParams({ path });
                if (state.browsing.mode === 'file') params.append('patterns', '*.epub,*.pdf');
                const data = await fetchJSON(`/api/fs?${params.toString()}`);
                qs('browser-path').value = data.path;
                state.browsing.path = data.path;
                const entries = qs('browser-entries');
                entries.innerHTML = '';
                if (data.parent) {
                    const parent = document.createElement('button');
                    parent.className = 'secondary';
                    parent.textContent = '⬆︎ Parent directory';
                    parent.addEventListener('click', () => loadBrowser(data.parent));
                    entries.appendChild(parent);
                }
                data.entries.forEach(entry => {
                    const btn = document.createElement('button');
                    btn.className = 'secondary';
                    btn.textContent = `${entry.is_dir ? '📁' : '📄'} ${entry.name}`;
                    btn.style.display = 'flex';
                    btn.style.justifyContent = 'space-between';
                    btn.addEventListener('click', () => {
                        if (entry.is_dir) {
                            loadBrowser(entry.path);
                        } else {
                            qs('browser-path').value = entry.path;
                        }
                    });
                    entries.appendChild(btn);
                });
            } catch (err) {
                alert(err.message);
            }
        }

        async function chooseBrowserSelection() {
            const path = qs('browser-path').value;
            if (!path) return;
            if (state.browsing.mode === 'file') {
                try {
                    const data = await fetchJSON('/api/file', {
                        method: 'POST',
                        body: JSON.stringify({ path })
                    });
                    closeBrowser();
                    await refreshState();
                } catch (err) {
                    alert(err.message);
                }
            } else {
                try {
                    await fetchJSON('/api/output', {
                        method: 'POST',
                        body: JSON.stringify({ path })
                    });
                    qs('output-path').value = path;
                    closeBrowser();
                } catch (err) {
                    alert(err.message);
                }
            }
        }

        async function applyMargins() {
            try {
                const payload = {
                    header: Number(qs('margin-header').value || 0.07),
                    footer: Number(qs('margin-footer').value || 0.07),
                    left: Number(qs('margin-left').value || 0.07),
                    right: Number(qs('margin-right').value || 0.07)
                };
                const data = await fetchJSON('/api/margins', { method: 'POST', body: JSON.stringify(payload) });
                state.selection = new Set(data.chapters.filter(ch => ch.selected).map(ch => ch.index));
                renderChapters(data.chapters);
                updateSelectionBadge(data.selected_count);
            } catch (err) {
                alert(err.message);
            }
        }

        document.addEventListener('DOMContentLoaded', () => {
            setupListeners();
            refreshState();
        });
    </script>
</body>
</html>
"""


# ------------------------------ CLI Entry ------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Start the audiblez web UI server")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765)")
    parser.add_argument("--open", action="store_true", help="Automatically open the UI in the default browser")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.open:
        import webbrowser

        url = f"http://{args.host}:{args.port}/"

        def _open() -> None:
            time.sleep(0.5)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
