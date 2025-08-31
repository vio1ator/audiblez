#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal UI (TUI) for audiblez.

Features:
- Browse for EPUB/PDF via a simple picker-based file browser.
- Choose output folder.
- Select voice, backend, speed, and device (CPU/CUDA/MPS).
- For PDF, set header/footer/left/right margins (0–0.3).
- Select chapters/pages with multiselect and optional audio preview.
- Run synthesis with progress updates.

Requires the existing dependency `pick`.
"""
from __future__ import annotations

import os
import sys
import fnmatch
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple

import platform

from pick import pick

from .voices import voices as VOICES_BY_LANG, flags as FLAGS
from . import core


def _list_dir_for_picker(path: Path, patterns: List[str]) -> List[str]:
    items: List[str] = []
    # Navigation helpers
    items.append("..")
    # Directories
    for d in sorted([p for p in path.iterdir() if p.is_dir()]):
        items.append(d.name + "/")
    # Files matching patterns
    for f in sorted([p for p in path.iterdir() if p.is_file()]):
        if any(fnmatch.fnmatch(f.name.lower(), pat) for pat in patterns):
            items.append(f.name)
    return items


def _file_browser(start_dir: Path, patterns: List[str]) -> Optional[Path]:
    cur = start_dir.resolve()
    while True:
        title = f"Select file (current: {cur})"
        options = _list_dir_for_picker(cur, patterns)
        try:
            choice, _ = pick(options, title, multiselect=False, min_selection_count=1)
        except Exception as e:
            print("Selection failed:", e)
            return None
        if choice == "..":
            parent = cur.parent
            if parent == cur:
                continue
            cur = parent
            continue
        if choice.endswith("/"):
            cur = (cur / choice[:-1]).resolve()
            continue
        # File
        sel = (cur / choice).resolve()
        if sel.is_file():
            return sel


def _dir_browser(start_dir: Path) -> Optional[Path]:
    cur = start_dir.resolve()
    while True:
        title = f"Select output folder (current: {cur})"
        options = ["[Use this directory]"] + _list_dir_for_picker(cur, patterns=["*"])
        try:
            choice, _ = pick(options, title, multiselect=False, min_selection_count=1)
        except Exception as e:
            print("Selection failed:", e)
            return None
        if choice == "[Use this directory]":
            return cur
        if choice == "..":
            parent = cur.parent
            if parent == cur:
                continue
            cur = parent
            continue
        if choice.endswith("/"):
            cur = (cur / choice[:-1]).resolve()
            continue
        # Ignore file selection here


def _choose_voice() -> str:
    items: List[str] = []
    for code, vlist in VOICES_BY_LANG.items():
        for v in vlist:
            items.append(f"{FLAGS.get(code, '')} {v}")
    title = "Select voice"
    choice, _ = pick(items, title, multiselect=False, min_selection_count=1)
    return choice.split(" ", 1)[1]


def _choose_backend() -> Tuple[str, Optional[str]]:
    default_model = 'mlx-community/Kokoro-82M-8bit'
    options = ["auto", "mlx", "kokoro"]
    backend, _ = pick(options, "Select TTS backend", multiselect=False, min_selection_count=1)
    mlx_model = default_model
    if backend == "mlx":
        # Offer model override via simple text prompt
        print(f"Enter MLX model id (default: {default_model}): ", end="", flush=True)
        line = sys.stdin.readline().strip()
        if line:
            mlx_model = line
    return backend, mlx_model


def _choose_device():
    try:
        import torch
    except Exception:
        print("Torch not available; using CPU.")
        return
    options = ["CPU", "CUDA", "MPS"]
    dev, _ = pick(options, "Select compute device", multiselect=False, min_selection_count=1)
    chosen = None
    if dev == "CUDA" and torch.cuda.is_available():
        chosen = 'cuda'
    elif dev == "MPS" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        chosen = 'mps'
    else:
        chosen = 'cpu'
    try:
        torch.set_default_device(chosen)
        print(f"Using device: {chosen}")
    except Exception:
        print(f"Failed to set device {chosen}; continuing.")


def _input_float(prompt: str, default: float, lo: float, hi: float) -> float:
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if not s:
            return default
        try:
            v = float(s)
            if lo <= v <= hi:
                return v
        except Exception:
            pass
        print(f"Enter a number in range [{lo}, {hi}].")


def _select_chapters_for_epub(book) -> List:
    from .core import find_document_chapters_and_extract_texts, find_good_chapters, chapter_beginning_one_liner
    chapters = find_document_chapters_and_extract_texts(book)
    good = find_good_chapters(chapters)
    labels = []
    by_label = {}
    for c in chapters:
        name = c.get_name()
        prev = chapter_beginning_one_liner(c, 50)
        key = f"{name} ({len(c.extracted_text)} chars) [{prev}]"
        by_label[key] = c
        labels.append(key)
    title = "Space to toggle, Enter to accept. Select chapters/pages"
    try:
        selected = pick(labels, title, multiselect=True, min_selection_count=1)
    except Exception:
        print("Picker failed to render. Try enlarging terminal or different terminal app.")
        raise
    selected_objs = [by_label[it[0]] for it in selected]
    # Preserve doc order
    selected_objs = [c for c in chapters if c in selected_objs]
    return chapters, selected_objs


def _select_chapters_for_pdf(pages) -> List:
    labels = []
    for p in pages:
        prev = (p.extracted_text or "").strip().splitlines()[0:1]
        prev = prev[0][:50] + ('...' if len(prev[0]) > 50 else '') if prev else ''
        labels.append(f"{p.get_name()} ({len(p.extracted_text)} chars) [{prev}]")
    title = "Space to toggle, Enter to accept. Select pages"
    try:
        selected = pick(labels, title, multiselect=True, min_selection_count=1)
    except Exception:
        print("Picker failed to render. Try enlarging terminal or different terminal app.")
        raise
    idxs = [i for (_, i) in selected]
    selected_pages = [p for i, p in enumerate(pages) if i in idxs]
    return pages, selected_pages


def _preview_loop(chapters, voice: str, speed: float, backend: str, mlx_model: str):
    # Allow previewing multiple times until user selects "Done".
    while True:
        opts = [f"{i+1}. {getattr(c, 'get_name', lambda: f'Chapter {i+1}')()}" for i, c in enumerate(chapters)]
        opts.append("[Done]")
        try:
            choice, idx = pick(opts, "Select a chapter/page to preview", multiselect=False, min_selection_count=1)
        except Exception as e:
            print("Preview selection failed:", e)
            return
        if choice == "[Done]":
            return
        chapter = chapters[idx]
        text = (chapter.extracted_text or "")[:300]
        if not text.strip():
            print("Chapter is empty – nothing to preview.")
            continue
        try:
            tmp = NamedTemporaryFile(suffix='.wav', delete=False)
            core.gen_text(text, voice=voice, output_file=tmp.name, speed=speed, play=True, backend=backend, mlx_model=mlx_model)
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
        except Exception as e:
            print("Preview failed:", e)


def _post_event(event_name, **kwargs):
    if event_name == 'CORE_STARTED':
        print("Synthesis started.")
    elif event_name == 'CORE_CHAPTER_STARTED':
        print(f"Chapter {kwargs.get('chapter_index')} started.")
    elif event_name == 'CORE_CHAPTER_FINISHED':
        print(f"Chapter {kwargs.get('chapter_index')} finished.")
    elif event_name == 'CORE_PROGRESS':
        stats = kwargs.get('stats')
        if stats:
            print(f"Progress: {stats.progress}% | ETA: {stats.eta} | Rate: {getattr(stats, 'chars_per_sec', 0):.0f} chars/s")
    elif event_name == 'CORE_FINISHED':
        print("Synthesis finished.")


def main():
    print("Audiblez TUI — generate audiobooks from EPUB/PDF\n")
    start_dir = Path.cwd()
    ebook = _file_browser(start_dir, patterns=["*.epub", "*.pdf"])
    if not ebook:
        print("No file selected. Exiting.")
        return

    out_dir = _dir_browser(start_dir) or Path('.')
    voice = _choose_voice()
    speed = _input_float("Speed (0.5–2.0)", 1.0, 0.5, 2.0)
    backend, mlx_model = _choose_backend()
    _choose_device()

    is_pdf = ebook.suffix.lower() == '.pdf'
    margins = dict(header=0.07, footer=0.07, left=0.07, right=0.07)
    if is_pdf:
        print("PDF detected. Set trimming margins (fractions 0–0.3). Press Enter to keep default 0.07.")
        margins['header'] = _input_float("Header", 0.07, 0.0, 0.3)
        margins['footer'] = _input_float("Footer", 0.07, 0.0, 0.3)
        margins['left'] = _input_float("Left", 0.07, 0.0, 0.3)
        margins['right'] = _input_float("Right", 0.07, 0.0, 0.3)

    # Build chapter list
    if is_pdf:
        from .pdf import extract_pages
        chapters = extract_pages(ebook, margins)
        all_chapters, selected = _select_chapters_for_pdf(chapters)
    else:
        from ebooklib import epub
        book = epub.read_epub(str(ebook))
        all_chapters, selected = _select_chapters_for_epub(book)

    # Offer previews
    try:
        do_prev, _ = pick(["Yes", "No"], "Preview chapters/pages before starting?", multiselect=False, min_selection_count=1)
    except Exception:
        do_prev = "No"
    if do_prev == "Yes":
        _preview_loop(selected, voice, speed, backend, mlx_model)

    print("Starting synthesis...\n")
    core.main(
        str(ebook), voice, pick_manually=False, speed=speed,
        output_folder=str(out_dir), selected_chapters=selected,
        backend=backend, mlx_model=mlx_model,
        header=margins.get('header', 0.07), footer=margins.get('footer', 0.07),
        left=margins.get('left', 0.07), right=margins.get('right', 0.07),
        post_event=_post_event,
    )


if __name__ == '__main__':
    main()
