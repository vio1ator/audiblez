#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal UI (TUI) for audiblez.

Features:
- Browse for EPUB/PDF via a simple picker-based file browser.
- Choose output folder.
- Select voice, backend, and speed. Device is auto-selected (CUDA/MPS/CPU).
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
from typing import List, Optional, Tuple, Set
import shutil

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
        choice, _ = select_with_letter_jump(options, title, jump_to_folders=True)
        if choice is None:
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
        choice, _ = select_with_letter_jump(options, title, jump_to_folders=True)
        if choice is None:
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
    """Auto-select compute device without prompting the user.

    Preference order: CUDA > MPS > CPU.
    Silently no-ops if torch is unavailable.
    """
    try:
        import torch
    except Exception:
        return
    chosen = 'cpu'
    try:
        if torch.cuda.is_available():
            chosen = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            chosen = 'mps'
        torch.set_default_device(chosen)
        print(f"Using device: {chosen}")
    except Exception:
        # If anything goes wrong, just continue on default device.
        pass


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
    lengths = []
    for c in chapters:
        name = c.get_name()
        prev = chapter_beginning_one_liner(c, 50)
        labels.append(f"{name} ({len(c.extracted_text)} chars) [{prev}]")
        lengths.append(len(c.extracted_text))
    # Preselect good chapters
    initial = {i for i, c in enumerate(chapters) if c in good}
    title = "Select chapters: Space toggle • a:All • n:None • t:Min chars • Enter accept • q quit"
    idxs = multiselect_with_controls(labels, title, initial_selected=initial, lengths=lengths)
    selected_objs = [chapters[i] for i in idxs]
    return chapters, selected_objs


def _select_chapters_for_pdf(pages) -> List:
    labels = []
    lengths = []
    for p in pages:
        prev = (p.extracted_text or "").strip().splitlines()[0:1]
        prev = prev[0][:50] + ('...' if len(prev[0]) > 50 else '') if prev else ''
        labels.append(f"{p.get_name()} ({len(p.extracted_text)} chars) [{prev}]")
        lengths.append(len(p.extracted_text or ""))
    # Default: select all pages initially
    initial = set(range(len(pages)))
    title = "Select pages: Space toggle • a:All • n:None • t:Min chars • Enter accept • q quit"
    idxs = multiselect_with_controls(labels, title, initial_selected=initial, lengths=lengths)
    selected_pages = [p for i, p in enumerate(pages) if i in idxs]
    return pages, selected_pages


def multiselect_with_controls(options: List[str], title: str, initial_selected: Optional[Set[int]] = None, lengths: Optional[List[int]] = None) -> List[int]:
    """Curses multiselect with select-all/none and min-length selection.

    Keys: arrows/jk move, space toggle, a select all, n select none, t set min chars,
    letters jump, Enter accept (requires >=1), q cancel.
    Returns selected indices list.
    """
    try:
        import curses
    except Exception:
        # Fallback to pick's multiselect (without advanced shortcuts)
        try:
            selected = pick(options, title, multiselect=True, min_selection_count=1)
            return [i for (_, i) in selected]
        except Exception as e2:
            print("Selection failed:", e2)
            raise

    initial_selected = set(initial_selected or set())

    def _prompt_threshold(stdscr) -> Optional[int]:
        h, w = stdscr.getmaxyx()
        prompt = "Min chars: "
        buf = ""
        while True:
            stdscr.move(h - 1, 0)
            stdscr.clrtoeol()
            try:
                stdscr.addstr(h - 1, 0, prompt + buf)
            except curses.error:
                pass
            ch = stdscr.getch()
            if ch in (10, 13):
                try:
                    return int(buf) if buf else None
                except Exception:
                    return None
            elif ch in (27,):
                return None
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                buf = buf[:-1]
            elif 48 <= ch <= 57:  # digits
                if len(buf) < 9:
                    buf += chr(ch)

    def _run(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        idx = 0
        top = 0
        selected: Set[int] = set(initial_selected)
        message = ""
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            vis_h = max(1, h - 4)
            # Adjust scrolling window
            if idx < top:
                top = idx
            elif idx >= top + vis_h:
                top = idx - vis_h + 1

            # Title and help
            title_str = (title or "")[: max(0, w - 1)]
            try:
                stdscr.addstr(0, 0, title_str, curses.A_BOLD)
            except curses.error:
                pass
            hint = "Space toggle • a All • n None • t Min chars • Enter accept • q quit"
            try:
                stdscr.addstr(1, 0, hint[: max(0, w - 1)])
            except curses.error:
                pass
            status = f"Selected {len(selected)}/{len(options)}"
            try:
                stdscr.addstr(2, 0, status[: max(0, w - 1)])
            except curses.error:
                pass

            # Render list
            for row in range(vis_h):
                j = top + row
                if j >= len(options):
                    break
                is_sel = j in selected
                checkbox = "[x]" if is_sel else "[ ]"
                line = options[j]
                text = f"{checkbox} {line}"
                try:
                    stdscr.addstr(3 + row, 0, text[: max(0, w - 1)], curses.A_REVERSE if j == idx else curses.A_NORMAL)
                except curses.error:
                    pass

            if message:
                try:
                    stdscr.addstr(h - 2, 0, message[: max(0, w - 1)], curses.A_DIM)
                except curses.error:
                    pass

            stdscr.refresh()

            key = stdscr.getch()
            message = ""
            if key in (curses.KEY_UP, ord('k')):
                idx = max(0, idx - 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                idx = min(len(options) - 1, idx + 1)
            elif key == curses.KEY_PPAGE:
                idx = max(0, idx - vis_h)
            elif key == curses.KEY_NPAGE:
                idx = min(len(options) - 1, idx + vis_h)
            elif key in (32,):  # space
                if idx in selected:
                    selected.remove(idx)
                else:
                    selected.add(idx)
            elif key in (ord('a'),):
                selected = set(range(len(options)))
            elif key in (ord('n'),):
                selected.clear()
            elif key in (ord('t'),):
                thr = _prompt_threshold(stdscr)
                if thr is not None and lengths is not None:
                    selected = {i for i, L in enumerate(lengths) if L >= thr}
                elif thr is None:
                    message = "Threshold canceled"
                else:
                    message = "No lengths available"
            elif key in (10, 13, curses.KEY_ENTER):
                if len(selected) >= 1:
                    return sorted(selected)
                else:
                    message = "Select at least 1 item"
            elif key in (27, ord('q')):
                return None
            elif 32 <= key <= 126:
                ch = chr(key)
                # Jump to first entry starting with letter
                for k, s in enumerate(options):
                    if s.lower().startswith(ch.lower()):
                        idx = k
                        break

    res = __import__('curses').wrapper(_run)
    if res is None:
        raise RuntimeError("Selection canceled")
    return res


def _preview_loop(chapters, voice: str, speed: float, backend: str, mlx_model: str):
    # Allow previewing multiple times until user selects "Done".
    while True:
        opts = [f"{i+1}. {getattr(c, 'get_name', lambda: f'Chapter {i+1}')()}" for i, c in enumerate(chapters)]
        opts.append("[Done]")
        try:
            choice, idx = select_with_letter_jump(opts, "Select a chapter/page to preview", jump_to_folders=False)
        except Exception as e:
            print("Preview selection failed:", e)
            return
        if choice is None:
            # User canceled (q/Esc). Exit preview loop gracefully.
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
            core.gen_text(text, voice=voice, output_file=tmp.name, speed=speed, play=False, backend=backend, mlx_model=mlx_model)
            _play_audio_blocking(tmp.name)
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
    do_prev, _ = select_with_letter_jump(["Yes", "No"], "Preview chapters/pages before starting?", jump_to_folders=False)
    if do_prev is None:
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


def select_with_letter_jump(options: List[str], title: str, jump_to_folders: bool = True, start_index: int = 0):
    """Curses-based selector with first-letter jump to folder.

    Returns (choice, index) or (None, None) on cancel/failure.
    """
    try:
        import curses
    except Exception as e:
        # Fallback to pick if curses unavailable
        try:
            return pick(options, title, multiselect=False, min_selection_count=1)
        except Exception as e2:
            print("Selection failed:", e2)
            return None, None

    def _run(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        idx = min(max(0, start_index), max(0, len(options) - 1))
        top = 0
        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            vis_h = max(1, h - 3)
            # Adjust scrolling window
            if idx < top:
                top = idx
            elif idx >= top + vis_h:
                top = idx - vis_h + 1

            # Render
            title_str = (title or "")[: max(0, w - 1)]
            try:
                stdscr.addstr(0, 0, title_str, curses.A_BOLD)
            except curses.error:
                pass
            hint = (
                "Arrows/jk move • letters jump to folder • Enter select • q quit"
                if jump_to_folders else
                "Arrows/jk move • letters jump • Enter select • q quit"
            )
            try:
                stdscr.addstr(1, 0, hint[: max(0, w - 1)])
            except curses.error:
                pass
            for row in range(vis_h):
                j = top + row
                if j >= len(options):
                    break
                line = options[j]
                mark = ">" if j == idx else " "
                text = f"{mark} {line}"
                try:
                    stdscr.addstr(2 + row, 0, text[: max(0, w - 1)], curses.A_REVERSE if j == idx else curses.A_NORMAL)
                except curses.error:
                    pass
            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_UP, ord('k')):
                idx = max(0, idx - 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                idx = min(len(options) - 1, idx + 1)
            elif key == curses.KEY_PPAGE:
                idx = max(0, idx - vis_h)
            elif key == curses.KEY_NPAGE:
                idx = min(len(options) - 1, idx + vis_h)
            elif key in (10, 13, curses.KEY_ENTER):
                return idx
            elif key in (27, ord('q')):
                return None
            elif 32 <= key <= 126:  # printable ASCII
                ch = chr(key)
                # Prefer folders ending with '/'
                def starts_with(s: str, c: str) -> bool:
                    return s.lower().startswith(c.lower())

                match_idx = None
                if jump_to_folders:
                    for k, s in enumerate(options):
                        if s.endswith('/') and starts_with(s, ch):
                            match_idx = k
                            break
                if match_idx is None:
                    for k, s in enumerate(options):
                        if starts_with(s, ch):
                            match_idx = k
                            break
                if match_idx is not None:
                    idx = match_idx
            # loop continues

    try:
        res = __import__('curses').wrapper(_run)
    except Exception:
        try:
            return pick(options, title, multiselect=False, min_selection_count=1)
        except Exception as e:
            print("Selection failed:", e)
            return None, None
    if res is None:
        return None, None
    return options[res], res


def _play_audio_blocking(path: str):
    # Prefer ffplay if available, otherwise use platform-appropriate players.
    if shutil.which('ffplay'):
        subprocess.run(['ffplay', '-autoexit', '-nodisp', path])
        return
    if platform.system() == 'Darwin' and shutil.which('afplay'):
        subprocess.run(['afplay', path])
        return
    if platform.system() == 'Linux' and shutil.which('aplay'):
        subprocess.run(['aplay', path])
        return
    # Last resort: try vlc
    if shutil.which('cvlc'):
        subprocess.run(['cvlc', '--play-and-exit', path])
        return
    print('No suitable audio player found (ffplay/afplay/aplay/cvlc).')
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
