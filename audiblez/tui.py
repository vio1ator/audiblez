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
    default_model = 'mlx-community/Kokoro-82M-bf16'
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


def _input_int(prompt: str, default: int, lo: int, hi: int) -> int:
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if not s:
            return default
        try:
            v = int(s)
            if lo <= v <= hi:
                return v
        except Exception:
            pass
        print(f"Enter an integer in range [{lo}, {hi}].")


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


def _flatten_voices() -> List[str]:
    res: List[str] = []
    for _, vlist in VOICES_BY_LANG.items():
        res.extend(vlist)
    return res


def _single_page_screen(ebook: Optional[Path],
                        out_dir: Path,
                        is_pdf: bool,
                        initial_margins: dict,
                        chapters_all: List,
                        initially_selected: List,
                        ) -> Tuple[str, dict]:
    """Run a single-page curses UI. Returns (action, state).

    action in {'start', 'preview', 'quit'}.
    state contains: voice, speed, backend, mlx_model, margins, token thresholds, selected_chapters.
    """
    voices_flat = _flatten_voices()

    state = {
        'voice': initially_selected and voices_flat[0] or voices_flat[0],
        'speed': 1.0,
        'backend': 'auto',
        'mlx_model': 'mlx-community/Kokoro-82M-bf16',
        'gold_min': 10,
        'gold_ideal': 25,
        'gold_max': 40,
        'margins': dict(initial_margins),
        'selected_set': set(range(len(chapters_all))) if is_pdf else {i for i, c in enumerate(chapters_all) if c in initially_selected},
        'ebook': ebook,
        'out_dir': out_dir,
    }

    # helper to format device (auto-selected)
    def detect_device_label() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
        except Exception:
            pass
        return 'cpu'

    device_label = detect_device_label()

    # Precompute chapter labels and lengths
    def build_labels_and_lengths():
        labels = []
        lengths = []
        if chapters_all and is_pdf:
            for p in chapters_all:
                prev = (p.extracted_text or '').strip().splitlines()[0:1]
                prev = prev[0][:50] + ('...' if len(prev[0]) > 50 else '') if prev else ''
                labels.append(f"{p.get_name()} ({len(p.extracted_text or '')} chars) [{prev}]")
                lengths.append(len(p.extracted_text or ''))
        elif chapters_all:
            from .core import chapter_beginning_one_liner
            for c in chapters_all:
                name = c.get_name()
                prev = chapter_beginning_one_liner(c, 50)
                labels.append(f"{name} ({len(c.extracted_text)} chars) [{prev}]")
                lengths.append(len(c.extracted_text))
        return labels, lengths

    labels, lengths = build_labels_and_lengths()

    # helpers to (re)load chapters when ebook changes
    def load_from_file(path: Path):
        nonlocal is_pdf, chapters_all, initially_selected, labels, lengths
        is_pdf = path.suffix.lower() == '.pdf'
        state['margins'] = dict(initial_margins)
        try:
            if is_pdf:
                from .pdf import extract_pages
                chapters_all = extract_pages(path, state['margins'])
                initially_selected = chapters_all
                state['selected_set'] = set(range(len(chapters_all)))
            else:
                from ebooklib import epub
                from .core import find_document_chapters_and_extract_texts, find_good_chapters
                book = epub.read_epub(str(path))
                ch = find_document_chapters_and_extract_texts(book)
                good = find_good_chapters(ch)
                chapters_all = ch
                initially_selected = good
                state['selected_set'] = {i for i, c in enumerate(chapters_all) if c in initially_selected}
        except Exception:
            chapters_all = []
            initially_selected = []
            state['selected_set'] = set()
        labels, lengths = build_labels_and_lengths()

    def run(stdscr):
        import curses
        curses.curs_set(0)
        stdscr.keypad(True)
        idx = 0
        top = 0
        message = ''
        voice_idx = 0
        
        def render_header(body_start_line: int = 5) -> int:
            """Render the top fixed header and return next line index."""
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            # Header
            file_str = str(state['ebook']) if state['ebook'] else '[press f to choose file]'
            out_str = str(state['out_dir'])
            header = f"Audiblez TUI — Single Page | File: {file_str} | Out: {out_str}"
            stdscr.addstr(0, 0, header[: max(0, w - 1)], curses.A_BOLD)
            # Parameters
            stdscr.addstr(1, 0, f"Voice [v/shift+v]: {state['voice']}")
            stdscr.addstr(2, 0, f"Backend [b]: {state['backend']}" + (f"  Model [m]: {state['mlx_model']}" if state['backend'] == 'mlx' else ''))
            stdscr.addstr(3, 0, f"Speed [-/=]: {state['speed']:.2f}    Device: {device_label}")
            stdscr.addstr(4, 0, f"Chunk tokens [1/2/3 +/-]: min/ideal/max = {state['gold_min']}/{state['gold_ideal']}/{state['gold_max']}")
            line_local = body_start_line
            if is_pdf:
                m = state['margins']
                stdscr.addstr(line_local, 0, f"PDF margins [H/F/L/R +/-]: header={m['header']:.02f} footer={m['footer']:.02f} left={m['left']:.02f} right={m['right']:.02f}")
                line_local += 1
            stdscr.addstr(line_local, 0, f"Chapters/pages — Space toggle • a All • n None • t Min chars • f File • d Output • p Preview • s Start • q Quit")
            return line_local + 1
        # If initial file not present, wait for user to pick
        while True:
            # Re-assert keypad mode in case a sub-view changed terminal mode
            stdscr.keypad(True)
            h, w = stdscr.getmaxyx()
            vis_h = max(3, h - 12)  # rows for list
            line = render_header()

            # Adjust scrolling window for list area starting at 'line'
            # show status
            selected = state['selected_set']
            total = len(labels)
            stdscr.addstr(line, 0, f"Selected {len(selected)}/{total}")
            line += 1

            if total > 0:
                if idx < top:
                    top = idx
                elif idx >= top + (h - line - 2):
                    top = max(0, idx - (h - line - 3))

            # Render list
            rows = max(1, h - line - 2)
            if total == 0:
                try:
                    stdscr.addstr(line, 0, 'No chapters/pages loaded. Press f to choose a file.')
                except curses.error:
                    pass
            else:
                for row in range(rows):
                    j = top + row
                    if j >= total:
                        break
                    is_sel = j in selected
                    checkbox = '[x]' if is_sel else '[ ]'
                    text = f"{checkbox} {labels[j]}"
                    try:
                        stdscr.addstr(line + row, 0, text[: max(0, w - 1)], curses.A_REVERSE if j == idx else curses.A_NORMAL)
                    except curses.error:
                        pass

            # Footer message
            if message:
                try:
                    stdscr.addstr(h - 1, 0, message[: max(0, w - 1)], curses.A_DIM)
                except curses.error:
                    pass
            stdscr.refresh()

            # Input
            key = stdscr.getch()
            message = ''

            # Navigation in list
            if key in (curses.KEY_UP, ord('k')) and len(labels) > 0:
                idx = max(0, idx - 1)
            elif key in (curses.KEY_DOWN, ord('j')) and len(labels) > 0:
                idx = min(len(labels) - 1, idx + 1)
            elif key == curses.KEY_PPAGE and len(labels) > 0:
                idx = max(0, idx - rows)
            elif key == curses.KEY_NPAGE and len(labels) > 0:
                idx = min(len(labels) - 1, idx + rows)
            # Toggle selection
            elif key == 32 and len(labels) > 0:  # space
                if idx in selected:
                    selected.remove(idx)
                else:
                    selected.add(idx)
            elif key in (ord('a'),):
                state['selected_set'] = set(range(len(labels)))
            elif key in (ord('n'),):
                state['selected_set'] = set()
                selected = state['selected_set']
            elif key in (ord('t'),):
                # Prompt line for threshold
                curses.echo()
                try:
                    stdscr.move(h - 1, 0)
                    stdscr.clrtoeol()
                    stdscr.addstr(h - 1, 0, 'Min chars: ')
                    thr_str = stdscr.getstr(h - 1, len('Min chars: '), 16).decode('utf-8')
                    thr = int(thr_str) if thr_str else None
                    if thr is not None and len(lengths) > 0:
                        state['selected_set'] = {i for i, L in enumerate(lengths) if L >= thr}
                        selected = state['selected_set']
                except Exception:
                    message = 'Invalid threshold'
                finally:
                    curses.noecho()
            # Voice cycle
            elif key in (ord('v'),):
                voice_idx = (voice_idx + 1) % len(voices_flat)
                state['voice'] = voices_flat[voice_idx]
            elif key in (ord('V'),):
                voice_idx = (voice_idx - 1) % len(voices_flat)
                state['voice'] = voices_flat[voice_idx]
            # Backend toggle
            elif key in (ord('b'),):
                order = ['auto', 'mlx', 'kokoro']
                cur = state['backend']
                state['backend'] = order[(order.index(cur) + 1) % len(order)] if cur in order else 'auto'
            # Model input if mlx
            elif key in (ord('m'),):
                if state['backend'] == 'mlx':
                    curses.echo()
                    try:
                        stdscr.move(h - 1, 0)
                        stdscr.clrtoeol()
                        stdscr.addstr(h - 1, 0, f"MLX model id [{state['mlx_model']}]: ")
                        s = stdscr.getstr(h - 1, len(f"MLX model id [{state['mlx_model']}]: "), 128).decode('utf-8').strip()
                        if s:
                            state['mlx_model'] = s
                    finally:
                        curses.noecho()
            # Speed adjust
            elif key in (ord('-'), ord('_')):
                state['speed'] = max(0.5, round(state['speed'] - 0.05, 2))
            elif key in (ord('='), ord('+')):
                state['speed'] = min(2.0, round(state['speed'] + 0.05, 2))
            # Tokens adjust (1=min, 2=ideal, 3=max) with +/-
            elif key in (ord('1'), ord('2'), ord('3')):
                # peek next char for +/-
                sub = {ord('1'): 'gold_min', ord('2'): 'gold_ideal', ord('3'): 'gold_max'}[key]
                stdscr.addstr(h - 1, 0, f"Adjust {sub} with +/- (Esc to cancel)")
                while True:
                    ch = stdscr.getch()
                    if ch in (27,):
                        break
                    if ch in (ord('-'), ord('_')):
                        state[sub] = max(10, state[sub] - 5)
                    elif ch in (ord('+'), ord('=')):
                        state[sub] = min(500, state[sub] + 5)
                    else:
                        break
                    # live refresh
                    line = render_header()
                    # Re-render status and a tiny slice of list header
                    stdscr.addstr(line, 0, f"Selected {len(state['selected_set'])}/{len(labels)}")
                    stdscr.refresh()
            # Margins adjust
            elif is_pdf and key in (ord('H'), ord('F'), ord('L'), ord('R')):
                mapk = {ord('H'): 'header', ord('F'): 'footer', ord('L'): 'left', ord('R'): 'right'}
                which = mapk[key]
                stdscr.addstr(h - 1, 0, f"Adjust {which} with +/- (Esc to cancel)")
                while True:
                    ch = stdscr.getch()
                    if ch in (27,):
                        break
                    if ch in (ord('-'), ord('_')):
                        state['margins'][which] = max(0.0, round(state['margins'][which] - 0.01, 2))
                    elif ch in (ord('+'), ord('=')):
                        state['margins'][which] = min(0.3, round(state['margins'][which] + 0.01, 2))
                    else:
                        break
                    # live refresh
                    line = render_header()
                    stdscr.addstr(line, 0, f"Selected {len(state['selected_set'])}/{len(labels)}")
                    stdscr.refresh()
            # Actions
            elif key in (ord('f'),):
                # inline file picker
                new_ebook = _browse_within_curses(stdscr, Path.cwd(), patterns=["*.epub", "*.pdf"], select_dir=False)
                if new_ebook:
                    state['ebook'] = new_ebook
                    load_from_file(new_ebook)
                    idx = 0
                    top = 0
                else:
                    message = 'No file selected'
            elif key in (ord('d'),):
                new_dir = _browse_within_curses(stdscr, Path.cwd(), patterns=["*"], select_dir=True)
                if new_dir:
                    state['out_dir'] = new_dir
                else:
                    message = 'No directory selected'
            elif key in (ord('p'),):
                return 'preview', state
            elif key in (ord('s'), 10, 13):
                if state['ebook'] is not None and len(state['selected_set']) >= 1:
                    return 'start', state
                else:
                    message = 'Pick a file (f) and select at least one chapter/page'
            elif key in (27, ord('q')):
                return 'quit', state
            elif 32 <= key <= 126:
                # first-letter jump in list
                ch = chr(key)
                for k, s in enumerate(labels):
                    if s.lower().startswith(ch.lower()):
                        idx = k
                        break

    action, final_state = __import__('curses').wrapper(run)
    return action, final_state


def main():
    print("Audiblez TUI — generate audiobooks from EPUB/PDF\n")
    # Start directly in the single-page screen; user can choose file/dir inside it.
    ebook: Optional[Path] = None
    out_dir = Path.cwd()
    is_pdf = False
    margins = dict(header=0.07, footer=0.07, left=0.07, right=0.07)
    chapters_all: List = []
    initially_selected: List = []

    while True:
        action, state = _single_page_screen(ebook, out_dir, is_pdf, margins, chapters_all, initially_selected)
        if action == 'quit' or action is None:
            print('Canceled.')
            return
        if action == 'preview':
            ebook = state['ebook']
            out_dir = state['out_dir']
            margins = state['margins']
            if ebook is None:
                print('No file selected.')
                continue
            # Build chapters from current ebook
            is_pdf = ebook.suffix.lower() == '.pdf'
            if is_pdf:
                from .pdf import extract_pages
                chapters_all = extract_pages(ebook, margins)
            else:
                from ebooklib import epub
                from .core import find_document_chapters_and_extract_texts
                book = epub.read_epub(str(ebook))
                chapters_all = find_document_chapters_and_extract_texts(book)
            if not chapters_all:
                print('No chapters/pages available.')
                continue
            selected = [chapters_all[i] for i in sorted(state['selected_set'])]
            _preview_loop(selected, state['voice'], state['speed'], state['backend'], state['mlx_model'])
            # loop back to screen
            continue
        if action == 'start':
            ebook = state['ebook']
            out_dir = state['out_dir']
            if ebook is None:
                print('No file selected.')
                return
            # Ensure MLX backend availability before jumping in
            chosen_backend = state['backend']
            try:
                from .tts_backends import has_mlx_audio
            except Exception:
                has_mlx_audio = lambda: False  # type: ignore
            # Only warn-block when user explicitly selects MLX but it's unavailable.
            # For 'auto', we will fall back to Kokoro below without blocking.
            if chosen_backend == 'mlx' and not has_mlx_audio():
                print("MLX-Audio is not installed. Install optional deps and retry:\n  uv sync -E mlx\nOr in pip env: pip install 'audiblez[mlx]'\nSwitch backend to Kokoro (press b) to continue without MLX.")
                continue
            # Recompute chapter list from current ebook and margins to ensure up-to-date
            is_pdf = ebook.suffix.lower() == '.pdf'
            if is_pdf and not chapters_all:
                from .pdf import extract_pages
                chapters_all = extract_pages(ebook, state['margins'])
            elif (not is_pdf) and not chapters_all:
                from ebooklib import epub
                from .core import find_document_chapters_and_extract_texts
                book = epub.read_epub(str(ebook))
                chapters_all = find_document_chapters_and_extract_texts(book)
            selected = [chapters_all[i] for i in sorted(state['selected_set'])]
            # Resolve effective backend for 'auto'
            eb = state['backend']
            try:
                from .tts_backends import has_mlx_audio
                if eb == 'auto':
                    eb = 'mlx' if has_mlx_audio() else 'kokoro'
            except Exception:
                if eb == 'auto':
                    eb = 'kokoro'
            backend = eb
            mlx_model = state['mlx_model']
            speed = state['speed']
            voice = state['voice']
            gold_min, gold_ideal, gold_max = state['gold_min'], state['gold_ideal'], state['gold_max']
            m = state['margins']
            # choose device automatically
            _choose_device()

            print("Starting synthesis...\n")
            core.main(
                str(ebook), voice, pick_manually=False, speed=speed,
                output_folder=str(out_dir), selected_chapters=selected,
                backend=backend, mlx_model=mlx_model,
                header=m.get('header', 0.07), footer=m.get('footer', 0.07),
                left=m.get('left', 0.07), right=m.get('right', 0.07),
                gold_min=gold_min, gold_ideal=gold_ideal, gold_max=gold_max,
                post_event=_post_event,
            )
            return


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


def _inline_pick(stdscr, options: List[str], title: str, jump_to_folders: bool = False, start_index: int = 0):
    """Single-select picker that reuses the current curses screen (no wrapper).

    Returns the selected index, or None if canceled.
    """
    import curses
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


def _browse_within_curses(stdscr, start_dir: Path, patterns: List[str], select_dir: bool = False) -> Optional[Path]:
    """Inline file/folder browser that stays within current curses context."""
    cur = start_dir.resolve()
    while True:
        if select_dir:
            title = f"Select output folder (current: {cur})"
            options = ["[Use this directory]"] + _list_dir_for_picker(cur, patterns=["*"])
        else:
            title = f"Select file (current: {cur})"
            options = _list_dir_for_picker(cur, patterns)
        idx = _inline_pick(stdscr, options, title, jump_to_folders=True, start_index=0)
        if idx is None:
            return None
        choice = options[idx]
        if select_dir and choice == "[Use this directory]":
            return cur
        if choice == "..":
            parent = cur.parent
            if parent != cur:
                cur = parent
            continue
        if choice.endswith('/'):
            cur = (cur / choice[:-1]).resolve()
            continue
        if not select_dir:  # file
            sel = (cur / choice).resolve()
            if sel.is_file():
                return sel

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
    return


if __name__ == '__main__':
    main()
