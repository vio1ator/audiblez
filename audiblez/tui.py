#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audiblez TUI (urwid)

A redesigned, user-friendly terminal UI using urwid.

Highlights
- Clear, single-screen workflow: pick file, tune settings, pick chapters/pages
- Built-in file/directory browser overlays (EPUB/PDF)
- Quick voice/backend/speed controls; token and PDF margin tuning
- Chapter/page list with checkboxes, Select All / None / Min chars helpers
- Preview focused item (generates a short WAV and plays via ffplay/afplay/aplay)
- Run synthesis with a live progress overlay (non-blocking UI)

Notes
- Requires the optional dependency: urwid
  - uv:  uv sync --group tui   (in this repo)
  - pip: pip install -e .[tui]   (in this repo)
  - pypi: pip install "audiblez[tui]"

"""
from __future__ import annotations

import os
import sys
import fnmatch
import queue
import shutil
import threading
import traceback
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Iterable, List, Optional, Tuple

try:
    import urwid  # type: ignore
except Exception:  # pragma: no cover - graceful message for missing extra
    print(
        "urwid is required for the TUI.\n"
        "Install the optional dependency group:\n"
        "- uv:  uv sync --group tui   (in this repo)\n"
        "- pip: pip install -e .[tui]   (in this repo)\n"
        "- PyPI users: pip install 'audiblez[tui]'\n"
    )
    raise

import platform

from .voices import voices as VOICES_BY_LANG, flags as FLAGS
from . import core


# ------------------------------ Utilities ------------------------------

def _flatten_voices() -> List[str]:
    out: List[str] = []
    for _, v in VOICES_BY_LANG.items():
        out.extend(v)
    return out


def _choose_device_auto() -> str:
    """Auto-select compute device without prompting the user. Returns label."""
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


def _play_audio_blocking(path: str) -> None:
    """Play a WAV file using available CLI players, blocking until done."""
    if shutil.which("ffplay"):
        subprocess.run(["ffplay", "-autoexit", "-nodisp", path])
        return
    if platform.system() == "Darwin" and shutil.which("afplay"):
        subprocess.run(["afplay", path])
        return
    if platform.system() == "Linux" and shutil.which("aplay"):
        subprocess.run(["aplay", path])
        return
    if shutil.which("cvlc"):
        subprocess.run(["cvlc", "--play-and-exit", path])
        return
    raise RuntimeError("No suitable audio player found (ffplay/afplay/aplay/cvlc)")


def _file_patterns() -> List[str]:
    return ["*.epub", "*.pdf"]


@dataclass
class Margins:
    header: float = 0.07
    footer: float = 0.07
    left: float = 0.07
    right: float = 0.07


# ------------------------------ Overlays ------------------------------

class Modal:
    """Minimal modal overlay helper."""

    def __init__(self, loop: urwid.MainLoop, frame_provider: Callable[[], urwid.Widget]):
        self.loop = loop
        self._frame_provider = frame_provider
        self._stack: List[urwid.Widget] = []

    def open(self, w: urwid.Widget, width=90, height=80, title: Optional[str] = None) -> None:
        if title:
            w = urwid.LineBox(w, title)
        # Use a BOX filler so inner Piles receive (maxcol, maxrow) and weighted
        # items can expand vertically (fixes one-line list issue in dialogs).
        overlay = urwid.Overlay(
            urwid.Filler(w, valign="top", height=("relative", 100)),
            self._frame_provider(),
            align="center",
            width=("relative", width),
            valign="middle",
            height=("relative", height),
        )
        self._stack.append(self.loop.widget)
        self.loop.widget = overlay

    def close(self) -> None:
        if self._stack:
            self.loop.widget = self._stack.pop()


class VoicePicker(urwid.WidgetWrap):
    """Filterable voice picker. Calls on_done(voice) or on_cancel()."""

    def __init__(self, voices: Iterable[str], on_done: Callable[[str], None], on_cancel: Callable[[], None]):
        self.on_done = on_done
        self.on_cancel = on_cancel
        self.all_voices = list(voices)
        self.edit = urwid.Edit("Filter: ")
        self.listbox = urwid.ListBox(urwid.SimpleFocusListWalker([]))
        self.hint = urwid.Text("Enter to select • Esc to cancel")
        hint_pad = urwid.Padding(self.hint, left=0, right=0)
        self._refresh()
        pile = urwid.Pile([
            (urwid.PACK, urwid.Padding(self.edit, left=0, right=0)),
            (urwid.PACK, urwid.Divider()),
            ("weight", 1, self.listbox),
            (urwid.PACK, urwid.Divider()),
            (urwid.PACK, hint_pad),
        ])
        super().__init__(pile)

    def _refresh(self) -> None:
        q = (self.edit.edit_text or "").strip().lower()
        items = []
        for v in self.all_voices:
            if not q or q in v.lower():
                items.append(urwid.AttrMap(urwid.SelectableIcon(v, 0), None, focus_map="reversed"))
        self.listbox.body[:] = items
        if items:
            self.listbox.focus_position = 0

    def keypress(self, size, key):  # noqa: D401
        if key == "esc":
            self.on_cancel()
            return None
        if key == "enter":
            try:
                w = self.listbox.get_focus()[0]
                voice = w.original_widget.get_text()[0]
                self.on_done(voice)
                return None
            except Exception:
                pass
        ret = super().keypress(size, key)
        if key not in ("up", "down", "page up", "page down", "enter", "esc"):
            self._refresh()
        return ret


class FileBrowser(urwid.WidgetWrap):
    """Simple in-app file browser.

    - If select_dir is True, returns a directory via on_done(Path)
    - Otherwise returns a file matching patterns
    """

    def __init__(self, start_dir: Path, patterns: List[str], select_dir: bool,
                 on_done: Callable[[Path], None], on_cancel: Callable[[], None]):
        self.cur = start_dir.resolve()
        self.patterns = patterns
        self.select_dir = select_dir
        self.on_done = on_done
        self.on_cancel = on_cancel

        self.title = urwid.Text("")
        title_pad = urwid.Padding(self.title, left=0, right=0)
        self.listbox = urwid.ListBox(urwid.SimpleFocusListWalker([]))
        self._refresh()
        pile = urwid.Pile([
            (urwid.PACK, title_pad),
            (urwid.PACK, urwid.Divider()),
            ("weight", 1, self.listbox),
            (urwid.PACK, urwid.Divider()),
            (urwid.PACK, urwid.Padding(urwid.Text("Enter select • Backspace up • Esc cancel • letters jump"), left=0, right=0)),
        ])
        super().__init__(pile)

    def _list_dir(self) -> List[str]:
        items: List[str] = []
        if self.select_dir:
            items.append("[Use this directory]")
        items.append("..")
        for d in sorted([p for p in self.cur.iterdir() if p.is_dir()]):
            items.append(d.name + "/")
        for f in sorted([p for p in self.cur.iterdir() if p.is_file()]):
            if any(fnmatch.fnmatch(f.name.lower(), pat) for pat in self.patterns):
                items.append(f.name)
        return items

    def _refresh(self):
        self.title.set_text(f"Current: {self.cur}")
        items = []
        for s in self._list_dir():
            items.append(urwid.AttrMap(urwid.SelectableIcon(s, 0), None, focus_map="reversed"))
        self.listbox.body[:] = items
        if items:
            self.listbox.focus_position = 0

    def keypress(self, size, key):
        if key == "esc":
            self.on_cancel(); return None
        if key == "backspace":
            parent = self.cur.parent
            if parent != self.cur:
                self.cur = parent
                self._refresh()
            return None
        if key == "enter":
            try:
                label = self.listbox.get_focus()[0].original_widget.get_text()[0]
            except Exception:
                return None
            if self.select_dir and label == "[Use this directory]":
                self.on_done(self.cur)
                return None
            if label == "..":
                parent = self.cur.parent
                if parent != self.cur:
                    self.cur = parent
                    self._refresh()
                return None
            if label.endswith("/"):
                self.cur = (self.cur / label[:-1]).resolve()
                self._refresh(); return None
            if not self.select_dir:
                p = (self.cur / label).resolve()
                if p.exists() and p.is_file():
                    self.on_done(p)
                    return None
            return None
        # First-letter jump
        if len(key) == 1 and key.isprintable():
            target = key.lower()
            for i, w in enumerate(self.listbox.body):
                s = w.original_widget.get_text()[0]
                if s.lower().startswith(target):
                    self.listbox.focus_position = i
                    break
            return None
        return super().keypress(size, key)


# ------------------------------ Main UI ------------------------------

class AudiblezTUI:
    def __init__(self) -> None:
        self.palette = [
            ("reversed", "standout", ""),
            ("title", "light cyan,bold", ""),
            ("hint", "dark gray", ""),
            ("label", "light gray", ""),
            ("ok", "light green", ""),
            ("bad", "light red", ""),
        ]

        # State
        self.file_path: Optional[Path] = None
        self.out_dir: Path = Path.cwd()
        self.is_pdf: bool = False
        self.margins = Margins()
        self.voices = _flatten_voices()
        self.voice: str = self.voices[0] if self.voices else "af_sky"
        self.backend: str = "auto"  # auto|mlx|kokoro
        self.mlx_model: str = "mlx-community/Kokoro-82M-8bit"
        self.speed: float = 1.0
        self.gold_min: int = 10
        self.gold_ideal: int = 25
        self.gold_max: int = 40
        self.device: str = _choose_device_auto()

        # Document
        self.items: List[object] = []  # chapters or PageChapter
        self.labels: List[str] = []
        self.lengths: List[int] = []
        self.selected: set[int] = set()

        # Build widgets
        self.header = urwid.Text(("title", "Audiblez — Generate Audiobooks from EPUB/PDF"))
        self.header_hint = urwid.Text(("hint", "Tab/Shift+Tab to move • Enter to press • q to quit"))
        header_pad = urwid.Padding(self.header, left=0, right=0)
        header_hint_pad = urwid.Padding(self.header_hint, left=0, right=0)

        self.file_label = urwid.Text(("label", "File: "))
        self.file_value = urwid.Text("[select an EPUB/PDF]")
        self.file_btn = urwid.Button("Browse", on_press=self._browse_file)

        self.out_label = urwid.Text(("label", "Output: "))
        self.out_value = urwid.Text(str(self.out_dir))
        self.out_btn = urwid.Button("Browse", on_press=self._browse_outdir)

        self.voice_btn = urwid.Button(f"Voice: {self.voice}", on_press=self._choose_voice)
        # Backend radios
        self.rb_auto = urwid.RadioButton([], "Backend: auto", state=True, on_state_change=self._backend_changed)
        self.rb_mlx = urwid.RadioButton([self.rb_auto], "MLX", state=False, on_state_change=self._backend_changed)
        self.rb_kok = urwid.RadioButton([self.rb_auto], "Kokoro", state=False, on_state_change=self._backend_changed)
        self.model_edit = urwid.Edit("MLX model: ", self.mlx_model)
        self.speed_edit = urwid.Edit("Speed (0.5–2.0): ", "1.0")
        self.device_text = urwid.Text(("label", f"Device: {self.device}"))

        # Token group
        self.min_edit = urwid.IntEdit("Min tokens: ", default=self.gold_min)
        self.ideal_edit = urwid.IntEdit("Ideal: ", default=self.gold_ideal)
        self.max_edit = urwid.IntEdit("Max: ", default=self.gold_max)

        # PDF margins
        self.header_edit = urwid.Edit("Header [0–0.3]: ", str(self.margins.header))
        self.footer_edit = urwid.Edit("Footer [0–0.3]: ", str(self.margins.footer))
        self.left_edit = urwid.Edit("Left [0–0.3]: ", str(self.margins.left))
        self.right_edit = urwid.Edit("Right [0–0.3]: ", str(self.margins.right))
        self._update_margin_visibility()

        # Chapter list (custom to catch Tab)
        class ChaptersList(urwid.ListBox):
            def __init__(self, body, on_tab: Callable[[], None]):
                super().__init__(body)
                self._on_tab = on_tab
            def keypress(self, size, key):
                if key in ("tab", "ctrl i"):
                    try:
                        self._on_tab()
                        return None
                    except Exception:
                        pass
                return super().keypress(size, key)

        self.listbox = ChaptersList(urwid.SimpleFocusListWalker([]), on_tab=lambda: self._focus_actions_row())

        # Action buttons
        self.btn_select_all = urwid.Button("Select All", on_press=self._select_all)
        self.btn_select_none = urwid.Button("Select None", on_press=self._select_none)
        self.btn_min_chars = urwid.Button("Min chars…", on_press=self._select_min_chars)
        self.btn_view_text = urwid.Button("View text", on_press=self._view_current_text)
        self.btn_preview = urwid.Button("Preview", on_press=self._preview_current)
        self.btn_start = urwid.Button("Start", on_press=self._start)

        self.footer = urwid.Text(("hint", "Space toggles a checkbox • Use the buttons below to manage selections"))
        footer_pad = urwid.Padding(self.footer, left=0, right=0)

        # Layout
        file_row = urwid.Columns([
            (10, self.file_label), ("weight", 2, self.file_value), (12, self.file_btn)
        ], dividechars=1)
        out_row = urwid.Columns([
            (10, self.out_label), ("weight", 2, self.out_value), (12, self.out_btn)
        ], dividechars=1)
        source_box = urwid.LineBox(
            urwid.Pile([(urwid.PACK, file_row), (urwid.PACK, out_row)]),
            title="Source & Output",
        )

        backend_row = urwid.Columns([
            ("weight", 1, self.rb_auto), ("weight", 1, self.rb_mlx), ("weight", 1, self.rb_kok),
        ], dividechars=2)
        settings_items = [
            self.voice_btn,
            backend_row,
            self.model_edit,
            self.speed_edit,
            self.device_text,
            urwid.Divider(),
            urwid.Columns([("weight", 1, self.min_edit), ("weight", 1, self.ideal_edit), ("weight", 1, self.max_edit)], dividechars=2),
            urwid.Divider(),
            urwid.Columns([("weight", 1, self.header_edit), ("weight", 1, self.footer_edit), ("weight", 1, self.left_edit), ("weight", 1, self.right_edit)], dividechars=2),
        ]
        settings_box = urwid.LineBox(
            urwid.Pile([(urwid.PACK, w) for w in settings_items]),
            title="Settings",
        )

        # Actions row (catch Shift+Tab to go back)
        class ActionsRow(urwid.Columns):
            def __init__(self, contents, on_back: Callable[[], None], **kwargs):
                super().__init__(contents, **kwargs)
                self._on_back = on_back
            def keypress(self, size, key):
                if key in ("shift tab", "backtab"):
                    try:
                        self._on_back()
                        return None
                    except Exception:
                        pass
                return super().keypress(size, key)

        self.actions_row = ActionsRow([
            (12, urwid.AttrMap(self.btn_select_all, None, focus_map="reversed")),
            (14, urwid.AttrMap(self.btn_select_none, None, focus_map="reversed")),
            (12, urwid.AttrMap(self.btn_min_chars, None, focus_map="reversed")),
            (12, urwid.AttrMap(self.btn_view_text, None, focus_map="reversed")),
            (10, urwid.AttrMap(self.btn_preview, None, focus_map="reversed")),
            (10, urwid.AttrMap(self.btn_start, None, focus_map="reversed")),
        ], on_back=lambda: self._focus_chapter_list(), dividechars=1)

        # Keep handles to chapters layout parts for focus management
        self.chapters_pile = urwid.Pile([
            ("weight", 1, self.listbox),
            (urwid.PACK, urwid.Divider()),
            (urwid.PACK, self.actions_row),
        ])
        # Indices within chapters pile
        self._chapters_list_idx = 0
        self._chapters_actions_idx = 2

        self.chapters_box = urwid.LineBox(
            self.chapters_pile,
            title="Chapters / Pages",
        )

        self.body = urwid.Pile([
            (urwid.PACK, header_pad),
            (urwid.PACK, header_hint_pad),
            (urwid.PACK, urwid.Divider()),
            (urwid.PACK, source_box),
            (urwid.PACK, urwid.Divider()),
            (urwid.PACK, settings_box),
            (urwid.PACK, urwid.Divider()),
            ("weight", 1, self.chapters_box),
            (urwid.PACK, urwid.Divider()),
            (urwid.PACK, footer_pad),
        ])
        self.frame = urwid.Frame(body=self.body)
        self.loop = urwid.MainLoop(
            self.frame,
            self.palette,
            unhandled_input=self._unhandled,
            input_filter=self._input_filter,
        )
        self.modal = Modal(self.loop, lambda: self.frame)

        # Synthesis thread + queue
        self._worker: Optional[threading.Thread] = None
        self._queue: "queue.Queue[Tuple[str, dict]]" = queue.Queue()
        self._progress_overlay: Optional[urwid.Widget] = None
        self._progress_text = urwid.Text("")
        self._progress_bar = urwid.ProgressBar("ok", "bad", current=0, done=100)

    # ------------------------------ Wiring ------------------------------

    def _unhandled(self, key):  # noqa: D401
        if key in ("q", "Q") and not self._worker:
            raise urwid.ExitMainLoop()

    def _in_chapters_box(self) -> bool:
        try:
            if self.frame.focus_part != "body":
                return False
            w, _ = self.body.get_focus()
            return w is self.chapters_box
        except Exception:
            return False

    def _input_filter(self, keys, raw):
        # Intercept Tab before widgets handle it so we can jump
        # directly between the chapter list and the action buttons.
        if not keys:
            return keys
        try:
            if os.environ.get("AUDIBLEZ_TUI_DEBUG_KEYS"):
                # Helpful for diagnosing terminal key names
                try:
                    print("keys:", keys, "raw:", [repr(x) for x in (raw or [])])
                except Exception:
                    pass
            for key in keys:
                if key in ("tab", "ctrl i") and self._in_chapters_box() and self.chapters_pile.focus_position == self._chapters_list_idx:
                    # Jump to action buttons, focus first button
                    self.chapters_pile.focus_position = self._chapters_actions_idx
                    try:
                        self.actions_row.focus_position = 0
                    except Exception:
                        pass
                    return []  # swallow key
                if key in ("shift tab", "backtab") and self._in_chapters_box() and self.chapters_pile.focus_position == self._chapters_actions_idx:
                    # Jump back to the list
                    self.chapters_pile.focus_position = self._chapters_list_idx
                    return []
        except Exception:
            pass
        return keys

    # ----- Focus helpers for chapters/actions
    def _focus_actions_row(self) -> None:
        try:
            # Ensure chapters box is focused in body
            try:
                self.body.set_focus(self.chapters_box)
            except Exception:
                pass
            self.chapters_pile.focus_position = self._chapters_actions_idx
            try:
                self.actions_row.focus_position = 0
            except Exception:
                pass
        except Exception:
            pass

    def _focus_chapter_list(self) -> None:
        try:
            try:
                self.body.set_focus(self.chapters_box)
            except Exception:
                pass
            self.chapters_pile.focus_position = self._chapters_list_idx
        except Exception:
            pass

    def _backend_changed(self, btn: urwid.RadioButton, state: bool) -> None:
        if not state:
            return
        if btn is self.rb_auto:
            self.backend = "auto"
        elif btn is self.rb_mlx:
            self.backend = "mlx"
        else:
            self.backend = "kokoro"
        self._update_margin_visibility()  # just redraw

    def _update_margin_visibility(self) -> None:
        show = bool(self.file_path and self.is_pdf)
        for w in (self.header_edit, self.footer_edit, self.left_edit, self.right_edit):
            w.set_caption(w.caption if show else f"{w.caption.split('[')[0].strip()} (PDF only): ")

    # ------------------------------ Loaders ------------------------------

    def _browse_file(self, *_):
        def done(p: Path):
            self.modal.close()
            self._load_file(p)

        def cancel():
            self.modal.close()

        fb = FileBrowser(Path.cwd(), _file_patterns(), select_dir=False, on_done=done, on_cancel=cancel)
        self.modal.open(fb, width=95, height=90, title="Select EPUB or PDF")

    def _browse_outdir(self, *_):
        def done(p: Path):
            self.modal.close()
            self.out_dir = p
            self.out_value.set_text(str(self.out_dir))

        def cancel():
            self.modal.close()

        fb = FileBrowser(self.out_dir, ["*"], select_dir=True, on_done=done, on_cancel=cancel)
        self.modal.open(fb, width=95, height=90, title="Select Output Folder")

    def _choose_voice(self, *_):
        def done(v: str):
            self.modal.close()
            self.voice = v
            self.voice_btn.set_label(f"Voice: {self.voice}")

        def cancel():
            self.modal.close()

        picker = VoicePicker(self.voices, on_done=done, on_cancel=cancel)
        self.modal.open(picker, height=80, title="Select Voice")

    def _load_file(self, path: Path) -> None:
        self.file_path = path
        self.file_value.set_text(str(path))
        self.is_pdf = path.suffix.lower() == ".pdf"
        self._update_margin_visibility()
        try:
            if self.is_pdf:
                from .pdf import extract_pages
                self.items = extract_pages(path, {
                    "header": float(self.header_edit.edit_text or self.margins.header),
                    "footer": float(self.footer_edit.edit_text or self.margins.footer),
                    "left": float(self.left_edit.edit_text or self.margins.left),
                    "right": float(self.right_edit.edit_text or self.margins.right),
                })
                self.selected = set(range(len(self.items)))
            else:
                from ebooklib import epub  # type: ignore
                from .core import find_document_chapters_and_extract_texts, find_good_chapters, chapter_beginning_one_liner
                book = epub.read_epub(str(path))
                all_ch = find_document_chapters_and_extract_texts(book)
                good = find_good_chapters(all_ch)
                self.items = all_ch
                self.selected = {i for i, c in enumerate(all_ch) if c in good}
            self._rebuild_labels()
            self._rebuild_listbox()
        except Exception:
            traceback.print_exc()
            self.items = []
            self.selected = set()
            self._rebuild_labels()
            self._rebuild_listbox()

    def _rebuild_labels(self) -> None:
        self.labels = []
        self.lengths = []
        if not self.items:
            return
        if self.is_pdf:
            for p in self.items:
                # type: ignore[attr-defined]
                text = (p.extracted_text or "").strip()
                prev = text.splitlines()[0:1]
                prev = prev[0][:50] + ("..." if len(prev[0]) > 50 else "") if prev else ""
                name = p.get_name()
                self.labels.append(f"{name} ({len(text)} chars) [{prev}]")
                self.lengths.append(len(text))
        else:
            from .core import chapter_beginning_one_liner
            for c in self.items:
                name = c.get_name()
                prev = chapter_beginning_one_liner(c, 50)
                self.labels.append(f"{name} ({len(c.extracted_text)} chars) [{prev}]")
                self.lengths.append(len(c.extracted_text))

    def _rebuild_listbox(self) -> None:
        body = []
        for i, label in enumerate(self.labels):
            cb = urwid.CheckBox(label, state=(i in self.selected))
            def toggled(chk, state, idx=i):
                if state:
                    self.selected.add(idx)
                else:
                    self.selected.discard(idx)
            urwid.connect_signal(cb, 'change', toggled)
            body.append(urwid.AttrMap(cb, None, focus_map="reversed"))
        self.listbox.body[:] = body

    # ------------------------------ Selection helpers ------------------------------

    def _select_all(self, *_):
        self.selected = set(range(len(self.labels)))
        self._rebuild_listbox()

    def _select_none(self, *_):
        self.selected = set()
        self._rebuild_listbox()

    def _select_min_chars(self, *_):
        def ok(btn):
            try:
                thr = int(edit.edit_text.strip() or "0")
            except Exception:
                thr = 0
            self.selected = {i for i, L in enumerate(self.lengths) if L >= thr}
            self.modal.close()
            self._rebuild_listbox()

        def cancel(btn):
            self.modal.close()

        edit = urwid.Edit("Minimum characters: ", "500")
        actions = urwid.Columns([(10, urwid.Button("OK", on_press=ok)), (10, urwid.Button("Cancel", on_press=cancel))])
        self.modal.open(
            urwid.Pile([
                (urwid.PACK, edit),
                (urwid.PACK, urwid.Divider()),
                (urwid.PACK, actions),
            ]),
            height=30,
            title="Select by minimum length",
        )

    def _get_focused_index(self) -> Optional[int]:
        try:
            pos = self.listbox.focus_position
        except Exception:
            return None
        return int(pos)

    def _current_item_text(self) -> str:
        idx = self._get_focused_index()
        if idx is None or idx >= len(self.items):
            return ""
        if self.is_pdf:
            return self.items[idx].extracted_text or ""
        return self.items[idx].extracted_text

    def _view_current_text(self, *_):
        text = self._current_item_text() or ""
        text_w = urwid.Text(text)
        close_btn = urwid.Button("Close", on_press=lambda *_: self.modal.close())
        pile = urwid.Pile([
            ("weight", 1, urwid.Filler(urwid.Padding(text_w, left=1, right=1), valign="top")),
            (urwid.PACK, urwid.Divider()),
            (urwid.PACK, close_btn),
        ])
        self.modal.open(pile, height=90, title="Full text")

    # ------------------------------ Preview & Run ------------------------------

    def _preview_current(self, *_):
        if not self.items:
            return
        try:
            idx = self._get_focused_index()
            if idx is None:
                return
            text = self._current_item_text()
            if not text or len(text.strip()) < 5:
                self._notify("Nothing to preview for this item")
                return
            _choose_device_auto()
            tmp = NamedTemporaryFile(suffix=".wav", delete=False)
            # Resolve backend for 'auto'
            backend = self.backend
            try:
                from .tts_backends import has_mlx_audio  # type: ignore
                if backend == "auto":
                    backend = "mlx" if has_mlx_audio() else "kokoro"
            except Exception:
                backend = "kokoro" if backend == "auto" else backend
            core.gen_text(
                text, voice=self.voice, output_file=tmp.name,
                speed=float(self.speed_edit.edit_text or "1.0"),
                play=False, backend=backend, mlx_model=self.model_edit.edit_text or self.mlx_model,
            )
            _play_audio_blocking(tmp.name)
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
        except Exception:
            traceback.print_exc()
            self._notify("Preview failed (see logs)")

    def _notify(self, message: str) -> None:
        text = urwid.Text(message)
        btn = urwid.Button("OK", on_press=lambda *_: self.modal.close())
        self.modal.open(
            urwid.Pile([
                (urwid.PACK, urwid.Padding(text, left=0, right=0)),
                (urwid.PACK, urwid.Divider()),
                (urwid.PACK, btn),
            ]),
            height=30,
            title="Info",
        )

    def _start(self, *_):
        if not self.file_path:
            self._notify("Select an EPUB/PDF first")
            return
        if not self.selected:
            self._notify("Select at least one chapter/page")
            return
        # Prepare selection
        selected_items = [self.items[i] for i in sorted(self.selected)]
        backend = self.backend
        try:
            from .tts_backends import has_mlx_audio  # type: ignore
            if backend == "auto":
                backend = "mlx" if has_mlx_audio() else "kokoro"
        except Exception:
            backend = "kokoro" if backend == "auto" else backend

        # Launch worker thread
        args = dict(
            file_path=str(self.file_path),
            voice=self.voice,
            speed=float(self.speed_edit.edit_text or "1.0"),
            backend=backend,
            mlx_model=self.model_edit.edit_text or self.mlx_model,
            output_folder=str(self.out_dir),
            gold_min=int((self.min_edit.edit_text or str(self.gold_min)).strip() or self.gold_min),
            gold_ideal=int((self.ideal_edit.edit_text or str(self.gold_ideal)).strip() or self.gold_ideal),
            gold_max=int((self.max_edit.edit_text or str(self.gold_max)).strip() or self.gold_max),
        )
        margins = {
            "header": float(self.header_edit.edit_text or self.margins.header),
            "footer": float(self.footer_edit.edit_text or self.margins.footer),
            "left": float(self.left_edit.edit_text or self.margins.left),
            "right": float(self.right_edit.edit_text or self.margins.right),
        }

        def post_event(name: str, **kwargs):
            self._queue.put((name, kwargs))

        def worker():
            try:
                _choose_device_auto()
                if self.is_pdf:
                    core.main(
                        args["file_path"], args["voice"], pick_manually=False, speed=args["speed"],
                        output_folder=args["output_folder"], selected_chapters=selected_items,
                        backend=args["backend"], mlx_model=args["mlx_model"],
                        header=margins["header"], footer=margins["footer"], left=margins["left"], right=margins["right"],
                        gold_min=args["gold_min"], gold_ideal=args["gold_ideal"], gold_max=args["gold_max"],
                        post_event=post_event,
                    )
                else:
                    core.main(
                        args["file_path"], args["voice"], pick_manually=False, speed=args["speed"],
                        output_folder=args["output_folder"], selected_chapters=selected_items,
                        backend=args["backend"], mlx_model=args["mlx_model"],
                        gold_min=args["gold_min"], gold_ideal=args["gold_ideal"], gold_max=args["gold_max"],
                        post_event=post_event,
                    )
            except Exception:
                traceback.print_exc()
                self._queue.put(("ERROR", {"message": "Synthesis failed; see logs"}))
            finally:
                self._queue.put(("DONE", {}))

        # Progress overlay UI
        self._progress_text.set_text("Preparing…")
        self._progress_bar.set_completion(0)
        overlay = urwid.Pile([
            (urwid.PACK, urwid.Padding(urwid.Text("Synthesis in progress — this may take a while…"), left=0, right=0)),
            (urwid.PACK, urwid.Divider()),
            (urwid.PACK, self._progress_bar),
            (urwid.PACK, urwid.Divider()),
            ("weight", 1, urwid.Filler(urwid.Padding(self._progress_text, left=1, right=1), valign="top")),
        ])
        self._progress_overlay = overlay
        self.modal.open(overlay, height=70, title="Working…")

        # Start worker + poller
        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
        self.loop.set_alarm_in(0.1, self._poll_queue)

    def _poll_queue(self, *_):
        updated = False
        while True:
            try:
                name, data = self._queue.get_nowait()
            except queue.Empty:
                break
            updated = True
            if name == "CORE_STARTED":
                self._progress_text.set_text("Started…")
            elif name == "CORE_PROGRESS":
                st = data.get("stats")
                if st is not None:
                    # stats.progress, stats.eta, stats.chars_per_sec
                    try:
                        self._progress_bar.set_completion(int(getattr(st, "progress", 0)))
                    except Exception:
                        pass
                    self._progress_text.set_text(
                        f"Progress: {getattr(st, 'progress', 0)}%\n"
                        f"ETA: {getattr(st, 'eta', '?')}\n"
                        f"Rate: {getattr(st, 'chars_per_sec', 0):.0f} chars/s"
                    )
            elif name == "CORE_CHAPTER_STARTED":
                self._progress_text.set_text("Starting next chapter…")
            elif name == "CORE_CHAPTER_FINISHED":
                self._progress_text.set_text("Chapter finished.")
            elif name == "CORE_FINISHED":
                self._progress_text.set_text("Done. Creating m4b…")
            elif name == "ERROR":
                self._progress_text.set_text(data.get("message", "Error"))
            elif name == "DONE":
                self.modal.close()
                self._worker = None
        if self._worker:
            self.loop.set_alarm_in(0.2, self._poll_queue)

    # ------------------------------ Run ------------------------------

    def run(self) -> None:
        self.loop.run()


def main() -> None:
    """Entrypoint for `audiblez-tui`."""
    app = AudiblezTUI()
    app.run()


if __name__ == "__main__":
    main()
