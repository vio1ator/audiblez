#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PDF text extraction using PyMuPDF (fitz).

Provides `extract_pages(file_path, margins)` which returns a list of
chapter-like page objects with `.extracted_text`, `.chapter_index`, and
`.get_name()` suitable for reuse by the existing audiobook pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import fitz  # PyMuPDF
import re


@dataclass
class PageChapter:
    page_num: int  # 1-based
    extracted_text: str

    # assigned by caller for UI/ordering compatibility
    chapter_index: int = 0

    def get_name(self) -> str:
        # Keep it simple and safe for filenames
        return f"Page {self.page_num}"


def _clean_text(text: str) -> str:
    """Normalize PDF-extracted text into readable sentences.

    - Remove hard line breaks inside paragraphs
    - Fix hyphenation at line endings (e.g., "Austro-\nPrussian" -> "Austro-Prussian")
    - Preserve paragraph breaks (double newlines)
    """
    if not text:
        return ""
    # Normalize newlines
    t = text.replace("\r", "\n")
    # Collapse trailing spaces per line
    t = "\n".join(ln.strip() for ln in t.split("\n"))
    # Normalize multiple blank lines to exactly two (paragraph break)
    t = re.sub(r"\n{3,}", "\n\n", t)
    # Remove hyphenation across line breaks: word-\nWord -> wordWord
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    # Join single line breaks that are likely mid-sentence into spaces.
    # If a newline is not followed by another newline (i.e., not a paragraph)
    # and the preceding character isn't sentence-ending punctuation, replace with space.
    # Consider ., !, ?, :, ;, em/en dash as potential soft boundaries to keep.
    t = re.sub(
        r"([^\n])(\n)([^\n])",
        lambda m: m.group(1) + ("\n" if m.group(1) in ".!?" else " ") + m.group(3),
        t,
    )
    # Collapse residual spaces
    t = re.sub(r"[ \t]+", " ", t)
    # Ensure paragraphs separated by exactly one blank line
    t = re.sub(r"\n\s*\n", "\n\n", t)
    return t.strip()


def extract_pages(file_path: str | Path, margins: Dict[str, float]) -> List[PageChapter]:
    """Extract text from each PDF page.

    - Trims header/footer/left/right using fractional margins of page size.
    - Sorts blocks top→bottom then left→right to preserve reading order.
    - Joins and cleans block texts.

    Args:
        file_path: Path to the PDF file.
        margins: Dict with keys 'header','footer','left','right' each in [0, 0.3].

    Returns:
        List of PageChapter objects (one per page).
    """
    p = Path(file_path)
    chapters: List[PageChapter] = []
    header = float(margins.get('header', 0.07) or 0)
    footer = float(margins.get('footer', 0.07) or 0)
    left = float(margins.get('left', 0.07) or 0)
    right = float(margins.get('right', 0.07) or 0)

    with fitz.open(p) as doc:
        for i, page in enumerate(doc, start=1):
            rect = page.rect
            w, h = rect.width, rect.height
            # Trim region by margins
            clip = fitz.Rect(
                rect.x0 + left * w,
                rect.y0 + header * h,
                rect.x1 - right * w,
                rect.y1 - footer * h,
            )
            # Extract text blocks within the clipped rectangle
            blocks = page.get_text("blocks", clip=clip) or []
            # blocks: list of tuples (x0,y0,x1,y1, text, block_no, block_type)
            # Sort by y (top→bottom), then x (left→right)
            blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
            pieces = []
            for b in blocks:
                text = (b[4] or "").strip()
                if text:
                    pieces.append(text)
            page_text = _clean_text("\n".join(pieces))
            ch = PageChapter(page_num=i, extracted_text=page_text, chapter_index=i - 1)
            chapters.append(ch)
    # Stitch sentence fragments across page boundaries
    _stitch_boundary_sentences(chapters)
    return chapters


def extract_page_preview(file_path: str | Path, margins: Dict[str, float], page_num: int) -> str:
    """Extract cleaned text for a single page without cross-page stitching.

    Args:
        file_path: Path to the PDF file.
        margins: Dict with keys 'header','footer','left','right' each in [0, 0.3].
        page_num: 1-based page number to extract.

    Returns:
        Cleaned text content of the page after applying margins.
    """
    p = Path(file_path)
    header = float(margins.get('header', 0.07) or 0)
    footer = float(margins.get('footer', 0.07) or 0)
    left = float(margins.get('left', 0.07) or 0)
    right = float(margins.get('right', 0.07) or 0)

    if page_num < 1:
        raise ValueError("page_num must be 1-based and >= 1")

    with fitz.open(p) as doc:
        if page_num > len(doc):
            return ""
        page = doc[page_num - 1]
        rect = page.rect
        w, h = rect.width, rect.height
        clip = fitz.Rect(
            rect.x0 + left * w,
            rect.y0 + header * h,
            rect.x1 - right * w,
            rect.y1 - footer * h,
        )
        blocks = page.get_text("blocks", clip=clip) or []
        blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
        pieces: List[str] = []
        for b in blocks:
            text = (b[4] or "").strip()
            if text:
                pieces.append(text)
        return _clean_text("\n".join(pieces))


def _stitch_boundary_sentences(chapters: List[PageChapter]) -> None:
    """Move trailing mid‑sentence fragments from page N to the start of N+1.

    Heuristics:
    - If a page doesn't end with sentence punctuation, take the trailing
      fragment after the last punctuation and prepend it to the next page.
    - Skip if the fragment looks like a standalone heading (very short or
      mostly uppercase), to avoid polluting body text.
    """
    def last_sentence_punct_idx(s: str) -> int | None:
        last = None
        for m in re.finditer(r'[\.\!\?…]["”\)\]]*', s or ''):
            last = m
        return last.end() if last else None

    def is_heading(fragment: str) -> bool:
        f = (fragment or '').strip()
        if not f:
            return True
        # Short fragments are often headings or section labels
        if len(f) <= 25:
            return True
        # Mostly uppercase (excluding punctuation/spaces) -> heading
        letters = [c for c in f if c.isalpha()]
        if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.7:
            return True
        return False

    for i in range(len(chapters) - 1):
        a = chapters[i].extracted_text or ''
        b = chapters[i + 1].extracted_text or ''
        idx = last_sentence_punct_idx(a)
        if idx is None:
            # Whole page is a fragment; if it's not heading, move it forward
            frag = a.strip()
            if frag and not is_heading(frag):
                chapters[i].extracted_text = ''
                joiner = '' if frag.endswith('-') else ' '
                chapters[i + 1].extracted_text = (frag + joiner + b.lstrip()).strip()
            continue
        trailing = a[idx:].strip()
        if trailing and not is_heading(trailing):
            kept = a[:idx].rstrip()
            joiner = '' if kept.endswith('-') else ' '
            chapters[i].extracted_text = kept
            chapters[i + 1].extracted_text = (trailing + joiner + b.lstrip()).strip()
