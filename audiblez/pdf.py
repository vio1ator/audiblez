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
    # Basic cleanup: normalize whitespace, strip, and ensure newline termination
    if not text:
        return ""
    t = text.replace('\r', '\n')
    # Collapse multiple newlines and spaces
    lines = [ln.strip() for ln in t.split('\n')]
    t = '\n'.join([ln for ln in lines if ln])
    # Avoid empty trailing whitespace
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
    return chapters

