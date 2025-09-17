# -*- coding: utf-8 -*-
"""TTS backends for audiblez.

Provides a thin abstraction so core can switch between:
- MLX-Audio (Apple Silicon) using 4-bit Kokoro by default
- Original Kokoro KPipeline

The MLX backend shells out to the MLX-Audio CLI to avoid tightly coupling to
its Python API surface, which may evolve. If MLX-Audio is not installed or
invocation fails, callers should fall back to Kokoro.
"""
from __future__ import annotations

import os
import sys
import tempfile
import subprocess
from typing import Generator, Tuple
from shutil import which

import numpy as np


def has_mlx_audio() -> bool:
    # Consider either importable module or console script as presence
    if which('mlx-audio') or which('mlx_audio') or which('mlx_audio.tts.generate'):
        return True
    try:
        __import__("mlx_audio")
        return True
    except Exception:
        return False


class MlxAudioPipeline:
    """Minimal wrapper invoking MLX-Audio's TTS generator via its module CLI.

    Yields a single audio segment per call (the whole input chunk). The
    signature mirrors Kokoro's KPipeline so the rest of the code can stay the
    same.
    """

    def __init__(self, model: str = "mlx-community/Kokoro-82M-8bit", sample_rate: int = 24000) -> None:
        self.model = model
        self.sample_rate = sample_rate
        self.exe_override = os.environ.get('MLX_AUDIO_EXE')

    def __call__(self, text: str, voice: str, speed: float = 1.0, split_pattern: str = "\n\n\n") -> Generator[Tuple[None, None, np.ndarray], None, None]:
        # Use a temporary WAV file as an exchange format. Write text to file to
        # avoid oversized argv issues and to match common CLI signatures.
        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "out.wav")
            txt_path = os.path.join(td, "in.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            # Prefer console scripts if present (more stable than -m entrypoints)
            exe = self.exe_override or which('mlx-audio') or which('mlx_audio') or which('mlx_audio.tts.generate')
            cmd_variants = []
            if exe:
                cmd_variants.extend([
                    # Newer console script layout uses module name as script
                    # If the script name already targets generate, pass args directly
                    [exe, '--model', self.model, '--voice', voice, '--speed', str(speed), '--file_prefix', os.path.splitext(wav_path)[0], '--audio_format', 'wav', '--join_audio'],
                    # Or via subcommand style
                    [exe, 'tts', '--model', self.model, '--voice', voice, '--speed', str(speed), '--file_prefix', os.path.splitext(wav_path)[0], '--audio_format', 'wav', '--join_audio'],
                    [exe, 'generate', '--model', self.model, '--voice', voice, '--speed', str(speed), '--file_prefix', os.path.splitext(wav_path)[0], '--audio_format', 'wav', '--join_audio'],
                ])
            # Module invocations as fallback
            cmd_variants.extend([
                # Module invocation using explicit 'generate' entry point
                [sys.executable, "-m", "mlx_audio.tts.generate",
                 "--model", self.model,
                 "--voice", voice,
                 "--speed", str(speed),
                 "--file_prefix", os.path.splitext(wav_path)[0],
                 "--audio_format", 'wav', '--join_audio'],
            ])
            last_err = None
            for cmd in cmd_variants:
                try:
                    cp = subprocess.run(cmd, check=True, input=text.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    break
                except subprocess.CalledProcessError as e:
                    last_err = e
                    continue
            else:
                detail = ''
                if isinstance(last_err, subprocess.CalledProcessError):
                    try:
                        detail = (last_err.stderr or b'').decode('utf-8', 'ignore')
                    except Exception:
                        detail = ''
                raise RuntimeError(f"MLX-Audio invocation failed. Last error:\n{detail}" )

            # Load result
            import soundfile as sf
            audio, sr = sf.read(wav_path, dtype='float32')
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            yield None, None, audio.astype(np.float32)


class KokoroPipelineWrapper:
    """Small adapter to unify Kokoro KPipeline call signature."""

    def __init__(self, lang_code: str) -> None:
        from kokoro import KPipeline  # late import to avoid import cost when unused
        self.pipeline = KPipeline(lang_code=lang_code)
        # Force Misaki-only G2P so the runtime never falls back to espeak-ng.
        try:
            if getattr(self.pipeline, 'lang_code', '').lower() in ('a', 'b'):
                g2p = getattr(self.pipeline, 'g2p', None)
                if g2p is not None and hasattr(g2p, 'fallback'):
                    g2p.fallback = None
        except Exception:
            # Fallback removal should never block synthesis; log handled upstream.
            pass

    def __call__(self, text: str, voice: str, speed: float = 1.0, split_pattern: str = "\n\n\n"):
        # Proxy generator directly
        yield from self.pipeline(text, voice=voice, speed=speed, split_pattern=split_pattern)


def create_pipeline(backend: str, voice: str, mlx_model: str) -> object:
    """Create a pipeline compatible with gen_audio_segments.

    backend: 'mlx' or 'kokoro'
    voice: voice id like 'af_sky' (first char is language code)
    mlx_model: MLX-Audio model id/path
    """
    lang_code = voice[:1]
    if backend == 'mlx':
        if has_mlx_audio():
            return MlxAudioPipeline(model=mlx_model)
        raise RuntimeError("MLX-Audio is not installed. Install with: pip install 'audiblez[mlx]' or pip install mlx-audio")
    # fallback to kokoro
    return KokoroPipelineWrapper(lang_code=lang_code)
