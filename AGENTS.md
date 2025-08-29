# Repository Guidelines

## Project Structure & Module Organization
- `audiblez/`: source package.
  - `cli.py`: command-line entry (`audiblez`).
  - `core.py`: EPUB parsing, chapter selection, TTS, WAV/M4B assembly.
  - `ui.py`: desktop GUI entry (`audiblez-ui`).
  - `voices.py`: voice registry and helpers.
- `test/`: unit tests using `unittest` (e.g., `test_*.py`).
- `imgs/`: README images; `samples/`: short audio samples.

## Build, Test, and Development Commands
- Install (editable): `python -m pip install -e .`
- CLI help: `audiblez --help`
- Run GUI: `audiblez-ui`
- Run tests: `python -m unittest discover test -v`
- Optional dep check: `python -m pip install deptry && deptry .`

System prerequisites for audio packaging: install `ffmpeg` and `espeak-ng` (e.g., `brew install ffmpeg espeak-ng` or `sudo apt-get install ffmpeg espeak-ng`).

## Coding Style & Naming Conventions
- Python ≥ 3.9, PEP 8, 4‑space indentation.
- Use descriptive names (`snake_case` for functions/vars, `PascalCase` for classes, `UPPER_CASE` for constants).
- Add docstrings to public functions; prefer type hints for new/changed code.
- Keep modules focused: CLI/UI thin; logic lives in `core.py`.

## Testing Guidelines
- Framework: `unittest`.
- Place tests in `test/` and name files `test_*.py`; test methods `test_*` inside `unittest.TestCase`.
- Some tests download EPUBs and require `ffmpeg`; when adding tests, prefer small fixtures and avoid network unless necessary.
- Run locally with `python -m unittest discover test -v`.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject; optionally prefix with scope/type (e.g., `feat(core):`, `fix(cli):`, `perf(gui):`).
- PRs: include summary, rationale, before/after notes (logs or file outputs), and steps to reproduce. Link related issues and mention platform (CPU/CUDA/MPS) if relevant.
- Keep PRs focused and incremental; include tests or sample commands when behavior changes.

## Security & Configuration Tips
- Do not commit generated audio (`*.wav`, `*.m4b`) or large assets.
- GPU: CLI supports `--cuda`; Apple Silicon supports `--mps`. Validate fallbacks to CPU.
- Handle user‑supplied EPUB paths safely; prefer pathlib operations and validate output folders.

