# Project Guidelines

## Running Scripts

**ALWAYS use `uv run` to execute Python scripts.** This project uses `uv` for
dependency management. Never use bare `python` or `python3` commands.

```bash
uv run python scripts/run_dataset_pipeline.py --dataset data/synthetic
```

## Critical Rules (NEVER violate these)

1. **BEFORE creating ANY function, search for existing implementations** in
   `src/common/` (stats, schemas, file I/O, tokenization) and `src/inference/`
   (embeddings, residual streams). Only create new code if nothing reusable exists.
2. **ALL `__init__.py` files use auto-export** via
   `from src.common.auto_export import auto_export` +
   `__all__ = auto_export(__file__, __name__, globals())`.
3. **ALL imports go at the top of every file.** No inline imports.
4. **NO legacy code, NO backwards-compatibility shims, NO dead code.**
5. **No two `.py` files anywhere in the repo may share a filename.**
6. **No single-word `.py` filenames** (except `__init__.py`). Use descriptive
   compound names: `marked_words_analyzer.py`, not `lexical.py`.
7. **No nested raw `dict`/`list` in function signatures or returns.** Define a
   `BaseSchema` subclass (`src/common/base_schema.py`) for structured data.
8. **Every dataclass that crosses a module boundary or gets serialized MUST
   inherit from `BaseSchema`.**
9. Keep files small and focused (aim under ~150 lines); split rather than grow.

## Architecture

- `src/common/` — dataset schema `D=(x,y,z,d,c)` (`dataset_annotations.py`,
  `prompt_set_schema.py`), config, stats, file I/O (no heavy deps)
- `src/inference/` — model-backed runners (sentence-transformers, residual
  streams, OpenAI embeddings), shared `EmbeddingStore` cache
- `src/annotation/` — markedness/codedness LLM annotator (paper §3.2)
- `src/lexical/ syntactic/ semantic/ distributional/ topical/ interactional/`
  — one folder per distinguishability dimension; each exposes
  `compute_<dimension>(target, baseline, config, context) -> <Dimension>Result`
- `src/usage/` — usage & attitudes from interaction context c (paper §5.2)
- `src/viz/` — one plotting module per section + shared `plot_style.py`;
  each exposes `plot_<dimension>(result, out_dir) -> list[Path]`. Plots are
  minimal: `headline()` short title + one stat line, never verbose subtitles;
  target is always `TARGET_COLOR`, baseline always `BASELINE_SET_COLOR`.
- `src/pipeline/` — registry, dataset runner, explorations, summary; writes
  `runs/{dataset}/{comparison}/{section}/` (+ `codedness/`, `slices/` subdirs)
- `src/results_viewer/` — local HTTP server + single-page UI over `runs/`
  (`uv run python scripts/serve_results_viewer.py`; switch datasets in a
  sidebar, verdict tables with evidence bars, every plot + section JSON)
- `data/synthetic/` — exaggerated fixture dataset (parquet tables + dataset.json)
- `scripts/` — CLI entry points only; business logic lives in `src/`

## Verification

Never report a run as done without opening its actual outputs (JSONs, PNGs).
Log every verification in `VERIFICATION_LOG.md` (WHAT / HOW / RESULT).
