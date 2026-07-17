# Project Guidelines — differential-treatment

Implementation of the paper `paper/differential_treatment.pdf`: a pipeline that
discovers and measures how a target LLM treats two user communities differently,
without explicit identity markers.

## Running

**ALWAYS use `uv run`** — never bare `python`.

```bash
uv run dtreat --help              # CLI: one subcommand per pipeline stage
uv run pytest                     # mock + synthetic tests (no network)
uv run pytest -m live             # opt-in live-API smoke tests (costs money)
```

## Pipeline ↔ paper map

| Stage CLI        | Package                              | Paper   |
|------------------|--------------------------------------|---------|
| `dtreat prompts` | `stages/prompt_collection` (extraction + matching + comparability) | §4.1, Eq 1–3 |
| `dtreat distinguish` | `stages/prompt_distinguishability` (bridge to vendored `distinguish/`) | dist. paper §3.4 |
| `dtreat hypotheses` | `stages/hypothesis_generation`    | §4.2    |
| `dtreat responses`  | `stages/response_collection`      | §4.3, Eq 4 |
| `dtreat score`      | `stages/response_scoring` (judge panels + rubrics) | §4.4, Eq 5–7 |
| `dtreat calibrate-judge` | `stages/response_scoring/judge_calibration_stage` | §5.3 |
| `dtreat analyze`    | `stages/treatment_analysis` (+ input-vs-output comparison) | §4.5, Eq 9–14 |

Each stage reads the previous stage's JSON artifact from a run directory
(`out/runs/<run_id>/`) and writes its own — every intermediate artifact is
human-readable JSON so any result can be audited end to end. Any stage can be
run separately against a hand-authored upstream artifact.

## Realism levels for testing

- **L0 mock**: `mock` backend for helper/target/judge; deterministic, seeded,
  planted differential behavior. End-to-end in seconds, no network.
- **L1 synthetic**: skip LLMs; generate Bernoulli score matrices with known
  effect sizes to exercise the statistics (power, FDR calibration, C2ST).
- **L2 smoke**: real APIs, tiny scale, opt-in (`pytest -m live`).
- **L3 full**: real experiment at configured scale.

## Code conventions (inherited from queering-nlp-bias)

1. Every structured record is a `BaseSchema` dataclass (`common/base_schema.py`)
   — never raw nested dicts across module boundaries.
2. Every `__init__.py` auto-exports its submodules' public symbols.
3. All imports at the top of the file.
4. Unique, multi-word `.py` filenames across the whole repo; no `utils.py`.
5. Keep files small and single-responsibility (~150 lines target).
6. No dead code, no back-compat shims, no TODOs left behind.
7. Update the docs in a folder when you change its code.

## Verification

Never claim a stage/experiment works without running it and looking at the
actual artifacts. Log verifications in `VERIFICATION_LOG.md`.
