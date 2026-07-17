# differential-treatment

Implementation of **"Towards Local Bias Mitigation: Discovering How LLMs Treat
LGBTQ+ Users Differently, Without Explicit Markers"**
(`paper/differential_treatment.pdf`).

A pipeline that discovers and measures how a target LLM treats two user
communities differently in a specific deployment setting — without relying on
explicit identity markers. A helper LLM proposes interpretable axes of
possible differential treatment, the target LLM is sampled on
instruction-comparable prompts from both communities, an LLM judge scores
every response along every axis, and the resulting behavior distributions are
compared with permutation tests, FDR control, information-theoretic ranking,
and a classifier two-sample test.

Code conventions and infrastructure are inherited from
[queering-nlp-bias](https://github.com/unrulyabstractions/queering-nlp-bias).

## Quick start

```bash
uv sync --dev

# fictional case-study data (invented, mirrors the paper's running example)
uv run python scripts/generate_case_study_prompts.py

# full pipeline against deterministic mock LLMs with planted bias — no keys, ~seconds
uv run dtreat run-all -c configs/mock_biased.json

# read the result
cat out/runs/mock_biased/stage5_analysis/analysis_summary.md

# explore every stage in the browser
uv run dtreat serve            # → http://127.0.0.1:8321
```

For a real experiment, put `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` in `.env`
and use `configs/live_smoke.json` as a template (`uv run dtreat estimate-cost
-c <config>` first).

## Pipeline ↔ paper

```
prompts ──▶ hypotheses ──▶ responses ──▶ score ──▶ analyze
 §4.1         §4.2           §4.3         §4.4      §4.5
```

| CLI stage | What it does | Paper |
|-----------|--------------|-------|
| `dtreat prompts` | Load both community prompt sets, validate, check **instruction comparability** (TV distance + χ² over instruction frequencies) | §3.1, §4.1, Eq 1–3 |
| `dtreat hypotheses` | Helper LLM proposes interpretable yes/no **axes of treatment**; seeds + literature notes supported; robust JSON parsing + dedup | §4.2 |
| `dtreat responses` | Sample K responses per prompt from the target LLM; refusals recorded as first-class data; resumable | §4.3, Eq 4 |
| `dtreat score` | LLM judge scores every response on every axis (community never disclosed); per-response or per-axis calls | §2.3, §4.4, Eq 5–7 |
| `dtreat analyze` | Rate gaps Δ_j, prompt-level **permutation tests** + **Benjamini–Hochberg** FDR, **mutual-information ranking** I_j, treatment profiles + **D_π** (KL, bits), **C2ST**, refusal analysis | §4.5, Eq 9–14 |

Every stage reads the previous stage's artifact from `out/runs/<run>/` and
writes its own — all human-readable JSON/JSONL, so any stage can be run
separately against hand-authored inputs and any result can be audited end to
end (the paper's *Verifiable* property).

```
out/runs/<run_name>/
├── experiment_config.json        # provenance snapshot
├── llm_cache/                    # response cache → free stage resumability
├── llm_trace.jsonl               # every LLM call: tokens, cost, latency, refusals
├── stage1_prompts/prompt_sets.json
├── stage2_hypotheses/hypothesis_set.json
├── stage3_responses/{responses.jsonl, collection_manifest.json}
├── stage4_scores/{scored_responses.jsonl, scoring_manifest.json}
├── stage5_analysis/{analysis_report.json, analysis_summary.md}
└── quarantine/                   # failed calls, never silently dropped
```

## Debugging & diagnostics

```bash
uv run dtreat status   --run-dir out/runs/mock_biased   # which artifacts exist
uv run dtreat validate --run-dir out/runs/mock_biased   # cross-stage consistency
uv run dtreat inspect  out/runs/mock_biased             # summarize any artifact
uv run dtreat trace    --run-dir out/runs/mock_biased --errors   # LLM call trace
uv run dtreat estimate-cost -c configs/live_smoke.json  # $ before you spend it
uv run dtreat serve                                     # web UI (below)
```

`dtreat serve` gives one debug view per stage: instruction-distribution
comparison (stage 1), hypotheses + raw helper reply (2), filterable response
browser + length histograms (3), per-axis rate bars + prompt×axis heatmap +
raw judge replies (4), and the full analysis: diverging Δ chart with
click-through **permutation-null histograms**, MI ranking, C2ST/D_π tiles
(5) — plus the LLM trace with grep and error filters.

## Testing at graded realism

| Level | What runs | Command |
|-------|-----------|---------|
| **L0 mock** | Deterministic mock helper/target/judge with *planted* per-community behavior rates; end-to-end must recover the planted gaps and stay quiet on the null profile | `uv run pytest tests/integration/test_mock_pipeline_end_to_end.py` |
| **L1 synthetic** | No LLMs: Bernoulli score matrices with known effect sizes; power, FDR calibration, C2ST calibration; the statistics reproduce the paper's worked example (D_π = 2.37 bits, I₁ = 0.46) | `uv run pytest tests/unit tests/integration/test_synthetic_statistics.py` |
| **L2 live smoke** | Real APIs, tiny scale (< $0.01): helper proposes parseable axes, judge follows the verdict protocol | `uv run pytest -m live` |
| **L3 full** | A real experiment at configured scale | `uv run dtreat run-all -c <your config>` |

The mock world mirrors the paper's fictional fitness/nutrition case study:
`mock:target:biased` plants known rate gaps keyed off community cue words in
prompts (the paper's premise — cues without explicit markers), and
`mock:target:null` behaves identically for both communities, so false
positives are detectable. `mock:judge:noisy` adds seeded verdict flips.

## Repository layout

```
src/dtreat/
├── common/        # BaseSchema, JSON/JSONL IO, seeds, logging, information math,
│                  # judge protocol, robust JSON extraction   (ported from base repo)
├── llm/           # chat backends (anthropic/openai/mock), retry, cache,
│                  # pricing, tracing, parallel execution
├── stages/        # one package per pipeline stage (schemas + logic + runner)
├── pipeline/      # experiment config, run-dir layout, stage registry, CLI
├── diagnostics/   # inspect / validate / trace / estimate-cost
└── server/        # FastAPI debug server + static UI
configs/           # mock_biased, mock_null, live_smoke experiment configs
data/prompts/      # generated fictional case-study prompt sets
scripts/           # generate_case_study_prompts.py
tests/             # unit / integration / live  (L0–L2)
```

## Notes

- The case-study prompts and communities are **fictional and invented** for
  development, mirroring the paper's disclaimer; they are not real user data
  and results on them are not empirical findings.
- Whether a measured distributional difference is *harmful* is outside the
  framework's scope (paper §5): the target community is best positioned to
  judge that.
- `VERIFICATION_LOG.md` records what has actually been verified and how.
