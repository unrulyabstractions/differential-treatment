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
              ┌─▶ hypotheses (§4.2) ─┐
prompts (§4.1)┤                      ├─▶ score (§4.4) ─▶ analyze (§4.5)
              └─▶ responses  (§4.3) ─┘
```

Fig. 1 is a DAG: hypothesis generation and response collection are parallel
branches joining at scoring. `run-all` executes responses first (a valid
topological order) so behavior-grounded hypothesis methods can observe real
responses on a fresh run.

| CLI stage | What it does | Paper |
|-----------|--------------|-------|
| `dtreat prompts` | Load both community prompt sets, validate; **instruction annotation** either provided or LLM-extracted (two-pass: phrase extraction → cross-community canonicalization); optional **frequency matching** by subsampling to identical instruction distributions; **comparability check** (TV distance + χ²) | §3.1, §4.1, Eq 1–3 |
| `dtreat distinguish` | **Input-side distinguishability** of the two prompt sets via the vendored `distinguish/` pipeline (lexical / syntactic / semantic / distributional / topical dimensions, `paper/distinguishability.pdf`) | dist. §3.4 |
| `dtreat hypotheses` | ALL hypothesis-generation methods run by default — **a-priori** (zero_context, two_stage), **literature** (bundled + **RAG-retrieved arXiv abstracts** for the pair/domain), **prompt-subsample grounding**, **behavior-subsample grounding** (response_grounded), and **seeds** — union scored once, every axis tagged with the method(s) that proposed it; per-method comparison lands in the analysis report | §4.2 |
| `dtreat responses` | Sample K responses per prompt from the target LLM; refusals recorded as first-class data; resumable | §4.3, Eq 4 |
| `dtreat score` | **Judge panel** (one or many models, majority/unanimous/any aggregation) scores every response on every axis with rubrics (community never disclosed); per-judge verdicts retained | §2.3, §4.4, Eq 5–7 |
| `dtreat calibrate-judge` | Judge validation: pairwise **Cohen's κ**, panel **Fleiss' κ**, self-consistency flip rates under reseeding, optional gold-label accuracy | §5.3 |
| `dtreat analyze` | Rate gaps Δ_j, prompt-level **permutation tests** + **Benjamini–Hochberg** FDR, **mutual-information ranking** I_j, treatment profiles + **D_π** (KL, bits), **C2ST**, refusal analysis, and the **input-vs-output comparison** (prompt legibility vs behavior separability, signal-usage ratio) | §4.5, Eq 9–14; dist. §5.2 |

Real-data testing: `scripts/collect_reddit_prompts.py` builds LOCAL-ONLY
prompt sets from curated, research-verified public Reddit posts via the
PullPush archive (gitignored — community data stewardship; excluded: minors,
removed posts, usernames never collected).

Every stage reads the previous stage's artifact from `out/runs/<run>/` and
writes its own — all human-readable JSON/JSONL, so any stage can be run
separately against hand-authored inputs and any result can be audited end to
end (the paper's *Verifiable* property).

```
out/runs/<run_name>/
├── experiment_config.json        # provenance snapshot
├── llm_cache/                    # response cache → free stage resumability
├── llm_trace.jsonl               # every LLM call: tokens, cost, latency, refusals
├── stage1_prompts/{prompt_sets.json, input_distinguishability.json}
├── stage2_hypotheses/hypothesis_set.json
├── stage3_responses/{responses.jsonl, collection_manifest.json}
├── stage4_scores/{scored_responses.jsonl, scoring_manifest.json,
│                  judge_calibration.json}
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
distinguish/       # vendored prompt-distinguishability project (input side)
configs/           # mock_biased, mock_null, mock_panel, live_smoke, real_reddit
data/prompts/      # generated case-study sets + real_* (LOCAL ONLY, gitignored)
scripts/           # generate_case_study_prompts.py, collect_reddit_prompts.py
tests/             # unit / integration / live  (L0–L2)
```

## Findings so far (2026-07-17, local real-data study)

From `out/runs/real_study` — 60 real Reddit prompts/community (36/side after
instruction matching), gpt-4o-mini as target, 24 union axes from 4 helper
conditions, 216 responses, 5-judge cross-provider panel. All numbers
independently recomputed from raw artifacts.

- **Differential treatment is real and directional**: 21/24 axes significant
  (BH-FDR 0.05), every significant Δ positive — LGBTQ+-voiced prompts get
  MORE inclusive language (+0.34), more identity-integrated and body-diverse
  advice (+0.27/+0.34), more body-positivity framing (+0.33), more
  mental-health emphasis (+0.27), more emotional support (+0.32), and more
  hedging on sensitive topics (+0.21); plus a non-significant trend of less
  purely practical advice (−0.10). Whether this framing serves or patronizes
  is the community's call (paper §5) — the pipeline's job is to surface it.
- **The finding is judge-invariant**: 12 axes are significant under ALL five
  judges (gpt-4o-mini, gpt-4.1-mini, claude-haiku-4-5, gemini-2.5-flash,
  gemini-3.5-flash); pairwise Cohen's κ 0.61–0.79, with gemini-3.5-flash the
  most conservative judge.
- **Inputs are legible**: prompt-side C2ST 0.889, 13/20 distinguishability
  tests significant; ~57% of the input signal carries into behavior
  separability.
- **Helper conditions matter**: two_stage (separate brainstorm → question
  formation) captured the most information (0.66 bits) vs zero_context
  (0.57), grounded (0.44), literature (0.39) — and conditions propose almost
  entirely disjoint axes, so unioning conditions materially widens coverage.
- **Method lessons**: LLM canonicalization of instructions needs an explicit
  merge budget and enough output tokens (truncation silently destroys
  frequency matching); Gemini judges need thinking disabled or replies starve;
  complete-case C2ST loses most rows under a 24-axis panel with ties (158/216
  dropped) — per-axis stats carry the evidence there.

### Cross-group comparison (same domain, same target model)

The pipeline is group-agnostic (`data/group_pairs/` specs + one config per
pair; `dtreat compare-runs` for the side-by-side). Three pairs in the same
fitness-advice deployment, gpt-4o-mini as target, ~40 matched prompts/side:

| pair | input C2ST | output significant axes | behavior C2ST |
|------|-----------:|------------------------:|--------------:|
| lgbtq vs cishet | 0.889 | **21/24** | 0.722 |
| women vs men | 0.738 | 0/8, and 0/32 after a 4-strategy union | **0.700** [0.55, 0.82] on 8 axes — separable |
| over40 vs young | 0.613 | 0/8 | 0.542 — chance |

Reading: the model's differential behavior tracks the *kind* of group, not
just input legibility. LGBTQ+ voice triggers heavy explicit adaptation
(21 significant axes). Gender-coded voice produces behavior that IS
separable (8-axis C2ST above chance) but no single axis reaches
significance even after widening to 32 axes from four helper strategies —
weak, diffuse differentiation spread across many small effects (best
candidates: less recovery emphasis Δ=−0.18, more strength-training push
+0.16, all q≈0.43): the Λ* discovery gap made concrete. Age-coded voice
produces essentially no behavioral difference at this scale.

## Method lineage

Beyond the two papers in `paper/`, the pipeline adopts three insights from
Eloundou et al., *First-Person Fairness in Chatbots* (arXiv:2410.19803):
**response-grounded axis discovery** (`helper-study` condition
`response_grounded` — axes enumerated from observed response differences,
their Bias Enumeration Algorithm, which targets exactly the case where
behavior is separable but a-priori axes miss it), **per-task partitioning**
(within-instruction stratum gaps in the analysis report — bias concentrated
in one kind of ask no longer dilutes in the aggregate), and
**dimension-dependent judge reliability** (per-axis inter-judge κ joined into
the analysis; axes with κ < 0.4 are flagged as unreliable). Their name-based
counterfactual injection is deliberately NOT adopted: this pipeline's premise
(both papers') is naturally-voiced real prompts, not artificial identity
injection.

## Notes

- The case-study prompts and communities are **fictional and invented** for
  development, mirroring the paper's disclaimer; they are not real user data
  and results on them are not empirical findings.
- Whether a measured distributional difference is *harmful* is outside the
  framework's scope (paper §5): the target community is best positioned to
  judge that.
- `VERIFICATION_LOG.md` records what has actually been verified and how.
