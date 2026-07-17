# prompt-distinguishability

A framework for measuring **how distinguishable two prompt sets are** — a
**Target (LGBTQ+)** set vs a **Baseline (cis-heterosexual)** set — implementing
the methods of *Prompts from the Community: LGBTQ+ Uses of LLM Chatbots in
Mental Health, Sexuality, and Relationships* (§3.2, §3.3, §5.2). Each analysis
section has its own statistical test:

| Section | Method | Key output |
|---|---|---|
| **lexical** | Calibrated Marked Words (log-odds with the count-space hybrid Dirichlet prior + per-side calibration of [Mickel et al. 2025](https://arxiv.org/abs/2503.00333) Algorithm 3, from [Monroe et al. 2008](https://doi.org/10.1093/pan/mpn018)); BH-FDR or raw z≥1.96, both always reported; word clouds + calibration-justification plots | marked words per set |
| **syntactic** | [NeuroBiber](https://arxiv.org/abs/2502.18590) 96 binary style features, smoothed log-odds; per-group feature-count distributions | divergent style features |
| **semantic** | [MMD-Fuse](https://arxiv.org/abs/2306.08777) two-sample tests per embedding space: local, OpenAI, and Cohere text embeddings + residual streams from several model families | p-value per embedding space |
| **distributional** | Classifier Two-Sample Test ([Lopez-Paz & Oquab 2017](https://arxiv.org/abs/1610.06545)) per embedding space, author-level CV + permutations; fine-tuned [ModernBERT](https://arxiv.org/abs/2412.13663) | held-out accuracy, ROC/AUC |
| **topical** | Assignment over the paper's 15 survey topics **or** a generated [TopicGPT](https://arxiv.org/abs/2311.01449) taxonomy (intent-level seeds) + Jensen–Shannon divergence | topic/domain JSD |
| **interactional** | Speech acts, self-disclosure depth, anthropomorphization (§3.3.6); per-facet JSD + permutation test | facet share contrasts |
| **usage** | Usage frequency / professional-help frequency / aversion / satisfaction from the survey context (§5.2); author-level Mann-Whitney + BH | attitude & usage gaps |

A separate **annotation** module (§3.2) estimates per-prompt **markedness**
(explicit identity disclosure) and **codedness** (implicit community signal,
[0,1]) with an LLM rubric: `scripts/annotate_prompt_set.py`.

## Quick start

```bash
uv sync

uv run python scripts/run_dataset_pipeline.py --dataset data/synthetic
```

One dataset = one run. The pipeline executes every comparison in the dataset's
manifest with the COMPLETE variant space — every embedder (local
sentence-transformers, OpenAI, Cohere, and residual streams from the Qwen,
Llama, and Gemma families), both classifiers, both assignment backends —
plus per-section explorations. API variants whose key is missing are skipped
and reported by name, never silently dropped.

```
runs/synthetic/
├── summary.json                     # all comparisons, all verdicts, skipped variants
├── config.json
├── comparison_matrix.png            # evidence per test, side by side across comparisons
├── target_vs_baseline/              # one dir per manifest comparison
│   ├── summary_overview.png
│   ├── conditional_{domain,provenance}.png  # marginal vs conditional distinguishability
│   ├── lexical/                     # lexical.json + lexical_*.png + calibration/
│   │   ├── implicit/                # codedness sweep + markedness splits
│   │   │   └── <rerun>/             #   each rerun's full section output
│   │   ├── slices/{gender,race,age,transgender}/<value>/  # full per-slice output
│   │   └── conditional/{domain,provenance}/  # within-content-stratum distinguishability
│   ├── syntactic/ semantic/ distributional/ topical/ interactional/ usage/
└── target_vs_twin/                  # null comparison (main sections only)
```

**Aggregate vs conditional distinguishability.** Every dimension is measured both
marginally (pooled target vs baseline) and *conditionally* — within strata of a
content variable (`domain`, `provenance`, …). Comparing them separates *what* the
groups discuss from *how* they discuss it: a difference that vanishes when you hold
the topic fixed is topic-choice, one that survives (or is only revealed by
conditioning — a Simpson reversal where groups separate within each stratum but not
when pooled) is genuine coded style. See `conditional_{var}.png` and each section's
`conditional/{var}/`.

## Data format

Datasets follow the paper's schema `D = {(x, y, z, d, c)}`
(`src/common/prompt_set_schema.py`, `src/common/dataset_annotations.py`):

- **x** — the prompt text;
- **y** — labels: `lgbtq` (target vs baseline), `markedness` (explicit signal),
  `codedness` (implicit signal strength in [0,1]);
- **z** — the author's self-reported identity (transgender status, gender,
  orientation, pronouns);
- **d** — the author's demographics (race, age, disability, education, income);
- **c** — interaction context (survey topic 1-15, domain MH/GSH/REL, provenance
  real/hypothetical, adoption, frequency scales 1-8, aversion/satisfaction 1-5).

`"*"` marks a prefer-not-to-answer response; ordinal `0` means unrecorded.
A dataset is one directory of **Parquet tables** — `prompts.parquet` (x, y, c +
cohort) and `authors.parquet` (z, d + cohort) — that scale to arbitrarily many
samples, plus `dataset.json` naming the cohorts and the comparisons they
support. The synthetic dataset is a deliberately exaggerated test fixture (see
`data/README.md`): target-vs-baseline should light up every section;
target-vs-target_twin is a null check.

## Configuration

`configs/config.json` holds the complete defaults (documentation by example —
every knob explicit, `_description` keys explain usage). `--config` takes any
partial patch of that shape and may repeat; unknown keys fail loudly. Notable
knobs: `lexical.reference_corpus` (the Calibrated Marked Words calibration
corpus — `wordfreq:<lang>` or a path to a word-frequency JSON),
`semantic.text_embedders` / `residual_models`, `distributional.classifiers`,
`topical.assignment_backends`, and the `explorations` block (codedness
thresholds, markedness splits, identity slices).

## Validation on real datasets

Beyond the synthetic fixture, `scripts/convert_*.py` import open datasets (Blog
Authorship, Reddit-L2, PRISM, PAN-2017, TwitterAAE, Multi-VALUE, ThoughtTrace,
WildChat) into the parquet schema — each with a negative-control null-split.
Running the pipeline on them gives **above-chance C2ST wherever the groups are
established as separable, against a matched null at ≈0.50** — the honest
validation, since (as every source paper confirms) the "published" figures are
different quantities (6-way vs binary, whole-blog vs segment, k-fold profiling vs
held-out C2ST) and most of these papers report no separability metric at all.
**`docs/FINDINGS.md`** reports the analysis with the true per-paper comparators
and the module-sensitivity findings (C2ST robust where MMD over-rejects on
heterogeneous populations; embedding tests blind to meaning-preserving dialect).

## GPU runs (vast.ai)

Heavy variants (ModernBERT fine-tuning, ~9B residual-stream models) run on a
rented GPU box via `cloud/` — see `cloud/README.md` for the launch → sync →
run → sync-back → destroy flow. API keys never leave the local machine.

## Repository layout

```
src/
├── common/          # dataset schema (x,y,z,d,c), config, stats, file I/O
├── inference/       # embedding runners, residual-stream extractor, cache
├── annotation/      # markedness/codedness LLM annotator (paper 3.2)
├── lexical/ syntactic/ semantic/ distributional/ topical/ interactional/
├── usage/           # usage & attitudes from context c (paper 5.2)
├── viz/             # one plots module per section + shared plot style
└── pipeline/        # registry, orchestration, summary schema
data/<name>/         # datasets: parquet tables (prompts, authors) + manifest
scripts/             # pipeline CLI, annotation CLI, dataset converters
docs/                # FINDINGS.md (real-dataset analysis), design/research notes
cloud/               # vast.ai GPU tooling
```

Conventions (see `CLAUDE.md`): `uv run` for everything, `BaseSchema` dataclasses
for all structured data, auto-exporting `__init__.py`, one focused module per
concern. Inference utilities are adapted from
[queering-nlp-bias](https://github.com/unrulyabstractions/queering-nlp-bias).
