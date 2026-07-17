# configs/ — one complete config, patched by partial files

`config.json` is the COMPLETE-analysis default spelled out in full —
documentation by example. It mirrors the dataclass defaults in
`src/common/run_config.py` exactly, so a run with no `--config` flag is
identical to a run with `--config configs/config.json`. There are no
per-section preset files anymore: to change anything, write a small patch
file naming only the keys you want changed.

## How patches compose

```bash
uv run python scripts/run_dataset_pipeline.py --dataset data/synthetic \
    --config my_patch.json --config another_patch.json
```

`--config` may be passed multiple times; files apply left to right on top of
the defaults. Patching is field-wise and recursive:

- a top-level key patches that `PipelineConfig` field;
- a nested object (e.g. `"lexical": {...}`) patches only the fields it names
  inside that section — every other field keeps its default;
- lists replace wholesale, they are never merged — to run only local
  embedders, write out the whole shortened list;
- keys starting with `_` (like `_description`) are documentation for humans:
  the loader ignores them, so use `_description` to record when a patch file
  should be used;
- unknown keys fail loudly, so a patch can never silently drift from the
  schema.

Example patch — offline-only semantic tests plus a stricter lexical FDR:

```json
{
    "_description": "Local embedders only; stricter lexical FDR.",
    "semantic": {
        "text_embedders": ["sentence-transformers/all-MiniLM-L6-v2"],
        "residual_models": []
    },
    "lexical": {"fdr_alpha": 0.01}
}
```

## Provider specs and skipped variants

Every model-list entry uses one grammar: `openai:<model>`, `cohere:<model>`,
anything else is a sentence-transformers model name. API-backed variants
whose key (`OPENAI_API_KEY`, `COHERE_API_KEY`) is missing are skipped
gracefully, never crash the run: each section records them in its result's
`skipped_variants` list (e.g. `"cohere:embed-v4.0 (COHERE_API_KEY not set)"`)
and the run summary aggregates them.

## Key knobs

### Top level
- `run_name` — run directory name ("" → the dataset name).
- `dimensions` — which sections run (lexical, syntactic, semantic,
  distributional, topical, interactional).
- `include_usage_attitudes` — also run the usage & attitudes section (§5.2).
- `random_seed` — seeds every permutation/CV RNG.

### lexical — calibrated marked words (Monroe log-odds + BH FDR)
- `reference_corpus` — the calibration side of the hybrid Dirichlet prior:
  `wordfreq:<lang>` (default `wordfreq:en`, built-in frequency lists) **or** a
  filesystem path to a JSON file mapping word → relative frequency (a
  domain-specific reference corpus). Words missing from the corpus get a
  1e-9 floor; the result JSON echoes the corpus used.
- `english_prior_weight` — weight on the reference corpus vs the combined
  target+baseline corpus in the prior (0 = corpus only, 1 = reference only).
- `prior_strength` — total Dirichlet prior mass.
- `min_word_count` — combined count a word needs to enter the vocabulary.
- `fdr_alpha`, `top_words_reported` — BH level; words listed per side.

### syntactic — NeuroBiber features via smoothed log-odds
- `model_name`, `batch_size` — the NeuroBiber extractor.
- `smoothing_count` — Haldane-Anscombe smoothing for absent/universal features.
- `fdr_alpha`, `top_features_reported`.

### semantic — MMD-Fuse two-sample tests per embedding space
- `text_embedders` — provider specs (see grammar above).
- `residual_models` — HF causal LMs whose residual streams are tested.
- `residual_layer_fraction` — which layer depth to read (0–1 of the stack).
- `significance_alpha`.

### distributional — C2ST (classifier two-sample test)
- `embedders` — provider specs for the linear classifier's features.
- `classifiers` — any of `linear`, `modernbert` (fine-tuned end to end).
- `cv_folds`, `n_permutations`, `significance_alpha`.
- `modernbert_model_name`, `modernbert_epochs`, `modernbert_learning_rate`.

### topical — survey-topic assignment + Jensen-Shannon divergence
- `assignment_backends` — any of `embedding`, `openai:<model>`.
- `embedding_model`, `n_permutations`, `significance_alpha`.

### interactional — speech acts, disclosure, anthropomorphization
- `annotation_backends` — any of `embedding`, `openai:<model>`.
- `embedding_model`, `n_permutations`, `significance_alpha`.

### usage — interaction-context scales (author-level Mann-Whitney)
- `fdr_alpha`.

### explorations — filtered reruns of every section
- `run_implicit_breakdown` — rerun each section with target prompts restricted
  to `codedness >= t` for each `t` in `codedness_thresholds`.
- `include_markedness_splits` — extra reruns splitting on markedness
  (implicit-only / marked-only prompts), per H1/H2.
- `run_slices` — rerun each section per identity/demographic slice.
- `slices` — list of `{name, facet, z_field, op, value, apply_to}` where `op`
  is `eq` | `in` | `contains` | `not_contains` and `apply_to` is `both` |
  `target` (e.g. transgender only exists in the target set).
- `run_conditional` — conditional distinguishability within strata of each
  variable in `conditioning_variables` (marginal vs conditional evidence).
- `full_outputs` — also write each rerun's full section output directory.
- `n_permutations` — lighter permutation budget for reruns.
- `min_prompts_per_side` — a filtered rerun below this is skipped (and
  recorded), never run underpowered.

Exploration reruns always use a lightened per-section config (local embedder,
linear classifier, reduced permutations) — the full variant space belongs to
the main run.
