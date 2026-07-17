# PRISM (gender) — real LLM prompts by gold-survey gender

Converted from **HannahRoseKirk/prism-alignment** (Kirk et al., 2024) by
`scripts/convert_prism.py` (seed 0). This is real user data: human-written
text is **CC-BY-4.0**; model responses (not used here) are CC-BY-NC-4.0. Only the
human first-turn prompts are exported.

## What each field is

- **x (text)** = the **first-turn** user prompt (`utterances.jsonl`, `turn==0`),
  **restricted to `conversation_type == "unguided"`** (excludes the values-guided
  and controversy-guided conversation types).
- **y** = gold-survey gender: `lgbtq` (our generic y flag) is **1 for the female
  cohort (target)**, **0 for the male cohort (baseline)**. Gender is the
  self-reported survey answer, not inferred from text.
- **z/d (author)**: `gender` mapped to our gender list (Female->["woman"],
  Male->["man"]); `age` mapped to brackets (`25-34 years old`->`25-34`, up to
  `65+`); `race` = the survey `ethnicity.categorised` label ("Prefer not to say"
  and "Other" -> `[]`); `education` = the survey education string ("Prefer not to
  say" -> `"*"`). Not surveyed in PRISM and left unrecorded: `transgender` (""),
  `orientation` ([]), `pronouns` ([]), `disability` (""), `income` (""). **Religion
  is intentionally dropped** (no matching column).
- **c (context)**: only `llm_freq` is filled, mapped from the survey
  `lm_frequency_use`. All other context is unrecorded (0 / ""), so the topical and
  usage sections skip gracefully. `provenance` is left "" — PRISM has no
  recalled-vs-hypothetical (c^prov) distinction.

### `lm_frequency_use` -> `llm_freq` (FREQUENCY_SCALE 1-8) mapping

PRISM's five frequency buckets do not map one-to-one onto our 8-point scale; the
mapping below is monotonic-non-decreasing and documented for transparency:

| survey `lm_frequency_use` | `llm_freq` | scale label |
|---|---|---|
| Less than one a year | 1 | very rarely |
| Once per month | 3 | ~once/mo |
| More than once a month | 4 | ~once/wk (tie) |
| Every week | 4 | ~once/wk (tie) |
| Every day | 5 | ~once/day |
| missing / null | 0 | unrecorded |

## Sampling (seed 0)

1. Keep first turns of **unguided** conversations only; whitespace-normalise.
2. Drop prompts with **< 5 words** (trivial "hi"/"hello") and
   **de-duplicate** exact (lowercased) prompt texts globally.
3. **Truncate** prompts longer than **120 words** to their first
   120 words (only 3 unguided first-turn prompts exceed this), and collapse
   interior whitespace (newlines/tabs/repeated spaces) to single spaces.
4. Among users with >=1 usable prompt, seeded-shuffle and take
   **80 female (target) + 80 male
   (baseline)** authors, **<= 4 prompts each** (author
   structure preserved: every prompt's author is in `authors.parquet`).
5. **null_split_a / null_split_b**: seeded random split of the 80
   target (female) authors into two disjoint halves — the required **negative
   control** (`null_control` comparison, expectation "null").
6. **target_n12/24 & baseline_n12/24**: seeded author subsamples of the main
   cohorts for the **power-analysis** comparisons (exploratory, expectation "").

Author ids are cohort-prefixed (`target__user123`) so the power/null cohorts are
independent row sets (a user may appear in several cohorts under distinct ids).

## Per-cohort counts

```
              prompts  authors
cohort                        
baseline          159       80
baseline_n12       20       12
baseline_n24       48       24
null_split_a       74       40
null_split_b       75       40
target            149       80
target_n12         27       12
target_n24         45       24
```

## Comparisons

- `target_vs_baseline` — female (y=1) vs male (y=0). **Exploratory**: no published
  per-prompt gender-separability benchmark for PRISM (expectation "").
- `null_control` — `null_split_a` vs `null_split_b`. **Negative control**, expect
  ~chance C2ST, 0 BH survivors, null MMD (expectation "null").
- `target_vs_baseline_n12`, `target_vs_baseline_n24` — power-analysis subsamples
  (12 / 24 authors per side), exploratory.

## Provenance, license, caveats

- **Provenance**: HuggingFace `HannahRoseKirk/prism-alignment` (survey.jsonl +
  utterances.jsonl). Downloaded into a scratch dir, never committed.
- **License**: human text CC-BY-4.0 (this export contains only human prompts).
- **Caveats**: (a) prompts are short real questions (median ~9 words) — per-prompt
  gender separability is expected to be **low**; author-level aggregation is the
  fair comparison. (b) Gender is self-reported; the label is a survey answer, not a
  text property. (c) Non-binary and "prefer not to say" users are excluded from the
  binary female/male contrast. (d) Long-prompt truncation (>120
  words) affects only 3 prompts.
