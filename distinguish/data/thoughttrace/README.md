# ThoughtTrace (converted)

Real user turns from human-LLM conversations, converted to the paper's
D=(x,y,z,d,c) table format (`src/common/dataset_tables.py`). Regenerate with
`uv run python scripts/convert_thoughttrace.py --seed 0` (idempotent).

## Provenance and license

- Source: Hugging Face `SCAI-JHU/ThoughtTrace` (`ThoughtTrace.jsonl`, ~30 MB),
  license **CC-BY-4.0**. Paper: Jin et al. 2026, "ThoughtTrace: Understanding
  User Thoughts in Real-World LLM Interactions" (arXiv:2605.20087). Cite when used.
- Raw corpus: 2,155 conversations from 1,058 users across 20 LLMs; user turns
  carry self-reported *reasons*, and each user has a gold exit survey
  (age, gender, education, occupation, AI-usage frequency, purposes).
- Author id = the `user{N}` prefix of the conversation `id`.

## Cohorts and sampling (seed=0)

- y = survey-reported gender: `target` = Female (lgbtq flag 1 = generic y),
  `baseline` = Male (0). Non-binary / prefer-not-to-say users (9) are excluded.
- x = **user turns** (all turns, not just conversation openers), stripped,
  interior whitespace (newlines/tabs/repeated spaces) collapsed to single
  spaces, minimum 30 chars, corpus-wide exact-duplicate turns dropped, and
  **truncated to the first 120 words** (turns longer than that are cut).
- Seeded subsample: 100 authors per side, up to 4 turns per
  author (author structure preserved; every prompt's author is in authors.parquet).
- `null_split_a` / `null_split_b`: negative control — 60+60
  held-out **female** authors (disjoint from `target`), split at random with the
  same seed. Expect C2ST ~= 0.5, zero BH survivors, null MMD.

## Field mapping (z, d, c)

- `gender`: survey Female -> ["woman"], Male -> ["man"].
- `age`: survey integer age mapped to brackets 18-24 / 25-34 / 35-44 / 45-54 /
  55-64 / 65+ ("" if missing).
- `education`: survey value lowercased (graduate / undergraduate / high school /
  other; "" if missing).
- `llm_freq`: survey `frequency` stored **raw** — it is ThoughtTrace's own
  1-5 AI-usage-frequency scale, NOT this repo's F 1-8 scale
  (`src/common/dataset_annotations.py`); 0 = missing. Do not read it against
  `FREQUENCY_SCALE` labels.
- `domain`: the survey `purposes` free-text list (comma-separated), NOT the
  MH/GSH/REL catalog; `topic_id` stays 0. Topical/usage modules should treat
  these as unannotated.
- Unrecorded everywhere else: `transgender`/`disability`/`income` "", `orientation`/
  `pronouns`/`race` [], `markedness` 0, `codedness` 0.0, ordinals 0.
- `provenance` = "thoughttrace" (dataset tag, not the paper's real/hyp flag).

## Caveats

- **Observer-aware study**: participants knew their turns and thoughts were
  recorded for research, which may shift register vs organic LLM usage.
- No published gender-separability benchmark on this corpus
  (`expected_accuracy` 0.0 = none); treat `target_vs_baseline` as exploratory
  (modality-match role in docs/ITERATION4_PLAN.md's validation design).
- Turns are mid-conversation and often short (median ~12 words); several turns
  from the same conversation are not independent samples.
- The paper's headline numbers concern *thought* prediction, not demographic
  classification.
