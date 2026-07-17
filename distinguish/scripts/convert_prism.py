"""Convert the PRISM alignment survey into our D=(x,y,z,d,c) Parquet format.

Source: HannahRoseKirk/prism-alignment (HF dataset, human text CC-BY-4.0).
  survey.jsonl      one row per user: gold demographics (gender, age, ...).
  utterances.jsonl  one row per turn: first-turn (turn==0) user_prompt = x.

Design (docs/ITERATION4_PLAN.md, "Validation design" -> PRISM):
  y = gold-survey gender, female (target, y=1) vs male (baseline, y=0).
  x = first-turn user prompt, RESTRICTED to conversation_type == "unguided".
  z/d filled from the survey (age brackets, gender list, ethnicity->race list,
  education); c.llm_freq mapped from lm_frequency_use. religion is dropped.
  No published separability benchmark -> exploratory (expectation "").

Cohorts:
  target / baseline            main female-vs-male contrast (~80 authors/side).
  null_split_a / null_split_b  seeded random author split WITHIN the female
                               (target) group -> negative control, expect null.
  target_n12 / baseline_n12    seeded 12-author-per-side subsample of the main
  target_n24 / baseline_n24    cohorts -> power-analysis comparisons.

This script is idempotent: it downloads the raw jsonl into a scratch dir (never
into the repo, HF cache reused across runs) and rewrites data/prism/ each run.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from src.common.dataset_tables import (
    CohortSpec,
    ComparisonSpec,
    DatasetManifest,
    PromptDataset,
)
from src.common.file_io import save_json

HF_REPO = "HannahRoseKirk/prism-alignment"
DATASET_NAME = "prism"
DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_datasets" / "prism"

# Sampling knobs (all seeded; documented in the generated README).
MAIN_AUTHORS_PER_SIDE = 80
MAX_PROMPTS_PER_AUTHOR = 4
MIN_PROMPT_WORDS = 5
MAX_PROMPT_WORDS = 120
POWER_SIZES = (12, 24)

# --- value mappings (survey category -> our schema) -----------------------

_GENDER_TO_LIST = {
    "Male": ["man"],
    "Female": ["woman"],
    "Non-binary / third gender": ["nonbinary"],
    "Prefer not to say": [],
}
_AGE_TO_BRACKET = {
    "18-24 years old": "18-24",
    "25-34 years old": "25-34",
    "35-44 years old": "35-44",
    "45-54 years old": "45-54",
    "55-64 years old": "55-64",
    "65+ years old": "65+",
    "Prefer not to say": "*",
}
# lm_frequency_use -> FREQUENCY_SCALE index (dataset_annotations.FREQUENCY_SCALE,
# 1-8: very rarely / ~once-yr / ~once-mo / ~once-wk / ~once-day / ...). PRISM's
# five buckets do not map 1:1; this is a documented, monotonic-non-decreasing
# approximation (README notes the "More than once a month" / "Every week" tie).
_LM_FREQ_TO_ORDINAL = {
    "Less than one a year": 1,  # rarer than yearly -> "very rarely"
    "Once per month": 3,  # ~once/mo
    "More than once a month": 4,  # between monthly and weekly -> ~once/wk
    "Every week": 4,  # ~once/wk
    "Every day": 5,  # ~once/day
}


def _download_raw(raw_dir: Path) -> tuple[Path, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    survey = hf_hub_download(
        HF_REPO, "survey.jsonl", repo_type="dataset", local_dir=raw_dir
    )
    utterances = hf_hub_download(
        HF_REPO, "utterances.jsonl", repo_type="dataset", local_dir=raw_dir
    )
    return Path(survey), Path(utterances)


def _load_survey(path: Path) -> dict[str, dict]:
    return {r["user_id"]: r for r in (json.loads(line) for line in path.open())}


def _clean_text(raw: str) -> str | None:
    """Whitespace-normalise, drop trivial prompts, truncate to MAX_PROMPT_WORDS."""
    words = raw.split()
    if len(words) < MIN_PROMPT_WORDS:
        return None
    if len(words) > MAX_PROMPT_WORDS:
        words = words[:MAX_PROMPT_WORDS]
    return " ".join(words)


def _first_turn_prompts(utterances_path: Path) -> dict[str, list[dict]]:
    """user_id -> list of {conversation_id, text} for unguided first turns.

    De-duplicates exact (lowercased) prompt texts globally, preserving order.
    """
    first_by_conv: dict[str, dict] = {}
    for line in utterances_path.open():
        u = json.loads(line)
        if u["turn"] != 0 or u["conversation_type"] != "unguided":
            continue
        first_by_conv.setdefault(u["conversation_id"], u)

    seen_texts: set[str] = set()
    by_user: dict[str, list[dict]] = {}
    for u in first_by_conv.values():
        text = _clean_text(u["user_prompt"])
        if text is None:
            continue
        key = text.lower()
        if key in seen_texts:
            continue
        seen_texts.add(key)
        by_user.setdefault(u["user_id"], []).append(
            {"conversation_id": u["conversation_id"], "text": text}
        )
    return by_user


def _author_row(user_id: str, cohort: str, survey_row: dict, out_id: str) -> dict:
    """z/d from the survey; unknown fields -> defaults ('' / [] / '*')."""
    gender = _GENDER_TO_LIST.get(survey_row["gender"], [])
    age = _AGE_TO_BRACKET.get(survey_row["age"], "*")
    eth = survey_row["ethnicity"]["categorised"]
    race = [] if eth in ("Prefer not to say", "Other") else [eth]
    education = survey_row["education"]
    if education == "Prefer not to say":
        education = "*"
    return {
        "author_id": out_id,
        "cohort": cohort,
        "transgender": "",  # not surveyed in PRISM
        "gender": gender,
        "orientation": [],  # not surveyed
        "pronouns": [],  # not surveyed
        "race": race,
        "age": age,
        "disability": "",  # not surveyed
        "education": education,
        "income": "",  # not surveyed
    }


def _prompt_row(
    prompt_id: str,
    out_author_id: str,
    cohort: str,
    text: str,
    lgbtq: int,
    llm_freq: int,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "author_id": out_author_id,
        "cohort": cohort,
        "text": text,
        "lgbtq": lgbtq,
        "markedness": 0,
        "codedness": 0.0,
        "topic_id": 0,
        "domain": "",
        "provenance": "",  # PRISM has no recalled-vs-hypothetical (c^prov) label
        "adoption": 0,
        "general_freq": 0,
        "llm_freq": llm_freq,
        "professional_freq": 0,
        "aversion": 0,
        "satisfaction": 0,
    }


def _eligible_authors(
    survey: dict[str, dict], prompts_by_user: dict[str, list[dict]], gender: str
) -> list[str]:
    """User ids of the given gender that have >=1 usable unguided prompt."""
    return sorted(
        uid
        for uid, ps in prompts_by_user.items()
        if ps and survey.get(uid, {}).get("gender") == gender
    )


def _emit_cohort(
    user_ids: list[str],
    cohort: str,
    lgbtq: int,
    survey: dict[str, dict],
    prompts_by_user: dict[str, list[dict]],
    counter: list[int],
) -> tuple[list[dict], list[dict]]:
    """Build author + prompt rows for one cohort (up to MAX_PROMPTS_PER_AUTHOR)."""
    author_rows, prompt_rows = [], []
    for uid in user_ids:
        out_author_id = f"{cohort}__{uid}"
        srow = survey[uid]
        author_rows.append(_author_row(uid, cohort, srow, out_author_id))
        llm_freq = _LM_FREQ_TO_ORDINAL.get(srow.get("lm_frequency_use"), 0)
        for p in prompts_by_user[uid][:MAX_PROMPTS_PER_AUTHOR]:
            pid = f"{DATASET_NAME}_{counter[0]:05d}"
            counter[0] += 1
            prompt_rows.append(
                _prompt_row(pid, out_author_id, cohort, p["text"], lgbtq, llm_freq)
            )
    return author_rows, prompt_rows


def _build_manifest() -> DatasetManifest:
    cohorts = [
        CohortSpec(
            "target",
            "target",
            display_name="Female users (y=1)",
            description=(
                "PRISM users who self-reported gender Female (gold survey). "
                "x = first-turn user prompt from unguided conversations."
            ),
        ),
        CohortSpec(
            "baseline",
            "baseline",
            display_name="Male users (y=0)",
            description=(
                "PRISM users who self-reported gender Male (gold survey). "
                "x = first-turn user prompt from unguided conversations."
            ),
        ),
        CohortSpec(
            "null_split_a",
            "target",
            display_name="Female split A (null control)",
            description=(
                "Seeded random half of the female (target) authors — negative "
                "control partner for null_split_b (same underlying group)."
            ),
        ),
        CohortSpec(
            "null_split_b",
            "target",
            display_name="Female split B (null control)",
            description=(
                "Seeded random other half of the female (target) authors — "
                "negative control partner for null_split_a."
            ),
        ),
    ]
    for n in POWER_SIZES:
        cohorts.append(
            CohortSpec(
                f"target_n{n}",
                "target",
                display_name=f"Female users, n={n} (power)",
                description=f"Seeded {n}-author subsample of the target cohort.",
            )
        )
        cohorts.append(
            CohortSpec(
                f"baseline_n{n}",
                "baseline",
                display_name=f"Male users, n={n} (power)",
                description=f"Seeded {n}-author subsample of the baseline cohort.",
            )
        )

    comparisons = [
        ComparisonSpec(
            "target_vs_baseline",
            "target",
            "baseline",
            expectation="",
            explorations=True,
            expected_accuracy=0.0,
            notes=(
                "Modality match (real first-turn LLM prompts). Female (y=1) vs "
                "male (y=0), unguided conversations only. Exploratory: no "
                "published per-prompt gender-separability benchmark for PRISM."
            ),
        ),
        ComparisonSpec(
            "null_control",
            "null_split_a",
            "null_split_b",
            expectation="null",
            explorations=False,
            expected_accuracy=0.0,
            notes=(
                "Negative control: seeded random author-split within the female "
                "(target) group. Expect C2ST~0.5, 0 BH survivors, null MMD."
            ),
        ),
    ]
    for n in POWER_SIZES:
        comparisons.append(
            ComparisonSpec(
                f"target_vs_baseline_n{n}",
                f"target_n{n}",
                f"baseline_n{n}",
                expectation="",
                explorations=False,
                expected_accuracy=0.0,
                notes=f"power analysis n={n} authors per side",
            )
        )

    return DatasetManifest(
        name=DATASET_NAME,
        description=(
            "PRISM alignment survey (HannahRoseKirk/prism-alignment): real "
            "first-turn user prompts from unguided conversations, labelled by "
            "gold-survey gender (female target y=1 vs male baseline y=0). "
            "Human text CC-BY-4.0. Subsampled with author structure preserved; "
            "z/d/c partially filled from the survey (see README)."
        ),
        cohorts=cohorts,
        comparisons=comparisons,
    )


def convert(raw_dir: Path, out_dir: Path, seed: int) -> None:
    survey_path, utterances_path = _download_raw(raw_dir)
    survey = _load_survey(survey_path)
    prompts_by_user = _first_turn_prompts(utterances_path)

    rng = random.Random(seed)
    female = _eligible_authors(survey, prompts_by_user, "Female")
    male = _eligible_authors(survey, prompts_by_user, "Male")
    rng.shuffle(female)
    rng.shuffle(male)
    target_users = female[:MAIN_AUTHORS_PER_SIDE]
    baseline_users = male[:MAIN_AUTHORS_PER_SIDE]

    counter = [0]
    author_rows: list[dict] = []
    prompt_rows: list[dict] = []

    for users, cohort, lgbtq in (
        (target_users, "target", 1),
        (baseline_users, "baseline", 0),
    ):
        a, p = _emit_cohort(users, cohort, lgbtq, survey, prompts_by_user, counter)
        author_rows += a
        prompt_rows += p

    # Negative control: seeded random split of the target (female) authors.
    split = list(target_users)
    rng.shuffle(split)
    half = len(split) // 2
    for users, cohort in (
        (split[:half], "null_split_a"),
        (split[half:], "null_split_b"),
    ):
        a, p = _emit_cohort(users, cohort, 1, survey, prompts_by_user, counter)
        author_rows += a
        prompt_rows += p

    # Power-analysis subsamples of the main cohorts.
    for n in POWER_SIZES:
        for users, base_cohort, lgbtq in (
            (target_users, "target", 1),
            (baseline_users, "baseline", 0),
        ):
            sub = list(users)
            rng.shuffle(sub)
            a, p = _emit_cohort(
                sub[:n], f"{base_cohort}_n{n}", lgbtq, survey, prompts_by_user, counter
            )
            author_rows += a
            prompt_rows += p

    prompts_df = pd.DataFrame(prompt_rows)
    authors_df = pd.DataFrame(author_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(_build_manifest().to_dict(), out_dir / "dataset.json")
    prompts_df.to_parquet(out_dir / "prompts.parquet", index=False)
    authors_df.to_parquet(out_dir / "authors.parquet", index=False)
    _write_readme(out_dir, prompts_df, authors_df, seed)

    # Fail loudly if the written dataset does not load/validate.
    PromptDataset.load(out_dir)
    print(f"Wrote {len(prompts_df)} prompts / {len(authors_df)} authors to {out_dir}")


def _write_readme(
    out_dir: Path, prompts_df: pd.DataFrame, authors_df: pd.DataFrame, seed: int
) -> None:
    counts = prompts_df.groupby("cohort").agg(
        prompts=("prompt_id", "size"),
        authors=("author_id", "nunique"),
    )
    lines = counts.to_string()
    readme = f"""# PRISM (gender) — real LLM prompts by gold-survey gender

Converted from **HannahRoseKirk/prism-alignment** (Kirk et al., 2024) by
`scripts/convert_prism.py` (seed {seed}). This is real user data: human-written
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

## Sampling (seed {seed})

1. Keep first turns of **unguided** conversations only; whitespace-normalise.
2. Drop prompts with **< {MIN_PROMPT_WORDS} words** (trivial "hi"/"hello") and
   **de-duplicate** exact (lowercased) prompt texts globally.
3. **Truncate + collapse interior whitespace** (newlines/tabs/repeated spaces
   become single spaces) — prompts longer than **{MAX_PROMPT_WORDS} words** to their first
   {MAX_PROMPT_WORDS} words (only 3 unguided first-turn prompts exceed this).
4. Among users with >=1 usable prompt, seeded-shuffle and take
   **{MAIN_AUTHORS_PER_SIDE} female (target) + {MAIN_AUTHORS_PER_SIDE} male
   (baseline)** authors, **<= {MAX_PROMPTS_PER_AUTHOR} prompts each** (author
   structure preserved: every prompt's author is in `authors.parquet`).
5. **null_split_a / null_split_b**: seeded random split of the {MAIN_AUTHORS_PER_SIDE}
   target (female) authors into two disjoint halves — the required **negative
   control** (`null_control` comparison, expectation "null").
6. **target_n12/24 & baseline_n12/24**: seeded author subsamples of the main
   cohorts for the **power-analysis** comparisons (exploratory, expectation "").

Author ids are cohort-prefixed (`target__user123`) so the power/null cohorts are
independent row sets (a user may appear in several cohorts under distinct ids).

## Per-cohort counts

```
{lines}
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
  binary female/male contrast. (d) Long-prompt truncation (>{MAX_PROMPT_WORDS}
  words) affects only 3 prompts.
"""
    (out_dir / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PRISM to our format.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / DATASET_NAME,
    )
    args = parser.parse_args()
    convert(args.raw_dir, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
