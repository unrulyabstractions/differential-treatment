"""Convert ThoughtTrace (HF SCAI-JHU/ThoughtTrace, CC-BY-4.0) to the D=(x,y,z,d,c) tables.

x = real user turns from human-LLM conversations (truncated to <=120 words),
y = self-reported gender from the gold exit survey (female = target vs male =
baseline), z/d/c filled from the survey (gender, age bracket, education,
AI-usage frequency -> llm_freq, purposes -> domain). Includes the mandatory
negative-control comparison: a seeded random author split within the female
pool (null_split_a vs null_split_b, expectation "null").

Idempotent: downloads the raw jsonl to --raw-dir (never into the repo), then
rewrites data/thoughttrace/{dataset.json,prompts.parquet,authors.parquet,README.md}.

Usage: uv run python scripts/convert_thoughttrace.py [--seed 0]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from src.common.dataset_tables import (
    CohortSpec,
    ComparisonSpec,
    DatasetManifest,
    PromptDataset,
)
from src.common.file_io import ensure_dir, save_json

DATASET_NAME = "thoughttrace"
HF_REPO = "SCAI-JHU/ThoughtTrace"
HF_FILE = "ThoughtTrace.jsonl"
DEFAULT_RAW_DIR = str(Path(__file__).resolve().parent.parent / "data" / "raw_datasets" / "thoughttrace")
MIN_CHARS, MAX_WORDS = 30, 120
GENDER_MAP = {"Female": ["woman"], "Male": ["man"]}
AGE_BRACKETS = [
    (24, "18-24"),
    (34, "25-34"),
    (44, "35-44"),
    (54, "45-54"),
    (64, "55-64"),
]

PROMPT_DEFAULTS = {
    "markedness": 0,
    "codedness": 0.0,
    "topic_id": 0,
    "adoption": 0,
    "general_freq": 0,
    "professional_freq": 0,
    "aversion": 0,
    "satisfaction": 0,
}


def _age_bracket(raw: str | None) -> str:
    try:
        age = int(str(raw).strip())
    except (TypeError, ValueError):
        return ""
    if age < 18:
        return ""
    return next((label for cap, label in AGE_BRACKETS if age <= cap), "65+")


def _truncate(text: str) -> str:
    words = text.split()
    return " ".join(words[:MAX_WORDS])


def load_raw(raw_dir: Path) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Download (idempotent) and parse: per-user survey + eligible user turns."""
    path = hf_hub_download(
        HF_REPO, HF_FILE, repo_type="dataset", local_dir=str(ensure_dir(raw_dir))
    )
    survey_by_user: dict[str, dict] = {}
    turns_by_user: dict[str, list[str]] = defaultdict(list)
    seen_texts: set[str] = set()
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            uid = row["id"].split("_")[0]
            answers = row.get("survey_answers") or []
            if answers:
                survey_by_user.setdefault(uid, answers[0])
            for message in row["messages"]:
                text = (message.get("content") or "").strip()
                if message.get("type") != "user" or len(text) < MIN_CHARS:
                    continue
                if text in seen_texts:  # drop exact duplicates corpus-wide
                    continue
                seen_texts.add(text)
                turns_by_user[uid].append(_truncate(text))
    return survey_by_user, turns_by_user


def build_rows(seed: int, per_side: int, per_author: int, raw_dir: Path):
    survey_by_user, turns_by_user = load_raw(raw_dir)
    rng = np.random.default_rng(seed)
    pools = {
        gender: sorted(
            uid
            for uid, s in survey_by_user.items()
            if s.get("gender") == gender and turns_by_user.get(uid)
        )
        for gender in ("Female", "Male")
    }
    target = list(rng.choice(pools["Female"], size=per_side, replace=False))
    baseline = list(rng.choice(pools["Male"], size=per_side, replace=False))
    remaining_female = [u for u in pools["Female"] if u not in set(target)]
    null_pool = list(
        rng.choice(
            remaining_female, size=min(120, len(remaining_female)), replace=False
        )
    )
    half = len(null_pool) // 2
    cohort_authors = {
        "target": target,
        "baseline": baseline,
        "null_split_a": null_pool[:half],
        "null_split_b": null_pool[half : 2 * half],
    }

    prompt_rows, author_rows = [], []
    counter = 0
    for cohort, authors in cohort_authors.items():
        for uid in authors:
            survey = survey_by_user[uid]
            is_female = survey.get("gender") == "Female"
            author_rows.append(
                {
                    "author_id": uid,
                    "cohort": cohort,
                    "transgender": "",
                    "gender": GENDER_MAP.get(survey.get("gender"), []),
                    "orientation": [],
                    "pronouns": [],
                    "race": [],
                    "age": _age_bracket(survey.get("age")),
                    "disability": "",
                    "education": (survey.get("education") or "").lower(),
                    "income": "",
                }
            )
            turns = turns_by_user[uid]
            take = rng.choice(
                len(turns), size=min(per_author, len(turns)), replace=False
            )
            try:
                frequency = int(str(survey.get("frequency")).strip())
            except (TypeError, ValueError):
                frequency = 0
            for idx in sorted(take):
                prompt_rows.append(
                    dict(
                        prompt_id=f"{DATASET_NAME}_{counter:05d}",
                        author_id=uid,
                        cohort=cohort,
                        text=turns[idx],
                        lgbtq=int(is_female),
                        domain=survey.get("purposes") or "",
                        provenance=DATASET_NAME,
                        llm_freq=frequency,
                        **PROMPT_DEFAULTS,
                    )
                )
                counter += 1
    return prompt_rows, author_rows, half


def build_manifest() -> DatasetManifest:
    exploratory_note = (
        "Modality-match validation (real LLM prompts): y = self-reported gender "
        "(gold exit survey), female vs male. No published separability benchmark "
        "- exploratory. Observer-aware study: participants knew turns and "
        "thoughts were being recorded, which may shift register."
    )
    null_note = (
        "Negative control: seeded random author split within the female pool "
        "(disjoint from target cohort authors). Expect C2ST ~= 0.5, zero BH "
        "survivors, null MMD."
    )
    return DatasetManifest(
        name=DATASET_NAME,
        description=(
            "ThoughtTrace (SCAI-JHU/ThoughtTrace, CC-BY-4.0): real user turns from "
            "human-LLM conversations, cohorts by survey-reported gender "
            "(female target vs male baseline) plus a null author split within the "
            "female pool. Texts truncated to <=120 words; see data/thoughttrace/README.md."
        ),
        cohorts=[
            CohortSpec(
                "target",
                "target",
                "Female users (y=1)",
                "ThoughtTrace users self-reporting gender=Female in the exit survey.",
            ),
            CohortSpec(
                "baseline",
                "baseline",
                "Male users (y=0)",
                "ThoughtTrace users self-reporting gender=Male in the exit survey.",
            ),
            CohortSpec(
                "null_split_a",
                "target",
                "Null split A (female)",
                "Random half of held-out female users (seeded); null-control cohort.",
            ),
            CohortSpec(
                "null_split_b",
                "baseline",
                "Null split B (female)",
                "Other random half of held-out female users (seeded); null-control cohort.",
            ),
        ],
        comparisons=[
            ComparisonSpec(
                "target_vs_baseline",
                "target",
                "baseline",
                expectation="",
                explorations=True,
                expected_accuracy=0.0,
                notes=exploratory_note,
            ),
            ComparisonSpec(
                "null_split",
                "null_split_a",
                "null_split_b",
                expectation="null",
                explorations=False,
                expected_accuracy=0.5,
                notes=null_note,
            ),
        ],
    )


README_TEMPLATE = """\
# ThoughtTrace (converted)

Real user turns from human-LLM conversations, converted to the paper's
D=(x,y,z,d,c) table format (`src/common/dataset_tables.py`). Regenerate with
`uv run python scripts/convert_thoughttrace.py --seed {seed}` (idempotent).

## Provenance and license

- Source: Hugging Face `SCAI-JHU/ThoughtTrace` (`ThoughtTrace.jsonl`, ~30 MB),
  license **CC-BY-4.0**. Paper: Jin et al. 2026, "ThoughtTrace: Understanding
  User Thoughts in Real-World LLM Interactions" (arXiv:2605.20087). Cite when used.
- Raw corpus: 2,155 conversations from 1,058 users across 20 LLMs; user turns
  carry self-reported *reasons*, and each user has a gold exit survey
  (age, gender, education, occupation, AI-usage frequency, purposes).
- Author id = the `user{{N}}` prefix of the conversation `id`.

## Cohorts and sampling (seed={seed})

- y = survey-reported gender: `target` = Female (lgbtq flag 1 = generic y),
  `baseline` = Male (0). Non-binary / prefer-not-to-say users (9) are excluded.
- x = **user turns** (all turns, not just conversation openers), stripped,
  interior whitespace (newlines/tabs/repeated spaces) collapsed to single
  spaces, minimum {min_chars} chars, corpus-wide exact-duplicate turns dropped,
  and **truncated to the first {max_words} words** (turns longer than that are cut).
- Seeded subsample: {per_side} authors per side, up to {per_author} turns per
  author (author structure preserved; every prompt's author is in authors.parquet).
- `null_split_a` / `null_split_b`: negative control — {null_half}+{null_half}
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
"""


def write_readme(out_dir: Path, args: argparse.Namespace, null_half: int) -> None:
    (out_dir / "README.md").write_text(
        README_TEMPLATE.format(
            seed=args.seed,
            min_chars=MIN_CHARS,
            max_words=MAX_WORDS,
            per_side=args.authors_per_side,
            per_author=args.prompts_per_author,
            null_half=null_half,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--authors-per-side", type=int, default=100)
    parser.add_argument("--prompts-per-author", type=int, default=4)
    parser.add_argument("--raw-dir", type=Path, default=Path(DEFAULT_RAW_DIR))
    parser.add_argument("--out-dir", type=Path, default=Path("data") / DATASET_NAME)
    args = parser.parse_args()

    prompt_rows, author_rows, null_half = build_rows(
        args.seed, args.authors_per_side, args.prompts_per_author, args.raw_dir
    )
    out_dir = ensure_dir(args.out_dir)
    pd.DataFrame(prompt_rows).to_parquet(out_dir / "prompts.parquet", index=False)
    pd.DataFrame(author_rows).to_parquet(out_dir / "authors.parquet", index=False)
    save_json(build_manifest().to_dict(), out_dir / "dataset.json")
    write_readme(out_dir, args, null_half)

    dataset = PromptDataset.load(out_dir)  # runs validate()
    print(f"Loaded + validated {out_dir}")
    counts = dataset.prompts.groupby("cohort").agg(
        prompts=("prompt_id", "size"), authors=("author_id", "nunique")
    )
    print(counts)
    print(
        dataset.prompts.sample(3, random_state=args.seed)[
            ["prompt_id", "cohort", "author_id", "lgbtq", "llm_freq", "text"]
        ].to_string(max_colwidth=80)
    )


if __name__ == "__main__":
    main()
