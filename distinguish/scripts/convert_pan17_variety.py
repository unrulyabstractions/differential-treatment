"""Convert PAN-2017 author profiling (EN, Twitter) to data/pan17_variety/.

Source: Zenodo record 3745980 (open access, verified 2026-07-06), training
zip with per-author XML tweet feeds (100 tweets/author) and truth.txt lines
`authorhash:::gender:::variety`. License: PAN shared-task terms (research
use); cite Rangel, Rosso, Potthast & Stein, CLEF 2017.

y = English variety, United States (target) vs Great Britain (baseline);
x = one individual TWEET. Published anchor 71.5% is per-FEED (100 tweets),
so per-tweet numbers will run lower. Gender labels exist and are stored in
authors.parquet but this dataset keeps ONE primary y (variety).

Idempotent: raw zip + extraction cached in the scratch dir; converted output
is fully regenerated on each run. Usage:
    uv run python scripts/convert_pan17_variety.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

from src.common.dataset_tables import (
    CohortSpec,
    ComparisonSpec,
    DatasetManifest,
    PromptDataset,
)

DATASET_NAME = "pan17_variety"
ZIP_NAME = "pan17-author-profiling-training-dataset-2017-03-10.zip"
ZENODO_URL = f"https://zenodo.org/records/3745980/files/{ZIP_NAME}?download=1"
RAW_DIR = (
    Path(
        "/private/tmp/claude-501/-Users-unrulyabstractions-work-prompt-distinguishability"
        "/140b8c69-9c33-43d8-8683-a04e90d30bcf/scratchpad/raw_datasets"
    )
    / DATASET_NAME
)
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / DATASET_NAME

AUTHORS_PER_SIDE = 80
TWEETS_PER_AUTHOR = 4
MAX_WORDS = 120
MIN_WORDS, MIN_CHARS = 4, 20
GENDER_MAP = {"female": "woman", "male": "man"}
VARIETY_COHORT = {"united states": "target", "great britain": "baseline"}

PROMPT_DEFAULTS = {
    "markedness": 0,
    "codedness": 0.0,
    "topic_id": 0,
    "domain": "",
    "provenance": "pan17",
    "adoption": 0,
    "general_freq": 0,
    "llm_freq": 0,
    "professional_freq": 0,
    "aversion": 0,
    "satisfaction": 0,
}
AUTHOR_DEFAULTS = {
    "transgender": "",
    "orientation": [],
    "pronouns": [],
    "race": [],
    "age": "",
    "disability": "",
    "education": "",
    "income": "",
}


def download_and_extract() -> Path:
    """Fetch the train zip into the scratch dir (never the repo); idempotent."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / ZIP_NAME
    if not zip_path.exists() or zip_path.stat().st_size < 50_000_000:
        print(f"downloading {ZENODO_URL} -> {zip_path}")
        urllib.request.urlretrieve(ZENODO_URL, zip_path)
    extract_dir = RAW_DIR / "extracted"
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    en_dirs = sorted(p.parent for p in extract_dir.rglob("en/truth.txt"))
    if not en_dirs:
        raise FileNotFoundError(f"no en/truth.txt under {extract_dir}")
    return en_dirs[0]


def clean_tweet(text: str) -> str:
    """Normalize whitespace; drop retweets and low-content tweets ('' = drop)."""
    text = re.sub(r"\s+", " ", text).strip()
    if text.startswith("RT @") or text.startswith("RT:"):
        return ""
    stripped = re.sub(r"https?://\S+|@\w+", " ", text)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if len(stripped) < MIN_CHARS or len(stripped.split()) < MIN_WORDS:
        return ""
    words = text.split()
    return " ".join(words[:MAX_WORDS])  # tweets never hit this; guard only


def load_authors(en_dir: Path) -> dict[str, dict]:
    """truth.txt + XML feeds -> {author_id: {gender, variety, tweets}}."""
    authors: dict[str, dict] = {}
    for line in (en_dir / "truth.txt").read_text().splitlines():
        author_id, gender, variety = line.strip().split(":::")
        if variety not in VARIETY_COHORT:
            continue
        xml_path = en_dir / f"{author_id}.xml"
        if not xml_path.exists():
            continue
        docs = ET.parse(xml_path).getroot().iter("document")
        tweets, seen = [], set()
        for doc in docs:
            cleaned = clean_tweet(doc.text or "")
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                tweets.append(cleaned)
        authors[author_id] = {"gender": gender, "variety": variety, "tweets": tweets}
    return authors


def sample_rows(authors: dict[str, dict], seed: int) -> tuple[list, list]:
    """Seeded author-level subsample; returns (prompt_rows, author_rows)."""
    rng = random.Random(seed)
    prompt_rows, author_rows, counter = [], [], 0
    sampled: dict[str, list[str]] = {}  # cohort -> sampled author ids
    for cohort in ("target", "baseline"):
        pool = sorted(
            a
            for a, info in authors.items()
            if VARIETY_COHORT[info["variety"]] == cohort
            and len(info["tweets"]) >= TWEETS_PER_AUTHOR
        )
        sampled[cohort] = rng.sample(pool, AUTHORS_PER_SIDE)
        for author_id in sampled[cohort]:
            info = authors[author_id]
            author_rows.append(
                dict(
                    author_id=author_id,
                    cohort=cohort,
                    gender=[GENDER_MAP[info["gender"]]],
                    **AUTHOR_DEFAULTS,
                )
            )
            for tweet in rng.sample(info["tweets"], TWEETS_PER_AUTHOR):
                prompt_rows.append(
                    dict(
                        prompt_id=f"{DATASET_NAME}_{counter:05d}",
                        author_id=author_id,
                        cohort=cohort,
                        text=tweet,
                        lgbtq=int(cohort == "target"),
                        **PROMPT_DEFAULTS,
                    )
                )
                counter += 1
    # Negative control: seeded random split of the sampled TARGET authors.
    shuffled = list(sampled["target"])
    rng.shuffle(shuffled)
    half = dict.fromkeys(shuffled[: len(shuffled) // 2], "null_split_a")
    for row in [r for r in author_rows if r["cohort"] == "target"]:
        null_cohort = half.get(row["author_id"], "null_split_b")
        author_rows.append({**row, "cohort": null_cohort})
    for row in [r for r in prompt_rows if r["cohort"] == "target"]:
        null_cohort = half.get(row["author_id"], "null_split_b")
        prompt_rows.append(
            {
                **row,
                "cohort": null_cohort,
                "prompt_id": row["prompt_id"] + "_ns",
            }
        )
    return prompt_rows, author_rows


def build_manifest(seed: int) -> DatasetManifest:
    per_feed = (
        "Published PAN17 EN numbers are per-FEED (100 tweets/author): 0.9004 "
        "6-way variety accuracy (Tellez et al., CLEF 2017 overview Table 4); "
        "the 0.715 anchor for this US-vs-GB pair is per-feed too, so per-TWEET "
        "separability will run lower — compare per-author aggregates."
    )
    return DatasetManifest(
        name=DATASET_NAME,
        description=(
            "PAN-2017 author profiling (EN Twitter, Zenodo 3745980): English "
            "variety contrast, US vs GB, one tweet per prompt, "
            f"{AUTHORS_PER_SIDE} authors x {TWEETS_PER_AUTHOR} tweets per side "
            f"(seed={seed}). Subtle-signal / coded-language analog; author "
            "gender is stored but variety is the single primary y."
        ),
        cohorts=[
            CohortSpec(
                "target",
                "target",
                "US tweeters (variety=united states, y=1)",
                "Authors labeled 'united states' in truth.txt.",
            ),
            CohortSpec(
                "baseline",
                "baseline",
                "GB tweeters (variety=great britain, y=0)",
                "Authors labeled 'great britain' in truth.txt.",
            ),
            CohortSpec(
                "null_split_a",
                "target",
                "US null split A (negative control)",
                "Random half of the sampled US authors (seeded).",
            ),
            CohortSpec(
                "null_split_b",
                "baseline",
                "US null split B (negative control)",
                "Complementary half of the sampled US authors.",
            ),
        ],
        comparisons=[
            ComparisonSpec(
                name="target_vs_baseline",
                target_cohort="target",
                baseline_cohort="baseline",
                expectation="distinguishable",
                explorations=True,
                expected_accuracy=0.715,
                notes="Validation role: subtle-signal / coded-language analog. "
                + per_feed,
            ),
            ComparisonSpec(
                name="null_split",
                target_cohort="null_split_a",
                baseline_cohort="null_split_b",
                expectation="null",
                explorations=False,
                expected_accuracy=0.5,
                notes="Negative control: seeded random author split WITHIN the "
                "US cohort. Expect C2ST ~ 0.5, 0 BH survivors, null MMD.",
            ),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    en_dir = download_and_extract()
    authors = load_authors(en_dir)
    prompt_rows, author_rows = sample_rows(authors, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prompt_rows).to_parquet(OUT_DIR / "prompts.parquet", index=False)
    pd.DataFrame(author_rows).to_parquet(OUT_DIR / "authors.parquet", index=False)
    manifest = build_manifest(args.seed)
    (OUT_DIR / "dataset.json").write_text(
        json.dumps(manifest.to_dict(), indent=4) + "\n"
    )
    (OUT_DIR / "README.md").write_text(readme_text(args.seed, authors))
    dataset = PromptDataset.load(OUT_DIR)  # runs validate()
    print("PromptDataset.load OK; per-cohort prompt counts:")
    print(dataset.prompts.groupby("cohort")["prompt_id"].count().to_string())
    print("\nSample rows:")
    sample = dataset.prompts.sample(3, random_state=args.seed)
    print(sample[["prompt_id", "author_id", "cohort", "lgbtq", "text"]].to_string())


def readme_text(seed: int, authors: dict[str, dict]) -> str:
    n_us = sum(1 for a in authors.values() if a["variety"] == "united states")
    n_gb = sum(1 for a in authors.values() if a["variety"] == "great britain")
    return f"""# pan17_variety

PAN-2017 author profiling, English Twitter feeds (training split), converted
by `scripts/convert_pan17_variety.py --seed {seed}`.

## Provenance & license
- Zenodo record 3745980, `{ZIP_NAME}` — open access, no registration
  (verified 2026-07-06). Per-author XML feeds (100 tweets each) +
  `truth.txt` (`authorhash:::gender:::variety`).
- PAN shared-task data, research use; cite Rangel Pardo, Rosso, Potthast &
  Stein, "Overview of the 5th Author Profiling Task at PAN 2017", CLEF 2017.
- EN train pool used here: {n_us} US + {n_gb} GB authors (of 3,600 EN total).

## Task
- **y** = English variety: `target` = united states (lgbtq flag 1),
  `baseline` = great britain (flag 0). The generic `lgbtq` column is the y
  flag, not an identity claim.
- **x** = one individual TWEET. The published anchor (expected_accuracy
  0.715) is per-FEED over 100 tweets — per-tweet separability WILL be lower;
  compare per-author aggregates. Best published EN 6-way variety accuracy:
  0.9004; best EN gender: 0.8233 (CLEF 2017 overview, Tables 3-4).
- Gender labels are available and stored in `authors.parquet`
  (`gender=["woman"|"man"]`) for slices, but this dataset keeps ONE primary
  y (variety); a gender contrast would be a separate dataset.

## Sampling (seeded, author structure preserved)
- Tweet filter: whitespace-normalized; retweets (`RT @...`) dropped; after
  removing URLs/@mentions must keep >= {MIN_WORDS} words and >= {MIN_CHARS}
  chars; per-author exact duplicates dropped. Interior whitespace
  (newlines/tabs/repeated spaces) is collapsed to single spaces. Tweets are truncated to
  {MAX_WORDS} words as a guard (no real tweet reaches it — no text was
  actually truncated or segmented).
- {AUTHORS_PER_SIDE} authors/side sampled (seed={seed}) from authors with
  >= {TWEETS_PER_AUTHOR} qualifying tweets; {TWEETS_PER_AUTHOR} tweets each
  -> {AUTHORS_PER_SIDE * TWEETS_PER_AUTHOR} prompts per cohort.
- Negative control: the sampled US authors are randomly split (same seed)
  into `null_split_a` / `null_split_b`; their prompts are duplicated under
  `*_ns` prompt ids. Comparison `null_split` has expectation "null"
  (expect C2ST ~ 0.5, zero BH survivors, null MMD).

## Caveats
- z/d/c are unannotated except author gender: markedness/codedness/topic/
  usage fields are 0/"" ("unrecorded") -> implicit & usage modules skip.
- Variety labels come from PAN's collection procedure (user-declared
  location), not self-reported identity; treat as weak ground truth.
- Tweets are 2017-era Twitter register: URLs, @mentions and hashtags remain
  in the text (only used for the low-content filter).
"""


if __name__ == "__main__":
    main()
