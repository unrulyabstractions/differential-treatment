"""Convert WildChat-1M (allenai/WildChat-1M, ODC-BY) into data/wildchat/.

Usage:
    uv run python scripts/convert_wildchat.py [--seed 0] [--n-authors 80] \\
        [--out-dir data/wildchat] [--raw-dir <scratch>/raw_datasets/wildchat]

Downloads ONE shard (data/train-00000-of-00014.parquet, ~240 MB, 59,857
conversations) of the public 837,989-conversation release into --raw-dir
(never into the repo), keeps English non-redacted non-toxic FIRST user turns,
and builds y = country contrast: target United Kingdom vs baseline United
States (falls back to India as target if UK is sparse, and says so). Author
proxy = hashed IP (imperfect; <=4 prompts/author). Texts are truncated to the
first 120 words. Also emits a negative control: a seeded random split of
extra baseline-country authors (cohorts null_split_a / null_split_b). A
large-n variant for scale/asymptotics is one `--n-authors 500` rerun away.
Idempotent: same flags + seed regenerate identical tables.
"""

from __future__ import annotations

import argparse
import re
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.common.dataset_tables import (  # noqa: E402
    CohortSpec,
    ComparisonSpec,
    DatasetManifest,
    PromptDataset,
)
from src.common.file_io import save_json  # noqa: E402

HF_REPO = "allenai/WildChat-1M"
SHARD = "data/train-00000-of-00014.parquet"
DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_datasets" / "wildchat"
PROMPT_DEFAULTS = {
    "markedness": 0,
    "codedness": 0.0,
    "topic_id": 0,
    "domain": "",
    "provenance": "real",
    "adoption": 0,
    "general_freq": 0,
    "llm_freq": 0,
    "professional_freq": 0,
    "aversion": 0,
    "satisfaction": 0,
}  # unrecorded annotations; provenance="real": logged real user prompts
AUTHOR_DEFAULTS = dict(  # noqa: C408
    transgender="",
    gender=[],
    orientation=[],
    pronouns=[],
    race=[],
    age="",
    disability="",
    education="",
    income="",
)  # WildChat records no author demographics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-authors", type=int, default=80, help="authors per side")
    parser.add_argument("--max-per-author", type=int, default=4)
    parser.add_argument("--min-words", type=int, default=8)
    parser.add_argument("--max-words", type=int, default=120)
    parser.add_argument("--target-country", default="United Kingdom")
    parser.add_argument("--fallback-country", default="India")
    parser.add_argument("--baseline-country", default="United States")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("data/wildchat"))
    return parser.parse_args()


def load_first_turns(raw_dir: Path) -> pd.DataFrame:
    """Download the shard (cached) and extract first-user-turn fields."""
    shard = hf_hub_download(HF_REPO, SHARD, repo_type="dataset", local_dir=raw_dir)
    table = pq.read_table(
        shard, columns=["conversation_hash", "language", "conversation"]
    )
    first = pc.list_element(table["conversation"], 0)
    return pd.DataFrame(
        {
            "conversation_hash": table["conversation_hash"].to_pandas(),
            "language": table["language"].to_pandas(),
            "text": pc.struct_field(first, "content").to_pandas(),
            "country": pc.struct_field(first, "country").to_pandas(),
            "hashed_ip": pc.struct_field(first, "hashed_ip").to_pandas(),
            "redacted": pc.struct_field(first, "redacted").to_pandas(),
            "toxic": pc.struct_field(first, "toxic").to_pandas(),
        }
    )


def truncate_words(text: str, max_words: int) -> str:
    """Cut after the max_words-th word, preserving internal whitespace."""
    matches = list(islice(re.finditer(r"\S+", text), max_words))
    if not matches:
        return ""
    return text[: matches[-1].end()].strip()


def build_pool(raw: pd.DataFrame, min_words: int, max_words: int) -> pd.DataFrame:
    """English, non-redacted, non-toxic, >=min_words; truncated + deduplicated."""
    pool = raw[
        (raw["language"] == "English")
        & ~raw["redacted"].fillna(True)
        & ~raw["toxic"].fillna(True)
        & raw["hashed_ip"].notna()
        & raw["country"].notna()
    ].copy()
    pool["n_words"] = pool["text"].str.split().str.len()
    pool = pool[pool["n_words"] >= min_words]
    pool["truncated"] = pool["n_words"] > max_words
    pool["text"] = pool["text"].map(lambda t: truncate_words(t, max_words))
    pool = pool.sort_values(["hashed_ip", "conversation_hash"], kind="stable")
    return pool.drop_duplicates("text", keep="first").reset_index(drop=True)


def sample_cohort(
    pool: pd.DataFrame,
    country: str,
    cohort: str,
    y_flag: int,
    authors: np.ndarray,
    rng: np.random.Generator,
    max_per_author: int,
) -> pd.DataFrame:
    """Prompt rows for one cohort: <=max_per_author seeded picks per author."""
    rows = pool[(pool["country"] == country) & pool["hashed_ip"].isin(authors)]
    picks = [
        group.iloc[rng.permutation(len(group))[:max_per_author]]
        for _, group in rows.groupby("hashed_ip", sort=True)
    ]
    out = pd.concat(picks)[["hashed_ip", "text", "truncated"]]
    return out.assign(cohort=cohort, lgbtq=y_flag).rename(
        columns={"hashed_ip": "author_id"}
    )


def make_manifest(target_country: str, baseline_country: str) -> DatasetManifest:
    contrast = f"{target_country} vs {baseline_country}"
    return DatasetManifest(
        name="wildchat",
        description=(
            "First user turns of English WildChat-1M conversations "
            f"(allenai/WildChat-1M shard 0/14, ODC-BY). y = country proxy: {contrast}; "
            "author = hashed IP (imperfect). Modality-match (real chatbot prompts) + "
            "scale/asymptotics dataset per the Iter4 validation design; no published "
            "separability benchmark. Includes a seeded same-group null split."
        ),
        cohorts=[
            CohortSpec(
                "target",
                "target",
                f"{target_country} prompts (y=1)",
                f"English first-turn prompts, first-turn country {target_country}.",
            ),
            CohortSpec(
                "baseline",
                "baseline",
                f"{baseline_country} prompts (y=0)",
                f"English first-turn prompts, first-turn country {baseline_country}.",
            ),
            CohortSpec(
                "null_split_a",
                "baseline",
                f"{baseline_country} null split A (y=0)",
                f"Random half of extra {baseline_country} authors (disjoint from baseline); negative control.",
            ),
            CohortSpec(
                "null_split_b",
                "baseline",
                f"{baseline_country} null split B (y=0)",
                f"Other random half of the same extra {baseline_country} authors; negative control.",
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
                notes=(
                    f"Exploratory country contrast ({contrast}); no published benchmark "
                    "(expected_accuracy 0.0). y is a PROXY label (first-turn IP geolocation, "
                    "not self-reported identity) and author = hashed IP (IPs shift/shared). "
                    "Role: modality match + scale/asymptotics — MMD power and C2ST "
                    "convergence at large n via a --n-authors rerun."
                ),
            ),
            ComparisonSpec(
                "null_split",
                "null_split_a",
                "null_split_b",
                expectation="null",
                explorations=False,
                expected_accuracy=0.0,
                notes=(
                    f"Negative control: seeded random author split WITHIN one group "
                    f"(extra {baseline_country} authors, disjoint from baseline). "
                    "Expect C2ST ~= 0.5, zero BH survivors, null MMD."
                ),
            ),
        ],
    )


def convert(args: argparse.Namespace) -> pd.DataFrame:
    pool = build_pool(load_first_turns(args.raw_dir), args.min_words, args.max_words)
    per_country = pool.groupby("country")["hashed_ip"].nunique()
    target_country = args.target_country
    if per_country.get(target_country, 0) < args.n_authors:
        print(
            f"[report] target {target_country} SPARSE "
            f"({per_country.get(target_country, 0)} authors < {args.n_authors}); "
            f"falling back to {args.fallback_country}"
        )
        target_country = args.fallback_country
    rng = np.random.default_rng(args.seed)

    def draw(country: str, n: int, exclude: frozenset[str]) -> np.ndarray:
        eligible = np.array(
            sorted(set(pool[pool["country"] == country]["hashed_ip"]) - exclude)
        )
        if len(eligible) < n:
            print(
                f"[report] only {len(eligible)} {country} authors available for n={n}"
            )
        return rng.choice(eligible, size=min(n, len(eligible)), replace=False)

    target_authors = draw(target_country, args.n_authors, frozenset())
    baseline_authors = draw(args.baseline_country, args.n_authors, frozenset())
    null_pool = draw(
        args.baseline_country, 2 * args.n_authors, frozenset(baseline_authors)
    )
    half = len(null_pool) // 2
    null_a, null_b = null_pool[:half], null_pool[half : 2 * half]

    prompts = pd.concat(
        [
            sample_cohort(
                pool,
                target_country,
                "target",
                1,
                target_authors,
                rng,
                args.max_per_author,
            ),
            sample_cohort(
                pool,
                args.baseline_country,
                "baseline",
                0,
                baseline_authors,
                rng,
                args.max_per_author,
            ),
            sample_cohort(
                pool,
                args.baseline_country,
                "null_split_a",
                0,
                null_a,
                rng,
                args.max_per_author,
            ),
            sample_cohort(
                pool,
                args.baseline_country,
                "null_split_b",
                0,
                null_b,
                rng,
                args.max_per_author,
            ),
        ],
        ignore_index=True,
    )
    prompts.insert(0, "prompt_id", [f"wildchat_{i:05d}" for i in range(len(prompts))])
    for column, value in PROMPT_DEFAULTS.items():
        prompts[column] = value
    authors = prompts[["author_id", "cohort"]].drop_duplicates(ignore_index=True)
    for column, value in AUTHOR_DEFAULTS.items():
        authors[column] = [value] * len(authors) if value == [] else value

    args.out_dir.mkdir(parents=True, exist_ok=True)
    truncated_share = prompts.pop("truncated").mean()
    prompts.to_parquet(args.out_dir / "prompts.parquet", index=False)
    authors.to_parquet(args.out_dir / "authors.parquet", index=False)
    save_json(
        make_manifest(target_country, args.baseline_country).to_dict(),
        args.out_dir / "dataset.json",
    )
    write_readme(args, target_country, prompts, truncated_share)
    return prompts


def write_readme(
    args: argparse.Namespace,
    target_country: str,
    prompts: pd.DataFrame,
    truncated_share: float,
) -> None:
    counts = prompts.groupby("cohort").agg(
        prompts=("prompt_id", "size"), authors=("author_id", "nunique")
    )
    count_rows = "\n".join(
        f"| {cohort} | {row.prompts} | {row.authors} |"
        for cohort, row in counts.iterrows()
    )
    (args.out_dir / "README.md").write_text(
        f"""# WildChat country-contrast dataset (`wildchat`)

**Provenance.** First user turns of [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)
(Zhao et al., ICLR 2024), the public 837,989-conversation release (toxic and
journalist-flagged-PII conversations were removed by AI2 relative to the paper's
1M). Only shard `{SHARD}` (59,857 conversations) is read; it is downloaded to a
scratch directory, never stored in this repo. **License: ODC-BY** (attribution
required). Regenerate with:
`uv run python scripts/convert_wildchat.py --seed {args.seed} --n-authors {args.n_authors}`

**Task.** y = first-turn IP-geolocation country: target **{target_country}** (y=1)
vs baseline **{args.baseline_country}** (y=0). This is a PROXY label (location, not
self-reported identity or dialect) with **no published separability benchmark**
(`expected_accuracy` 0.0, expectation left exploratory). Validation-design role:
modality match (real chatbot prompts) + scale/asymptotics.

**Sampling (seed {args.seed}).** English-labeled conversations whose first turn is
non-redacted, non-toxic, has a country + hashed IP, and >= {args.min_words} words;
exact-duplicate texts dropped after truncation. **x = the first user turn,
truncated to its first {args.max_words} words** ({truncated_share:.0%} of kept prompts
were truncated). {args.n_authors} authors/side for target and baseline, plus
2x{args.n_authors} extra {args.baseline_country} authors (disjoint from baseline)
randomly halved into `null_split_a`/`null_split_b` — the negative-control
comparison (expectation "null": C2ST ~= 0.5, zero BH survivors, null MMD).
<= {args.max_per_author} prompts per author, seeded picks; author structure preserved.

| cohort | prompts | authors |
|---|---|---|
{count_rows}

**Author proxy caveat.** `author_id` = WildChat's per-turn `hashed_ip`. IPs shift
and can be shared (NAT/VPN), so author grouping is imperfect; in shard 0 no
hashed IP appears under more than one country, and cohorts are disjoint by
construction. Prompt counts per author are uneven (many IPs have one chat).

**Annotations.** Real logged prompts, so `provenance="real"`; everything else is
unrecorded: `markedness=0`, `codedness=0.0`, `topic_id=0`, `domain=""`, usage
ordinals 0; z/d author fields empty (WildChat has no demographics). Usage /
topical-survey / slice modules should skip gracefully.

**Large-n variant.** For MMD-power / C2ST-convergence asymptotics this converter
is deliberately one flag away: e.g. `--n-authors 500` (shard 0 holds ~940
{args.baseline_country} and ~170 {target_country} eligible authors; add shards
for more). Reruns overwrite `data/wildchat/`.
"""
    )


def verify(out_dir: Path) -> None:
    dataset = PromptDataset.load(out_dir)
    print(f"PromptDataset.load OK: {dataset.manifest.name}")
    counts = dataset.prompts.groupby("cohort").agg(
        prompts=("prompt_id", "size"), authors=("author_id", "nunique")
    )
    print(counts.to_string())
    words = dataset.prompts["text"].str.split().str.len()
    print(f"words/prompt: min {words.min()} median {words.median()} max {words.max()}")
    for cohort in ("target", "baseline", "null_split_a"):
        row = dataset.prompts[dataset.prompts["cohort"] == cohort].iloc[0]
        print(
            f"[{cohort}] {row.prompt_id} author={row.author_id[:12]}… "
            f"lgbtq={row.lgbtq}: {row.text[:110]!r}"
        )


if __name__ == "__main__":
    cli_args = parse_args()
    convert(cli_args)
    verify(cli_args.out_dir)
