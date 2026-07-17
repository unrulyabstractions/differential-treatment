"""Convert TwitterAAE (Blodgett/Green/O'Connor EMNLP 2016) to the D=(x,y,z,d,c) tables.

x = tweet text; y = the demographic-model posterior alignment: `target` =
AA-aligned (African-American posterior >= 0.8, lgbtq/generic-y flag 1) vs
`baseline` = White-aligned (White posterior >= 0.8, flag 0). Every prompt is
its own author: the TwitterAAE `all` file has only partial/reused author ids,
so we deliberately use the streamed ROW INDEX as author_id (one prompt per
author) and say so here + in the README. z/d/c are entirely unannotated
(defaults) - the corpus carries no self-reported identity, only the model
posterior. Includes the mandatory negative-control comparison: a seeded random
author split within the White-aligned pool (null_split_a vs null_split_b,
expectation "null").

**Circularity**: the y label was assigned FROM the tweet text (the posterior of
a lexical demographic model), so any lexical/semantic separability is
definitional. This dataset is an end-to-end SMOKE TEST only - see the manifest
notes and README.

Idempotent: streams only the head of the 5.9 GB `TwitterAAE-full-v1.zip`
(HTTP range requests against the `twitteraae_all` member, cached compressed to
--raw-dir; NEVER into the repo), then rewrites
data/twitteraae/{dataset.json,prompts.parquet,authors.parquet,README.md}.

Usage: uv run python scripts/convert_twitteraae.py [--seed 0] [--chunk-mb 32]
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.dataset_tables import (
    CohortSpec,
    ComparisonSpec,
    DatasetManifest,
    PromptDataset,
)
from src.common.file_io import ensure_dir, save_json

DATASET_NAME = "twitteraae"
ZIP_URL = "http://slanglab.cs.umass.edu/TwitterAAE/TwitterAAE-full-v1.zip"
# twitteraae_all member local-header offset, read from the zip central directory.
MEMBER_LOCAL_OFFSET = 1_579_776_134
MEMBER_NAME = "TwitterAAE-full-v1/twitteraae_all"
DEFAULT_RAW_DIR = str(Path(__file__).resolve().parent.parent / "data" / "raw_datasets" / "twitteraae")

POSTERIOR_THRESHOLD = 0.8  # cohort membership: AA (target) / White (baseline)
MIN_WORDS, MAX_WORDS = 6, 120  # short tweets carry little signal; cap per rule
MIN_ASCII_SHARE = 0.9  # drop non-English (posteriors label many foreign tweets)
POOL_CAP = 4000  # bound the collected pool per cohort

PROMPT_DEFAULTS = {
    "markedness": 0,
    "codedness": 0.0,
    "topic_id": 0,
    "domain": "",
    "adoption": 0,
    "general_freq": 0,
    "llm_freq": 0,
    "professional_freq": 0,
    "aversion": 0,
    "satisfaction": 0,
}
AUTHOR_DEFAULTS = {
    "transgender": "",
    "gender": [],
    "orientation": [],
    "pronouns": [],
    "race": [],
    "age": "",
    "disability": "",
    "education": "",
    "income": "",
}

CIRCULARITY_NOTE = (
    "End-to-end smoke test (validation-design role: ALL modules must fire). "
    "CIRCULAR by construction - smoke test ONLY: every module must fire (many BH "
    "survivors, huge MMD, C2ST >> 0.5); a silent module is broken. Document that "
    "success says NOTHING about detecting self-reported identity. The y label was "
    "assigned FROM the tweet text (posterior of Blodgett et al.'s lexical "
    "demographic model >= 0.8), so any lexical/semantic separability is "
    "definitional, not a discovery."
)
NULL_NOTE = (
    "Negative control: seeded random author split within the White-aligned pool "
    "(disjoint from the baseline cohort). Expect C2ST ~= 0.5, zero BH survivors, "
    "null MMD."
)


def _acceptable(text: str) -> bool:
    if text.startswith(("RT ", "@")) or "http" in text:
        return False
    if len(text.split()) < MIN_WORDS:
        return False
    ascii_share = sum(ch.isascii() for ch in text) / max(len(text), 1)
    return ascii_share >= MIN_ASCII_SHARE


def _truncate(text: str) -> str:
    return " ".join(text.split()[:MAX_WORDS])


def download_member_prefix(raw_dir: Path, chunk_mb: int) -> Path:
    """Range-download the first `chunk_mb` MB of the twitteraae_all member (cached)."""
    ensure_dir(raw_dir)
    prefix_path = raw_dir / f"twitteraae_all_prefix_{chunk_mb}mb.bin"
    n_bytes = chunk_mb * 1024 * 1024
    if prefix_path.exists() and prefix_path.stat().st_size >= n_bytes:
        return prefix_path
    start = MEMBER_LOCAL_OFFSET
    end = MEMBER_LOCAL_OFFSET + n_bytes - 1
    request = urllib.request.Request(ZIP_URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        data = response.read()
    prefix_path.write_bytes(data)
    return prefix_path


def _iter_member_lines(prefix_path: Path):
    """Decompress the cached (truncated) deflate prefix and yield complete lines."""
    raw = prefix_path.read_bytes()
    sig, _ver, _flags, method, _mt, _md, _crc, _cs, _us, nlen, elen = struct.unpack(
        "<IHHHHHIIIHH", raw[:30]
    )
    if sig != 0x04034B50:
        raise ValueError(f"Not a local file header at MEMBER_LOCAL_OFFSET: {hex(sig)}")
    name = raw[30 : 30 + nlen].decode()
    if name != MEMBER_NAME or method != 8:
        raise ValueError(f"Unexpected member {name!r} (method {method})")
    body = raw[30 + nlen + elen :]
    decompressor = zlib.decompressobj(-15)  # raw deflate
    remainder = ""
    step = 1 << 20
    for offset in range(0, len(body), step):
        try:
            chunk = decompressor.decompress(body[offset : offset + step])
        except zlib.error:
            break  # truncated prefix ended inside a block; use what we have
        remainder += chunk.decode("utf-8", errors="replace")
        lines = remainder.split("\n")
        remainder = lines.pop()  # keep the (possibly partial) trailing line
        yield from lines


def collect_pools(prefix_path: Path):
    """Parse the member prefix into AA / White candidate prompts (row-indexed)."""
    aa_pool: list[dict] = []
    white_pool: list[dict] = []
    seen_texts: set[str] = set()
    for row_index, line in enumerate(_iter_member_lines(prefix_path)):
        if len(aa_pool) >= POOL_CAP and len(white_pool) >= POOL_CAP:
            break
        parts = line.split("\t")
        if len(parts) != 10:
            continue
        try:
            text = json.loads(parts[5])
            posteriors = [float(x) for x in parts[6:10]]  # AA, Hispanic, Other, White
        except (ValueError, json.JSONDecodeError):
            continue
        if any(p != p for p in posteriors):  # nan posterior (no in-vocab words)
            continue
        text = text.strip()
        if not text or text in seen_texts or not _acceptable(text):
            continue
        candidate = {
            "author_id": f"{DATASET_NAME}_row_{row_index}",
            "text": _truncate(text),
        }
        if posteriors[0] >= POSTERIOR_THRESHOLD and len(aa_pool) < POOL_CAP:
            seen_texts.add(text)
            aa_pool.append(candidate)
        elif posteriors[3] >= POSTERIOR_THRESHOLD and len(white_pool) < POOL_CAP:
            seen_texts.add(text)
            white_pool.append(candidate)
    return aa_pool, white_pool


def build_rows(seed: int, per_side: int, raw_dir: Path, chunk_mb: int):
    prefix_path = download_member_prefix(raw_dir, chunk_mb)
    aa_pool, white_pool = collect_pools(prefix_path)
    if len(aa_pool) < per_side:
        raise RuntimeError(
            f"Only {len(aa_pool)} AA-aligned tweets found in the {chunk_mb} MB prefix; "
            f"need {per_side}. Increase --chunk-mb."
        )
    null_total = 2 * min(per_side, (len(white_pool) - per_side) // 2)
    if len(white_pool) < per_side + null_total:
        raise RuntimeError(
            f"Only {len(white_pool)} White-aligned tweets; need "
            f"{per_side + null_total}. Increase --chunk-mb."
        )
    rng = np.random.default_rng(seed)
    aa_idx = rng.choice(len(aa_pool), size=per_side, replace=False)
    white_idx = rng.choice(len(white_pool), size=per_side + null_total, replace=False)
    baseline_idx = white_idx[:per_side]
    null_idx = white_idx[per_side:]
    null_half = null_total // 2

    cohort_pick = {
        "target": [(aa_pool[i], 1) for i in aa_idx],
        "baseline": [(white_pool[i], 0) for i in baseline_idx],
        "null_split_a": [(white_pool[i], 0) for i in null_idx[:null_half]],
        "null_split_b": [(white_pool[i], 0) for i in null_idx[null_half:]],
    }

    prompt_rows, author_rows = [], []
    counter = 0
    for cohort, picks in cohort_pick.items():
        for candidate, y in picks:
            author_rows.append(
                dict(author_id=candidate["author_id"], cohort=cohort, **AUTHOR_DEFAULTS)
            )
            prompt_rows.append(
                dict(
                    prompt_id=f"{DATASET_NAME}_{counter:05d}",
                    author_id=candidate["author_id"],
                    cohort=cohort,
                    text=candidate["text"],
                    lgbtq=y,
                    provenance=DATASET_NAME,
                    **PROMPT_DEFAULTS,
                )
            )
            counter += 1
    return prompt_rows, author_rows, len(aa_pool), len(white_pool), null_half


def build_manifest() -> DatasetManifest:
    return DatasetManifest(
        name=DATASET_NAME,
        description=(
            "TwitterAAE (Blodgett/Green/O'Connor EMNLP 2016, research-use): tweet "
            "text, cohorts by the demographic model's posterior alignment - "
            "AA-aligned (>=0.8, target) vs White-aligned (>=0.8, baseline) - plus a "
            "null author split within the White-aligned pool. SMOKE TEST ONLY: the "
            "label is assigned from the text, so separability is circular. One "
            "prompt per author (row-index author ids). See data/twitteraae/README.md."
        ),
        cohorts=[
            CohortSpec(
                "target",
                "target",
                "AA-aligned tweets (y=1)",
                "Tweets with African-American posterior >= 0.8 under the "
                "Blodgett et al. mixed-membership demographic model.",
            ),
            CohortSpec(
                "baseline",
                "baseline",
                "White-aligned tweets (y=0)",
                "Tweets with White posterior >= 0.8 under the same model.",
            ),
            CohortSpec(
                "null_split_a",
                "target",
                "Null split A (White-aligned)",
                "Random half of held-out White-aligned tweets (seeded); "
                "null-control cohort.",
            ),
            CohortSpec(
                "null_split_b",
                "baseline",
                "Null split B (White-aligned)",
                "Other random half of held-out White-aligned tweets (seeded); "
                "null-control cohort.",
            ),
        ],
        comparisons=[
            ComparisonSpec(
                "target_vs_baseline",
                "target",
                "baseline",
                expectation="distinguishable",
                explorations=True,
                expected_accuracy=0.0,
                notes=CIRCULARITY_NOTE,
            ),
            ComparisonSpec(
                "null_split",
                "null_split_a",
                "null_split_b",
                expectation="null",
                explorations=False,
                expected_accuracy=0.5,
                notes=NULL_NOTE,
            ),
        ],
    )


README_TEMPLATE = """\
# TwitterAAE (converted)

Tweet text labelled by a demographic language model's posterior, converted to the
paper's D=(x,y,z,d,c) table format (`src/common/dataset_tables.py`). Regenerate
with `uv run python scripts/convert_twitteraae.py --seed {seed} --chunk-mb {chunk_mb}`
(idempotent).

## Provenance and license

- Source: `http://slanglab.cs.umass.edu/TwitterAAE/TwitterAAE-full-v1.zip`
  (5.9 GB). **Research use only**; cite Blodgett, Green & O'Connor,
  "Demographic Dialectal Variation in Social Media: A Case Study of
  African-American English", EMNLP 2016 (and Blodgett et al. ACL 2018).
- We do **not** download the full 5.9 GB zip. The converter issues HTTP range
  requests to read only the first **{chunk_mb} MB (compressed)** of the
  `twitteraae_all` member (via the zip central directory offset), caches that
  prefix to the scratch --raw-dir, and stream-decompresses it. The member is
  user-id-sorted, so the prefix is a contiguous head slice of the corpus - fine
  for a smoke test, documented here.
- Each `twitteraae_all` line is 10 tab-separated fields: tweet id, timestamp,
  user id, geo-coordinates, Census blockgroup, tweet text (JSON string), and the
  four model posteriors **[African-American, Hispanic, Other, White]**.

## Cohorts and sampling (seed={seed})

- y = the model's posterior alignment: `target` = African-American posterior
  **>= {threshold}** (lgbtq flag 1 = generic y), `baseline` = White posterior
  **>= {threshold}** (0). This is a lexical demographic model, NOT self-report.
- **One prompt per author, author_id = the streamed ROW INDEX.** The
  `twitteraae_all` file's user ids are partial/reused and we make no attempt to
  group by them; per the task spec each tweet is treated as its own author
  (`twitteraae_row_<n>`). There is therefore no within-author structure.
- x = tweet text, JSON-decoded, stripped, with interior whitespace
  (newlines/tabs/repeated spaces) collapsed to single spaces. Filters: dropped
  retweets (`RT `),
  tweets containing URLs, tweets under {min_words} words, and non-English tweets
  (< {ascii_pct}% ASCII characters - the corpus posterior-labels many foreign
  tweets); corpus-wide exact-duplicate texts removed. Tweets are short, but any
  text over **{max_words} words is truncated** to the first {max_words} for the
  per-unit length cap.
- Seeded subsample: {per_side} tweets per side (AA pool {aa_pool} candidates,
  White pool {white_pool} candidates in the prefix).
- `null_split_a` / `null_split_b`: negative control - {null_half}+{null_half}
  held-out **White-aligned** tweets (disjoint from `baseline`), split at random
  with the same seed. Expect C2ST ~= 0.5, zero BH survivors, null MMD.

## z / d / c annotations

- **Entirely unannotated.** The corpus has no self-reported identity or
  demographics - only the model posterior, which we encode solely as the cohort
  label. So `transgender`/`disability`/`education`/`income`/`age` = "",
  `gender`/`orientation`/`pronouns`/`race` = [], `markedness` 0, `codedness` 0.0,
  `topic_id`/`domain` and all usage ordinals 0 ("0 = unrecorded"). Usage,
  topical, and identity-slice modules should skip gracefully.
- `provenance` = "twitteraae" (dataset tag, not the paper's real/hyp flag).

## Caveats (READ THIS)

- **The label is circular.** {circularity}
- **SMOKE TEST ONLY.** High separability here validates that the pipeline's
  modules fire on an obvious signal; it says nothing about detecting real,
  self-reported identity from prompts.
- `expected_accuracy` is set to **0.0** for `target_vs_baseline`: there is no
  meaningful published classification benchmark (the label is definitional), so
  a large deviation is not interpretable as over/under-performance.
- The prefix is a contiguous, user-sorted head slice, not a uniform sample of
  the 59 M-tweet corpus; treat cohort composition as illustrative.
- Tweets contain informal, sometimes offensive social-media language.
"""


def write_readme(out_dir: Path, args, aa_pool, white_pool, null_half) -> None:
    (out_dir / "README.md").write_text(
        README_TEMPLATE.format(
            seed=args.seed,
            chunk_mb=args.chunk_mb,
            threshold=POSTERIOR_THRESHOLD,
            min_words=MIN_WORDS,
            max_words=MAX_WORDS,
            ascii_pct=int(MIN_ASCII_SHARE * 100),
            per_side=args.tweets_per_side,
            aa_pool=aa_pool,
            white_pool=white_pool,
            null_half=null_half,
            circularity=CIRCULARITY_NOTE,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tweets-per-side", type=int, default=80)
    parser.add_argument(
        "--chunk-mb",
        type=int,
        default=32,
        help="compressed MB of the twitteraae_all member to stream",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path(DEFAULT_RAW_DIR))
    parser.add_argument("--out-dir", type=Path, default=Path("data") / DATASET_NAME)
    args = parser.parse_args()

    prompt_rows, author_rows, aa_pool, white_pool, null_half = build_rows(
        args.seed, args.tweets_per_side, args.raw_dir, args.chunk_mb
    )
    out_dir = ensure_dir(args.out_dir)
    pd.DataFrame(prompt_rows).to_parquet(out_dir / "prompts.parquet", index=False)
    pd.DataFrame(author_rows).to_parquet(out_dir / "authors.parquet", index=False)
    save_json(build_manifest().to_dict(), out_dir / "dataset.json")
    write_readme(out_dir, args, aa_pool, white_pool, null_half)

    dataset = PromptDataset.load(out_dir)  # runs validate()
    print(f"Loaded + validated {out_dir}")
    counts = dataset.prompts.groupby("cohort").agg(
        prompts=("prompt_id", "size"),
        authors=("author_id", "nunique"),
        y_mean=("lgbtq", "mean"),
    )
    print(counts)
    print(
        dataset.prompts.sample(3, random_state=args.seed)[
            ["prompt_id", "cohort", "author_id", "lgbtq", "text"]
        ].to_string(max_colwidth=90)
    )


if __name__ == "__main__":
    main()
