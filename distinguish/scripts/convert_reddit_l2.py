"""Convert the Reddit-L2 corpus into this repo's dataset format.

Reddit-L2 (Rabinovich et al. TACL 2018; Goldin et al. EMNLP 2018, D18-1395):
English Reddit comments whose author's L1 is inferred from European-subreddit
country flair. We use the ``europe_data`` partition (posts written in the shared
r/europe / r/AskEurope subreddits), so the distinguishing signal between cohorts
is grammatical / stylistic (non-native English), NOT topical — every cohort
talks about the same things.

Cohorts:
  target        German L1 (non-native English), y=1
  baseline      Native English speakers (UK + US flairs), y=0
  null_split_a  Seeded random half of an INDEPENDENT German author pool, y=1
  null_split_b  The other half (negative control: same L1 -> expect null)

Source: the corpus is distributed via public Google Drive links published on the
now-archived Haifa CL project page (https://cl.haifa.ac.il/projects/L2/). The
official channel asks for a courtesy email to the authors (ellarabi@gmail.com);
this converter downloads the public ``reddit_full_posts_data.zip`` (Drive id
18iJuQJi_rIrarZjXJF-gdVR8ZddH-r7f) directly, extracting only the four members it
needs via HTTP range requests (no full 5.3 GB download, no gdown dependency).

Usage:
    uv run python scripts/convert_reddit_l2.py [--seed 0] [--out-dir data/reddit_l2]
"""

from __future__ import annotations

import argparse
import csv
import re
import struct
import sys
import urllib.request
import zlib
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from random import Random

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.common.base_schema import BaseSchema  # noqa: E402
from src.common.dataset_tables import (  # noqa: E402
    CohortSpec,
    ComparisonSpec,
    DatasetManifest,
    PromptDataset,
)
from src.common.file_io import ensure_dir, save_json  # noqa: E402

DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_datasets" / "reddit_l2"
DATASET_NAME = "reddit_l2"

# reddit_full_posts_data.zip on the archived Haifa CL Drive folder.
_ZIP_FILE_ID = "18iJuQJi_rIrarZjXJF-gdVR8ZddH-r7f"
_MEMBERS = {
    "reddit.Germany.tok.clean.csv": "reddit_full_posts_data/europe_data/reddit.Germany.tok.clean.csv",
    "reddit.UK.tok.clean.csv": "reddit_full_posts_data/europe_data/reddit.UK.tok.clean.csv",
    "reddit.US.tok.clean.csv": "reddit_full_posts_data/europe_data/reddit.US.tok.clean.csv",
    "readMe.txt": "reddit_full_posts_data/readMe.txt",
}

# Text unit sizing (words). Chunks pack consecutive comments of one user.
MIN_WORDS = 20
MAX_WORDS = 120

_UESC = re.compile(r"\\+u[0-9a-fA-F]{4}")
_ESCSEQ = re.compile(r"\\+[nrt]")
_WS = re.compile(r"\s+")
_ENTITIES = [
    ("& gt ;", ">"),
    ("& lt ;", "<"),
    ("& amp ;", "&"),
    ("&gt;", ">"),
    ("&lt;", "<"),
    ("&amp;", "&"),
]
_BOILERPLATE = (
    "questions about this removal",
    "contact the mods",
    "has been removed because",
    "megathread for posts of this kind",
    "this comment has been removed",
    "your submission has been removed",
)


@dataclass
class AuthorChunks(BaseSchema):
    """One author and the ordered text units (chunks) built from their posts."""

    author_id: str
    chunks: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Raw download (HTTP range extraction of selected zip members)                #
# --------------------------------------------------------------------------- #
def _download_url() -> str:
    page = f"https://drive.google.com/uc?id={_ZIP_FILE_ID}&export=download"
    with urllib.request.urlopen(page) as response:
        html = response.read().decode("utf-8", "replace")
    match = re.search(r'name="uuid" value="([^"]+)"', html)
    if not match:
        raise RuntimeError(
            "Could not obtain the Drive download confirmation token; the public "
            f"link for zip id {_ZIP_FILE_ID} may have changed."
        )
    return (
        "https://drive.usercontent.google.com/download"
        f"?id={_ZIP_FILE_ID}&export=download&confirm=t&uuid={match.group(1)}"
    )


def _fetch_range(url: str, start: int, end: int) -> tuple[bytes, int]:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request) as response:
        if response.status != 206:
            raise RuntimeError(
                "Drive did not honour an HTTP Range request (status "
                f"{response.status}); cannot stream-extract the zip."
            )
        total = int(response.headers["Content-Range"].split("/")[1])
        return response.read(), total


def _read_central_directory(
    url: str,
) -> tuple[dict[str, tuple[int, int, int, int]], int]:
    """Return {member_name: (method, crc, csize, local_header_offset)} and zip size."""
    _, total = _fetch_range(url, 0, 0)
    tail, _ = _fetch_range(url, max(0, total - 66560), total - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd == -1:
        raise RuntimeError("Zip end-of-central-directory record not found.")
    cd_size = struct.unpack("<I", tail[eocd + 12 : eocd + 16])[0]
    cd_off = total - 22 - cd_size
    cd, _ = _fetch_range(url, cd_off, cd_off + cd_size - 1)
    entries: dict[str, tuple[int, int, int, int]] = {}
    pos = 0
    while pos < len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            break
        fixed = struct.unpack("<IHHHHHHIIIHHHHHII", cd[pos : pos + 46])
        method, crc, csize = fixed[4], fixed[7], fixed[8]
        nlen, elen, clen, lho = fixed[10], fixed[11], fixed[12], fixed[16]
        name = cd[pos + 46 : pos + 46 + nlen].decode("utf-8", "replace")
        entries[name] = (method, crc, csize, lho)
        pos += 46 + nlen + elen + clen
    return entries, total


def _extract_member(url: str, meta: tuple[int, int, int, int], dest: Path) -> None:
    method, crc, csize, lho = meta
    header = None
    for candidate_off in (lho, lho + (1 << 32)):  # 32-bit offset wrap on >4 GB zips
        head, _ = _fetch_range(url, candidate_off, candidate_off + 29)
        if head[:4] == b"PK\x03\x04":
            header, lho = head, candidate_off
            break
    if header is None:
        raise RuntimeError(f"Local file header not found for {dest.name}.")
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    data_start = lho + 30 + name_len + extra_len
    raw, _ = _fetch_range(url, data_start, data_start + csize - 1)
    data = zlib.decompress(raw, -15) if method == 8 else raw
    if (zlib.crc32(data) & 0xFFFFFFFF) != crc:
        raise RuntimeError(f"CRC mismatch extracting {dest.name}.")
    dest.write_bytes(data)


def download_raw(raw_dir: Path) -> None:
    ensure_dir(raw_dir)
    if all((raw_dir / name).exists() for name in _MEMBERS):
        print(f"Raw files already present in {raw_dir}; skipping download.")
        return
    url = _download_url()
    entries, _ = _read_central_directory(url)
    for name, member in _MEMBERS.items():
        dest = raw_dir / name
        if dest.exists():
            continue
        if member not in entries:
            raise RuntimeError(f"Zip does not contain expected member {member}.")
        print(f"Downloading {name} ...")
        _extract_member(url, entries[member], dest)
        print(f"  wrote {dest} ({dest.stat().st_size} bytes, CRC ok)")


# --------------------------------------------------------------------------- #
# Text cleaning + chunking                                                    #
# --------------------------------------------------------------------------- #
def clean_text(post: str) -> str:
    text = _UESC.sub(" ", post)
    text = _ESCSEQ.sub(" ", text)
    text = text.replace("\\", " ")
    for src, dst in _ENTITIES:
        text = text.replace(src, dst)
    return _WS.sub(" ", text).strip()


def _is_boilerplate(text_lower: str) -> bool:
    return any(marker in text_lower for marker in _BOILERPLATE)


def _user_chunks(posts: list[str], max_units: int) -> list[str]:
    """Greedily pack an author's consecutive comments into <=MAX_WORDS chunks.

    Comments are cleaned, automod boilerplate and exact within-author duplicates
    are dropped, chunks shorter than MIN_WORDS are discarded, and a single
    comment longer than MAX_WORDS is truncated to MAX_WORDS words. Returns at
    most ``max_units`` chunks (early-stops once that many are available).
    """
    chunks: list[str] = []
    buffer: list[str] = []
    seen: set[str] = set()

    def flush() -> None:
        if len(buffer) >= MIN_WORDS:
            chunks.append(" ".join(buffer))

    for post in posts:
        if len(chunks) >= max_units:
            break
        cleaned = clean_text(post)
        if not cleaned or _is_boilerplate(cleaned.lower()) or cleaned in seen:
            continue
        seen.add(cleaned)
        tokens = cleaned.split()
        if len(tokens) > MAX_WORDS:
            flush()
            buffer = []
            chunks.append(" ".join(tokens[:MAX_WORDS]))
        elif len(buffer) + len(tokens) > MAX_WORDS:
            flush()
            buffer = tokens[:]
        else:
            buffer.extend(tokens)
    if len(chunks) < max_units:
        flush()
    return chunks[:max_units]


def eligible_authors(
    csv_path: Path, country: str, max_units: int
) -> list[AuthorChunks]:
    """Authors with >= max_units usable chunks, in deterministic file order."""
    csv.field_size_limit(sys.maxsize)
    result: list[AuthorChunks] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for user, rows in groupby(reader, key=lambda r: r["user"]):
            posts = [r["post"] for r in rows]
            chunks = _user_chunks(posts, max_units)
            if len(chunks) >= max_units:
                result.append(AuthorChunks(f"{country}:{user}", chunks))
    return result


# --------------------------------------------------------------------------- #
# Table construction                                                          #
# --------------------------------------------------------------------------- #
_PROMPT_DEFAULTS = {
    "markedness": 0,
    "codedness": 0.0,
    "topic_id": 0,
    "domain": "",
    "provenance": "",
    "adoption": 0,
    "general_freq": 0,
    "llm_freq": 0,
    "professional_freq": 0,
    "aversion": 0,
    "satisfaction": 0,
}
_AUTHOR_DEFAULTS = {
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

# Force list<string> for the (all-empty) identity list columns so the parquet
# schema matches a populated dataset (e.g. data/synthetic) instead of inferring
# list<null>.
_AUTHOR_SCHEMA = pa.schema(
    [
        ("author_id", pa.large_string()),
        ("cohort", pa.large_string()),
        ("transgender", pa.large_string()),
        ("gender", pa.list_(pa.string())),
        ("orientation", pa.list_(pa.string())),
        ("pronouns", pa.list_(pa.string())),
        ("race", pa.list_(pa.string())),
        ("age", pa.large_string()),
        ("disability", pa.large_string()),
        ("education", pa.large_string()),
        ("income", pa.large_string()),
    ]
)


def build_tables(
    assignments: list[tuple[str, int, AuthorChunks]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """assignments: (cohort_name, lgbtq_flag, AuthorChunks)."""
    prompt_rows: list[dict] = []
    author_rows: list[dict] = []
    counter = 0
    for cohort, lgbtq, author in assignments:
        author_rows.append(
            dict(author_id=author.author_id, cohort=cohort, **_AUTHOR_DEFAULTS)
        )
        for text in author.chunks:
            prompt_rows.append(
                dict(
                    prompt_id=f"{DATASET_NAME}_{counter:05d}",
                    author_id=author.author_id,
                    cohort=cohort,
                    text=text,
                    lgbtq=lgbtq,
                    **_PROMPT_DEFAULTS,
                )
            )
            counter += 1
    prompts = pd.DataFrame(prompt_rows)
    authors = pd.DataFrame(author_rows)
    for column in ("gender", "orientation", "pronouns", "race"):
        authors[column] = authors[column].apply(lambda _v: [])
    return prompts, authors


def build_manifest(counts: dict[str, int]) -> DatasetManifest:
    cohorts = [
        CohortSpec(
            name="target",
            group="target",
            display_name="German L1 (non-native English, y=1)",
            description=(
                "Reddit-L2 europe_data comments by users with a German country "
                "flair (r/europe, r/AskEurope). English-as-second-language "
                "writers; the y=1 target cohort."
            ),
        ),
        CohortSpec(
            name="baseline",
            group="baseline",
            display_name="Native English (UK/US flairs, y=0)",
            description=(
                "Reddit-L2 europe_data comments by users with UK or US country "
                "flair (native English speakers); the y=0 baseline cohort."
            ),
        ),
        CohortSpec(
            name="null_split_a",
            group="target",
            display_name="German null split A",
            description=(
                "Seeded random half of an INDEPENDENT pool of German-flair "
                "authors, disjoint from the target cohort. Same L1/distribution "
                "as null_split_b — the negative-control partner."
            ),
        ),
        CohortSpec(
            name="null_split_b",
            group="target",
            display_name="German null split B",
            description=(
                "The other seeded half of the independent German-flair author "
                "pool. Paired with null_split_a to check the pipeline reports a "
                "null when both sides share one L1."
            ),
        ),
    ]
    comparisons = [
        ComparisonSpec(
            name="target_vs_baseline",
            target_cohort="target",
            baseline_cohort="baseline",
            expectation="distinguishable",
            explorations=True,
            expected_accuracy=0.91,
            notes=(
                "Positive control (known-target L1 axis). Goldin et al. 2018 "
                "report 91.07% in-domain binary native/non-native accuracy with "
                "content features. Signal is grammatical/stylistic, NOT topical: "
                "all cohorts post in the shared r/europe subreddits, so syntactic "
                "and semantic dimensions should fire while topical JSD stays "
                "comparatively low. Flair-derived L1 labels are self-selected and "
                "noisy, which caps achievable accuracy below a clean-label ceiling."
            ),
        ),
        ComparisonSpec(
            name="null_control",
            target_cohort="null_split_a",
            baseline_cohort="null_split_b",
            expectation="null",
            explorations=False,
            expected_accuracy=0.0,
            notes=(
                "Negative control: a seeded random author-split of one group "
                "(German L1). Same distribution on both sides -> expect C2ST "
                "~0.5, no BH survivors, and a null MMD."
            ),
        ),
    ]
    _ = counts
    return DatasetManifest(
        name=DATASET_NAME,
        description=(
            "Reddit-L2 (Rabinovich et al. 2018; Goldin et al. 2018): English "
            "Reddit comments in shared European subreddits, with author L1 "
            "inferred from country flair. target=German L1 (non-native, y=1) vs "
            "baseline=native UK/US (y=0); the distinguishing signal is "
            "grammatical/stylistic rather than topical. null_split_a vs "
            "null_split_b is a seeded within-German negative control."
        ),
        cohorts=cohorts,
        comparisons=comparisons,
    )


def _sample(rng: Random, pool: list[AuthorChunks], k: int) -> list[AuthorChunks]:
    if len(pool) < k:
        raise RuntimeError(f"Not enough eligible authors: need {k}, have {len(pool)}.")
    ordered = sorted(pool, key=lambda a: a.author_id)
    return rng.sample(ordered, k)


# --------------------------------------------------------------------------- #
# README                                                                      #
# --------------------------------------------------------------------------- #
def write_readme(
    out_dir: Path, counts: dict[str, int], args: argparse.Namespace
) -> None:
    lines = [
        "# Reddit-L2 (native vs non-native English)",
        "",
        "Converted for the prompt-distinguishability pipeline "
        "(`D=(x,y,z,d,c)` tables).",
        "",
        "## Provenance",
        "",
        "- **Corpus**: Reddit-L2 (Rabinovich, Nisioi, Ordan & Wintner, *TACL* "
        "2018; Goldin, Rabinovich & Wintner, *EMNLP* 2018, D18-1395). English "
        "Reddit comments; author first language (L1) is inferred from the "
        "country flair of European subreddits (r/europe, r/AskEurope).",
        "- **Access**: the corpus is distributed via public Google Drive links "
        "on the now-archived Haifa CL project page "
        "(https://cl.haifa.ac.il/projects/L2/). The official channel asks for a "
        "courtesy email to the authors (ellarabi@gmail.com) — please do so for "
        "any substantial use. This converter downloads the public "
        "`reddit_full_posts_data.zip` (Drive id "
        "`18iJuQJi_rIrarZjXJF-gdVR8ZddH-r7f`) and extracts only four members "
        "(`europe_data/reddit.{Germany,UK,US}.tok.clean.csv` + `readMe.txt`) via "
        "HTTP range requests — no full 5.3 GB download and no `gdown` dependency. "
        "Raw files land in a scratch dir OUTSIDE the repo.",
        "- **License / terms**: research use; cite the two papers above. Raw "
        "Reddit text is user-generated content under Reddit's terms; only "
        "derived, truncated snippets are stored here.",
        "",
        "## What each row is",
        "",
        f"- **x (text unit)**: a *chunk* of one author's consecutive comments, "
        f"greedily packed to at most **{MAX_WORDS} words** (min **{MIN_WORDS} "
        f"words**). A single comment longer than {MAX_WORDS} words is TRUNCATED "
        f"to its first {MAX_WORDS} words. Automod/removal boilerplate and exact "
        "within-author duplicate comments are dropped before packing. The "
        "source `europe_data` comments are pre-tokenized (space-separated); we "
        "additionally normalize escaped newlines, mangled `\\uXXXX` escapes, and "
        "spaced HTML entities (`& gt ;` -> `>` etc.).",
        "- **y (`lgbtq` flag = generic target flag)**: 1 for German-L1 cohorts, "
        "0 for the native baseline.",
        "- **z / d / c**: Reddit-L2 carries no self-reported identity, "
        "demographics, or interaction context, so all author identity/demographic "
        "fields and all prompt context fields are left at their unrecorded "
        'defaults (`""`, `[]`, `0`, `0.0`). The only known signal — L1 — is '
        "encoded by the cohort, not by a per-row column (topic_id stays 0 so the "
        "topical module treats it as unrecorded rather than a survey topic).",
        "",
        "## Cohorts",
        "",
        "| cohort | group | y | source flair | n_authors | n_prompts |",
        "|---|---|---|---|---|---|",
        f"| target | target | 1 | Germany | {counts['target_a']} | {counts['target_p']} |",
        f"| baseline | baseline | 0 | UK + US | {counts['baseline_a']} | {counts['baseline_p']} |",
        f"| null_split_a | target | 1 | Germany (indep.) | {counts['null_a_a']} | {counts['null_a_p']} |",
        f"| null_split_b | target | 1 | Germany (indep.) | {counts['null_b_a']} | {counts['null_b_p']} |",
        "",
        f"The baseline draws {args.n_baseline // 2} UK + {args.n_baseline // 2} US "
        "authors. The two null-split cohorts come from a German author pool that "
        "is disjoint from the `target` cohort, then split in half by seed — a "
        "same-distribution (German-vs-German) negative control.",
        "",
        "## Comparisons",
        "",
        "- **target_vs_baseline** — expectation `distinguishable`, "
        "`expected_accuracy=0.91`. Positive control on the L1 axis. Goldin et "
        "al. 2018 report 91.07% in-domain binary native/non-native accuracy. The "
        "signal here is grammatical/stylistic, not topical (shared subreddits): "
        "syntactic + semantic should fire while topical JSD stays comparatively "
        "low. Flair-based labels are noisy, capping accuracy.",
        "- **null_control** — expectation `null`, `expected_accuracy=0.0`. Seeded "
        "random split of the German group; expect C2ST ~0.5, no BH survivors, "
        "null MMD.",
        "",
        "## Sampling",
        "",
        f"- Seed: `{args.seed}`.",
        f"- Eligible authors = users with at least {args.max_units} usable chunks; "
        f"up to {args.max_units} chunks kept per author (first N in file order).",
        f"- target: {args.n_target} German authors. baseline: {args.n_baseline} "
        f"native authors ({args.n_baseline // 2} UK + {args.n_baseline // 2} US). "
        f"null pool: {2 * args.n_null} German authors (disjoint from target), "
        f"split into {args.n_null} + {args.n_null}.",
        "- Author structure is preserved (every prompt's `author_id` is present "
        "in `authors.parquet`).",
        "",
        "## Caveats",
        "",
        "- **Noisy labels**: L1 is inferred from self-selected subreddit country "
        "flair, not verified. Some flaired users may be heritage/near-native "
        "speakers; this caps separability and is the intended realism.",
        "- **Truncation/segmentation**: chunking + the 120-word cap change the "
        "unit relative to the original per-comment / 100-sentence-chunk formats; "
        "absolute accuracy is not comparable to the papers' feed-level numbers.",
        "- **Confounds**: even within shared subreddits, German-flair users may "
        "raise Germany-specific topics; report topical JSD alongside syntactic/"
        "semantic to attribute the signal.",
        "- **Ethics**: real user comments. Only truncated snippets are stored; "
        "author ids are the corpus's opaque usernames, namespaced by flair.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Verification                                                                #
# --------------------------------------------------------------------------- #
def verify(out_dir: Path) -> None:
    dataset = PromptDataset.load(out_dir)
    print("\nPromptDataset.load() OK — validate() passed.")
    print("\nPer-cohort counts:")
    per_cohort = dataset.prompts.groupby("cohort").agg(
        n_prompts=("prompt_id", "count"),
        n_authors=("author_id", "nunique"),
        y=("lgbtq", "max"),
    )
    print(per_cohort.to_string())
    print("\nBuilt PromptSets (sanity):")
    for cohort in ("target", "baseline", "null_split_a", "null_split_b"):
        prompt_set = dataset.prompt_set(cohort)
        print(
            f"  {cohort:<14} authors={len(prompt_set.authors):>3} "
            f"prompts={len(prompt_set.prompts):>4}"
        )
    print("\n3 sample prompt rows (target):")
    sample = dataset.prompts[dataset.prompts["cohort"] == "target"].head(3)
    for _, row in sample.iterrows():
        print(
            f"  [{row['prompt_id']}] author={row['author_id']} lgbtq={row['lgbtq']} "
            f"topic_id={row['topic_id']} provenance='{row['provenance']}'"
        )
        print(f"      text: {row['text'][:160]}")
    print("\n1 sample author row (baseline):")
    author = dataset.authors[dataset.authors["cohort"] == "baseline"].iloc[0]
    print(f"  {author.to_dict()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "data" / DATASET_NAME
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--n-target", type=int, default=80)
    parser.add_argument(
        "--n-baseline", type=int, default=80, help="native authors (split UK/US)"
    )
    parser.add_argument("--n-null", type=int, default=70, help="authors per null split")
    parser.add_argument("--max-units", type=int, default=4, help="chunks per author")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = ensure_dir(Path(args.out_dir))
    download_raw(raw_dir)

    print("Building author chunks (this streams the raw CSVs)...")
    german = eligible_authors(
        raw_dir / "reddit.Germany.tok.clean.csv", "DE", args.max_units
    )
    uk = eligible_authors(raw_dir / "reddit.UK.tok.clean.csv", "UK", args.max_units)
    us = eligible_authors(raw_dir / "reddit.US.tok.clean.csv", "US", args.max_units)
    print(f"  eligible authors: German={len(german)} UK={len(uk)} US={len(us)}")

    rng = Random(args.seed)
    target_authors = _sample(rng, german, args.n_target)
    used = {a.author_id for a in target_authors}
    german_rest = [a for a in german if a.author_id not in used]
    null_pool = _sample(rng, german_rest, 2 * args.n_null)
    null_a = null_pool[: args.n_null]
    null_b = null_pool[args.n_null :]
    half = args.n_baseline // 2
    baseline_authors = _sample(rng, uk, half) + _sample(rng, us, half)

    assignments: list[tuple[str, int, AuthorChunks]] = []
    assignments += [("target", 1, a) for a in target_authors]
    assignments += [("baseline", 0, a) for a in baseline_authors]
    assignments += [("null_split_a", 1, a) for a in null_a]
    assignments += [("null_split_b", 1, a) for a in null_b]

    prompts, authors = build_tables(assignments)
    counts = {
        "target_a": len(target_authors),
        "target_p": args.n_target * args.max_units,
        "baseline_a": len(baseline_authors),
        "baseline_p": len(baseline_authors) * args.max_units,
        "null_a_a": len(null_a),
        "null_a_p": len(null_a) * args.max_units,
        "null_b_a": len(null_b),
        "null_b_p": len(null_b) * args.max_units,
    }
    manifest = build_manifest(counts)

    save_json(manifest.to_dict(), out_dir / "dataset.json")
    prompts.to_parquet(out_dir / "prompts.parquet", index=False)
    authors_table = pa.Table.from_pandas(
        authors, schema=_AUTHOR_SCHEMA, preserve_index=False
    )
    pq.write_table(authors_table, out_dir / "authors.parquet")
    write_readme(out_dir, counts, args)
    print(f"\nWrote dataset to {out_dir}")

    verify(out_dir)


if __name__ == "__main__":
    main()
