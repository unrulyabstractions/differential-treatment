"""Convert the Blog Authorship Corpus to our Parquet dataset format.

Source: HF `tasksource/blog_authorship_corpus` (single `blogtext.csv`, 681,284
posts by 19,320 bloggers; Schler, Koppel, Argamon & Pennebaker 2006). The
canonical `barilan/blog_authorship_corpus` is a loading-script dataset modern
`datasets` refuses to run, so we use the plain-CSV mirror.

Contrast: y = gender, target = female bloggers (y=1) vs baseline = male
bloggers (y=0). The published 2-way gender accuracy is 80.1% (Schler et al.
2006, Multi-Class Real Winnow over style+content features) — our
`expected_accuracy` anchor. NOTE our x-unit is a POST SEGMENT (first <=120
words of one post), far shorter than the per-author, full-post feature vectors
the 80.1% was measured on, so per-prompt separability is expected to run below
that ceiling; per-author aggregation is the right comparison.

Negative control (validation-design requirement): a seeded random author split
WITHIN the female group -> cohorts null_split_a / null_split_b, comparison
expectation "null".

Idempotent CLI: downloads the raw CSV into the scratchpad (never the repo),
subsamples with `--seed`, and writes
data/blog_authorship/{dataset.json,prompts.parquet,authors.parquet,README.md}.

    uv run python scripts/convert_blog_authorship.py --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import hf_hub_download  # noqa: E402

from src.common.dataset_tables import (  # noqa: E402
    CohortSpec,
    ComparisonSpec,
    DatasetManifest,
    PromptDataset,
)
from src.common.file_io import ensure_dir, save_json  # noqa: E402

DATASET_NAME = "blog_authorship"
HF_REPO = "tasksource/blog_authorship_corpus"
HF_FILE = "blogtext.csv"
SCRATCH_RAW = Path(
    "/private/tmp/claude-501/-Users-unrulyabstractions-work-prompt-distinguishability/"
    "140b8c69-9c33-43d8-8683-a04e90d30bcf/scratchpad/raw_datasets/blog_authorship"
)
MAX_SEGMENT_WORDS = 120
MIN_POST_WORDS = 20
MIN_POSTS_PER_AUTHOR = 2
AGE_GROUPS = ("13-17", "20s", "30s+")  # Schler's three sampled age bands

# Unknown-annotation defaults (task convention: "0 = unrecorded").
PROMPT_DEFAULTS = {
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


def age_bracket(age: int) -> str:
    """Map a numeric age to our canonical demographic bracket."""
    if age <= 17:
        return "13-17"
    if age <= 24:
        return "18-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    return "45-54"


def age_group(age: int) -> str:
    """Coarse band for balancing (matches the corpus's three sampled bands)."""
    if age <= 17:
        return "13-17"
    if age <= 27:
        return "20s"
    return "30s+"


def even_split(total: int, k: int = 3) -> list[int]:
    """Split `total` across `k` bins as evenly as possible (largest first)."""
    base, rem = divmod(total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def download_raw() -> Path:
    """Idempotently fetch blogtext.csv into the scratchpad raw dir."""
    ensure_dir(SCRATCH_RAW)
    target = SCRATCH_RAW / HF_FILE
    if target.exists() and target.stat().st_size > 0:
        return target
    hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILE,
        repo_type="dataset",
        local_dir=str(SCRATCH_RAW),
    )
    return target


def load_clean_posts(csv_path: Path) -> pd.DataFrame:
    """Load posts, clean text, keep substantive ones, add a <=120-word segment."""
    df = pd.read_csv(csv_path)
    clean = (
        df["text"]
        .str.replace("urlLink", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    word_lists = clean.str.split()
    word_count = word_lists.str.len()
    keep = word_count >= MIN_POST_WORDS
    df = df[keep].copy()
    df["segment"] = word_lists[keep].str[:MAX_SEGMENT_WORDS].str.join(" ")
    return df.reset_index(drop=True)


def author_table(posts: pd.DataFrame) -> pd.DataFrame:
    """One row per author with their list of clean-post row indices."""
    by_author = posts.groupby("id").indices  # id -> np.ndarray of row positions
    rows = []
    for blog_id, idx in by_author.items():
        if len(idx) < MIN_POSTS_PER_AUTHOR:
            continue
        first = posts.iloc[idx[0]]
        rows.append(
            {
                "blog_id": int(blog_id),
                "gender": first["gender"],
                "age": int(first["age"]),
                "group": age_group(int(first["age"])),
                "post_rows": np.asarray(idx),
            }
        )
    return pd.DataFrame(rows)


def build_rows(
    authors: list[dict],
    cohort: str,
    lgbtq: int,
    posts: pd.DataFrame,
    max_posts: int,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    """Emit author-table + prompt-table row dicts for one cohort."""
    author_rows, prompt_rows = [], []
    for a in sorted(authors, key=lambda r: r["blog_id"]):
        author_id = f"blog_{a['blog_id']}"
        gender_token = "woman" if a["gender"] == "female" else "man"
        author_rows.append(
            {
                "author_id": author_id,
                "cohort": cohort,
                "transgender": "",
                "gender": [gender_token],
                "orientation": [],
                "pronouns": [],
                "race": [],
                "age": age_bracket(a["age"]),
                "disability": "",
                "education": "",
                "income": "",
            }
        )
        post_idx = a["post_rows"].copy()
        rng.shuffle(post_idx)
        for k, row_pos in enumerate(post_idx[:max_posts]):
            prompt_rows.append(
                {
                    "prompt_id": f"{DATASET_NAME}_{cohort}_{author_id}_{k}",
                    "author_id": author_id,
                    "cohort": cohort,
                    "text": posts.iloc[int(row_pos)]["segment"],
                    "lgbtq": lgbtq,
                    **PROMPT_DEFAULTS,
                }
            )
    return author_rows, prompt_rows


def build_manifest() -> DatasetManifest:
    style_note = (
        "x-unit = one POST SEGMENT (first <=120 words of a post). Published "
        "80.1% (Schler et al. 2006) is per-author over full posts, so expect "
        "lower per-prompt separability; per-author aggregation is the fair "
        "comparison. Attribution of any signal to TOPIC (industry/age themes, "
        "e.g. teens writing about school) vs STYLE is itself the finding — read "
        "the topical JSD alongside lexical/syntactic."
    )
    return DatasetManifest(
        name=DATASET_NAME,
        description=(
            "Blog Authorship Corpus (Schler et al. 2006), gender contrast. "
            "target = female bloggers (y=1), baseline = male bloggers (y=0); "
            "x = a <=120-word segment of one blog post. Non-commercial research "
            "use only."
        ),
        cohorts=[
            CohortSpec(
                name="target",
                group="target",
                display_name="Female bloggers (y=1)",
                description="Female-authored blog posts (self-reported gender).",
            ),
            CohortSpec(
                name="baseline",
                group="baseline",
                display_name="Male bloggers (y=0)",
                description="Male-authored blog posts (self-reported gender).",
            ),
            CohortSpec(
                name="null_split_a",
                group="target",
                display_name="Female bloggers, random split A (null control)",
                description=(
                    "Seeded random split of held-out female authors (disjoint "
                    "from target); same distribution as split B."
                ),
            ),
            CohortSpec(
                name="null_split_b",
                group="target",
                display_name="Female bloggers, random split B (null control)",
                description=(
                    "Seeded random split of held-out female authors (disjoint "
                    "from target); same distribution as split A."
                ),
            ),
        ],
        comparisons=[
            ComparisonSpec(
                name="target_vs_baseline",
                target_cohort="target",
                baseline_cohort="baseline",
                expectation="distinguishable",
                explorations=True,
                expected_accuracy=0.801,
                notes=style_note,
            ),
            ComparisonSpec(
                name="null_control",
                target_cohort="null_split_a",
                baseline_cohort="null_split_b",
                expectation="null",
                explorations=False,
                expected_accuracy=0.0,
                notes=(
                    "Negative control: a seeded random author split within the "
                    "female group. Expect C2ST ~= 0.5, no BH survivors, null "
                    "MMD. Any strong signal here indicates a pipeline defect."
                ),
            ),
        ],
    )


def write_readme(out_dir: Path, counts: dict, seed: int, args) -> None:
    lines = [
        "# Blog Authorship Corpus (gender contrast)",
        "",
        "Converted by `scripts/convert_blog_authorship.py` to the paper's",
        "`D=(x,y,z,d,c)` Parquet format. **Not** synthetic — real blog posts.",
        "",
        "## Provenance",
        "",
        f"- Source: HF `{HF_REPO}` (`{HF_FILE}`, 681,284 posts / 19,320 bloggers).",
        "  Original corpus: Schler, Koppel, Argamon & Pennebaker, *Effects of Age",
        "  and Gender on Blogging* (AAAI Spring Symposium 2006).",
        "- License: original corpus is **non-commercial research use only**",
        "  (Schler et al.); the HF mirror is tagged apache-2.0. Treat as",
        "  non-commercial research data.",
        "",
        "## The contrast (y)",
        "",
        "- `target` = female bloggers (y=1, `lgbtq=1`); `baseline` = male",
        "  bloggers (y=0). `lgbtq` here is the generic target flag, not sexual",
        "  orientation (the corpus has no orientation field).",
        "- Published 2-way gender accuracy **80.1%** (Schler et al. 2006,",
        "  Multi-Class Real Winnow, style+content) -> `expected_accuracy=0.801`.",
        "",
        "## x-unit: a POST SEGMENT (reported per validation design)",
        "",
        "- Each post is cleaned (the corpus's `urlLink` placeholder tokens are",
        "  stripped and runs of whitespace collapsed) and TRUNCATED to its first",
        f"  **<= {MAX_SEGMENT_WORDS} words** (whitespace-tokenized) = one segment,",
        "  one segment per post. Posts with < "
        f"{MIN_POST_WORDS} clean words are dropped.",
        "- This is much shorter than the full posts / per-author feature vectors",
        "  behind the 80.1%; per-prompt separability will run lower. Aggregate to",
        "  the author (PromptSet groups by author) for the fair comparison.",
        "- **Topic vs style caveat:** the corpus's `topic` field (blogger",
        "  industry, e.g. Student/Technology) and any survey `c`-context are NOT",
        "  stored (`domain=''`, `topic_id=0`, all ordinals 0) because they do not",
        "  map onto the paper's MH/GSH/REL survey catalog. Topical analysis is",
        "  therefore derived from the text itself; attributing distinguishability",
        "  to topic (teens write about school) vs style is part of the finding.",
        "",
        "## Sampling (seeded, author structure preserved)",
        "",
        f"- Seed: {seed}. Authors kept only if they have >= "
        f"{MIN_POSTS_PER_AUTHOR} clean posts; up to **{args.max_posts} posts per",
        "  author** are sampled (seeded shuffle).",
        f"- `target`: {args.n_authors} female authors; `baseline`:",
        f"  {args.n_authors} male authors; balanced across the corpus's three",
        "  age bands (13-17 / 23-27 / 33-48) so age is not a gender confound.",
        f"- Negative control: {args.n_null_half} + {args.n_null_half} held-out",
        "  female authors (disjoint from `target`), randomly split into",
        "  `null_split_a` / `null_split_b`.",
        "",
        "## Author (z, d) fields",
        "",
        '- `gender`: female->["woman"], male->["man"] (self-reported binary).',
        "- `age`: numeric age mapped to a canonical bracket (13-17 kept as-is;",
        "  else 18-24 / 25-34 / 35-44 / 45-54). The corpus only contains ages",
        "  13-17, 23-27, 33-48 (Schler's design). **Slice caveat:** the",
        "  under35/35plus age slice matches bracket *strings*, so 13-17 authors",
        "  fall on the 35plus side -- rely on the balanced sampling, not that",
        "  slice, for age control.",
        "- `transgender`, `orientation`, `pronouns`, `race`, `disability`,",
        '  `education`, `income`: unrecorded (""/[]); the corpus lacks them.',
        "",
        "## Cohort counts (this build)",
        "",
        "| cohort | authors | prompts |",
        "|---|---|---|",
    ]
    for cohort in ("target", "baseline", "null_split_a", "null_split_b"):
        a, p = counts[cohort]
        lines.append(f"| {cohort} | {a} | {p} |")
    lines += [
        "",
        "## Comparisons",
        "",
        "- `target_vs_baseline`: female vs male, expectation *distinguishable*,",
        "  `expected_accuracy=0.801`.",
        "- `null_control`: random female author split, expectation *null*.",
        "",
        "Load with `PromptDataset.load(Path('data/blog_authorship'))`.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-authors", type=int, default=80, dest="n_authors")
    parser.add_argument("--n-null-half", type=int, default=60, dest="n_null_half")
    parser.add_argument("--max-posts", type=int, default=4, dest="max_posts")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "data" / DATASET_NAME
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    csv_path = download_raw()
    print(f"Loading + cleaning {csv_path} ...")
    posts = load_clean_posts(csv_path)
    authors = author_table(posts)
    authors["_used"] = False
    print(f"{len(posts)} clean posts, {len(authors)} eligible authors.")

    female = authors[authors["gender"] == "female"]
    male = authors[authors["gender"] == "male"]

    # Draw target (female) first, then baseline (male), then the null pool
    # (held-out female) from the same shared `authors` frame so all cohorts are
    # disjoint by construction (the `_used` flag prevents re-selection).
    target = sample_authors_by_gender(authors, female, even_split(args.n_authors), rng)
    baseline = sample_authors_by_gender(authors, male, even_split(args.n_authors), rng)
    null_pool = sample_authors_by_gender(
        authors, female, even_split(2 * args.n_null_half), rng
    )
    rng.shuffle(null_pool)
    split_a = null_pool[: args.n_null_half]
    split_b = null_pool[args.n_null_half : 2 * args.n_null_half]

    cohort_authors = {
        "target": (target, 1),
        "baseline": (baseline, 0),
        "null_split_a": (split_a, 1),
        "null_split_b": (split_b, 0),
    }
    all_author_rows, all_prompt_rows, counts = [], [], {}
    for cohort, (auth_list, lgbtq) in cohort_authors.items():
        a_rows, p_rows = build_rows(
            auth_list, cohort, lgbtq, posts, args.max_posts, rng
        )
        all_author_rows.extend(a_rows)
        all_prompt_rows.extend(p_rows)
        counts[cohort] = (len(a_rows), len(p_rows))

    out_dir = ensure_dir(args.out_dir)
    manifest = build_manifest()
    save_json(manifest.to_dict(), out_dir / "dataset.json")
    pd.DataFrame(all_author_rows).to_parquet(out_dir / "authors.parquet", index=False)
    pd.DataFrame(all_prompt_rows).to_parquet(out_dir / "prompts.parquet", index=False)
    write_readme(out_dir, counts, args.seed, args)

    dataset = PromptDataset.load(out_dir)
    print(f"PromptDataset.load OK: {out_dir}")
    for cohort in ("target", "baseline", "null_split_a", "null_split_b"):
        a, p = counts[cohort]
        print(f"  {cohort:14s} authors={a:3d} prompts={p:3d}")
    print("\nSample rows:")
    for row in dataset.prompts.head(3).to_dict("records"):
        seg = row["text"][:90].replace("\n", " ")
        print(f"  [{row['cohort']}] {row['prompt_id']} y={row['lgbtq']} :: {seg}...")


def sample_authors_by_gender(
    pool: pd.DataFrame,
    gender_pool: pd.DataFrame,
    quota: list[int],
    rng: np.random.Generator,
) -> list[dict]:
    """Sample authors of one gender, balanced across age groups, from `pool`."""
    chosen = []
    for g, n in zip(AGE_GROUPS, quota, strict=True):
        candidates = pool[
            pool.index.isin(gender_pool.index) & (pool["group"] == g) & (~pool["_used"])
        ]
        take = min(n, len(candidates))
        picks = rng.choice(candidates.index.to_numpy(), size=take, replace=False)
        pool.loc[picks, "_used"] = True
        chosen.extend(pool.loc[picks].to_dict("records"))
    return chosen


if __name__ == "__main__":
    main()
