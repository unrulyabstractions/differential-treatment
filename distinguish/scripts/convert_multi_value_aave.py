"""Convert a Multi-VALUE AAVE sensitivity dial into our D=(x,y,z,d,c) format.

DATASET 6 (validation design: "Sensitivity calibration"). This is NOT a scraped
corpus: it is a *transform* applied to an existing neutral corpus, so the label y
is exact by construction (same authors, same content — the only difference between
the two sides is whether AAVE morphosyntax was applied).

Source corpus
-------------
The Standard-American-English (SAE) substrate is pooled from ALL cohorts of
``data/synthetic`` (``target`` + ``baseline`` + ``target_twin``, read via
``PromptDataset``): 36 authors x 144 short prompts (35-88 words each). Every one
is SAE-grammar English — the synthetic cohorts differ in topical/register coding,
NOT in grammar — so transforming any of them to AAVE is still an exact-label dial
(the only difference between the two sides is whether AAVE morphosyntax was
applied; the shared content cancels in every value_pXX_vs_baseline comparison).
Pooling all three cohorts triples the per-cohort sample (~144 vs the former 48),
powering the sensitivity curve. ``--n-source`` caps the pool (default = all 144).

The dial
--------
We apply the Multi-VALUE ``Dialects.AfricanAmericanVernacular()`` transform
(SALT-NLP/multi-value, Apache-2.0; Ziems et al. ACL 2022 / arXiv 2212.08011) at
five rule-application densities p in {0.05, 0.10, 0.25, 0.50, 1.00}.

Density mechanism (documented, seeded, NESTED across p)
    1. Split each SAE prompt into sentence units (regex; terminal .!? kept).
    2. Transform each sentence once with the AAVE transform. A per-sentence
       ``dialect.set_seed`` (derived from --seed + prompt_id + sentence index)
       makes the otherwise-stochastic transform reproducible. A sentence is
       "changed" iff its transform differs from the original.
    3. Draw one seeded uniform u in [0,1) per sentence (independent of the
       transform seed and of p). In cohort ``value_pXX`` a changed sentence is
       KEPT transformed iff u < p, else it is reverted to its SAE original.
       Because u does not depend on p, the transformed-sentence sets are nested:
       p=0.05 changes  subset of  p=0.10 changes  subset of  ...  subset of  p=1.00 (full AAVE).
    This realizes "reverting a (1-p) fraction of changed sentences" (the recipe
    named in the plan) as a smooth, monotone, content-preserving dial. We chose it
    over DialectFromFeatureList (also available) because feature-subset dials are
    lumpy — features fire at wildly different rates — whereas the sentence-reversion
    dial gives an even, interpretable p-axis and keeps content identical to baseline.

Per-prompt annotations (implicit y breakdown)
    markedness = 1 iff any AAVE rule survives in the density-applied text, else 0.
    codedness  = (# AAVE rules that fired in surviving sentences) / (# words),
                 i.e. realized rule-application density per word, clamped to [0,1].
    Baseline and null-split prompts are pure SAE: markedness 0, codedness 0.

Cohorts / comparisons
    baseline (y=0), value_p05..value_p100 (y=1); one comparison per p vs baseline
    (expectation "" for low p, "distinguishable" for high p) forming the sensitivity
    curve. Plus null_split_a vs null_split_b: a seeded random split of the baseline
    authors, both pure SAE -> negative control (expectation "null").

Author z/d fields are inherited verbatim from the synthetic baseline authors (real
gender/race/age/education/income annotations of that fixture). They are metadata,
NOT the y variable: y here is "AAVE-transformed?", independent of author identity.

Idempotent CLI. Raw/intermediate artifacts go under the scratchpad
raw_datasets/multi_value_aave/ (NEVER into the repo); the expensive transform pass
is cached there and reused unless --force or --seed changes.
"""

from __future__ import annotations

import argparse
import functools
import io
import json
import re
import zlib
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.common.dataset_tables import (
    CohortSpec,
    ComparisonSpec,
    DatasetManifest,
    PromptDataset,
)

# Stanza checkpoints shipped with Multi-VALUE predate torch 2.6's weights_only=True
# default; loading them requires weights_only=False. These are trusted model files
# from the stanza release, downloaded by Multi-VALUE itself.
_ORIG_TORCH_LOAD = torch.load
torch.load = functools.wraps(_ORIG_TORCH_LOAD)(
    lambda *a, **k: _ORIG_TORCH_LOAD(*a, **{**k, "weights_only": False})
)

DATASET_NAME = "multi_value_aave"
SOURCE_DATASET = "data/synthetic"
# SAE substrate = ALL synthetic cohorts pooled (target+baseline+twin); every
# prompt is SAE-grammar (the cohorts differ in coding, not grammar), so
# transforming any of them is still an exact-label dial. --n-source caps the
# pool (default = the full 144 prompts).
SOURCE_COHORTS = ("target", "baseline", "target_twin")
N_SOURCE_ALL = 144
PROVENANCE = "multi_value_aave"

SCRATCH_RAW = (
    Path(
        "/private/tmp/claude-501/-Users-unrulyabstractions-work-prompt-distinguishability"
        "/140b8c69-9c33-43d8-8683-a04e90d30bcf/scratchpad/raw_datasets"
    )
    / DATASET_NAME
)

# Rule-application densities -> target cohort names.
DENSITIES: dict[str, float] = {
    "value_p05": 0.05,
    "value_p10": 0.10,
    "value_p25": 0.25,
    "value_p50": 0.50,
    "value_p100": 1.00,
}
# Prior expectation for each density's comparison vs baseline (sensitivity curve).
EXPECTATION: dict[str, str] = {
    "value_p05": "",
    "value_p10": "",
    "value_p25": "",
    "value_p50": "distinguishable",
    "value_p100": "distinguishable",
}

_SENT_RE = re.compile(r"(.*?[.!?]+)(\s+|$)", re.DOTALL)

PROMPT_DEFAULTS = {
    "topic_id": 0,
    "domain": "",
    "provenance": PROVENANCE,
    "adoption": 0,
    "general_freq": 0,
    "llm_freq": 0,
    "professional_freq": 0,
    "aversion": 0,
    "satisfaction": 0,
}


def split_sentences(text: str) -> list[tuple[str, str]]:
    """Split into (core, trailing_whitespace) pairs; concatenation restores text."""
    parts: list[tuple[str, str]] = []
    pos = 0
    for m in _SENT_RE.finditer(text):
        if not m.group(1).strip():
            continue
        parts.append((m.group(1), m.group(2)))
        pos = m.end()
    if pos < len(text) and text[pos:].strip():
        parts.append((text[pos:], ""))
    if not parts:  # no terminal punctuation at all
        parts.append((text, ""))
    return parts


def _crc(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def sentence_uniform(seed: int, prompt_id: str, idx: int) -> float:
    """Seeded uniform in [0,1); independent of p so cohorts nest monotonically."""
    rng = np.random.default_rng([seed, _crc(f"{prompt_id}:{idx}")])
    return float(rng.random())


def transform_seed(seed: int, prompt_id: str, idx: int) -> int:
    return (seed * 1_000_003 + _crc(f"{prompt_id}:{idx}:transform")) & 0x7FFFFFFF


def load_dialect():
    """Instantiate the AAVE transform, suppressing its noisy model-load banners."""
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        from multivalue import Dialects

        return Dialects.AfricanAmericanVernacular()


def build_transform_cache(
    prompts: pd.DataFrame, seed: int, force: bool
) -> pd.DataFrame:
    """Transform every source sentence once; cache to the scratch raw dir.

    Returns a frame with one row per (prompt_id, sent_idx):
    core_orig, sep, core_aave, n_rules, changed.
    """
    SCRATCH_RAW.mkdir(parents=True, exist_ok=True)
    cache_path = SCRATCH_RAW / "transformed_sentences.parquet"
    meta_path = SCRATCH_RAW / "meta.json"
    source_ids = sorted(prompts["prompt_id"])
    if cache_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        cached = pd.read_parquet(cache_path)
        if meta.get("seed") == seed and meta.get("source_prompt_ids") == source_ids:
            print(f"Reusing cached transforms: {cache_path} ({len(cached)} sentences)")
            return cached

    dialect = load_dialect()
    rows: list[dict] = []
    buf = io.StringIO()
    for rec in prompts.itertuples(index=False):
        for idx, (core, sep) in enumerate(split_sentences(rec.text)):
            dialect.set_seed(transform_seed(seed, rec.prompt_id, idx))
            try:
                with redirect_stdout(buf), redirect_stderr(buf):
                    aave = dialect.transform(core)
                n_rules = len(dialect.executed_rules)
            except Exception:  # transform is best-effort; treat failure as no-op
                aave, n_rules = core, 0
            changed = aave.strip() != core.strip()
            rows.append(
                {
                    "prompt_id": rec.prompt_id,
                    "author_id": rec.author_id,
                    "sent_idx": idx,
                    "core_orig": core,
                    "sep": sep,
                    "core_aave": aave,
                    "n_rules": int(n_rules),
                    "changed": bool(changed),
                }
            )
    cache = pd.DataFrame(rows)
    cache.to_parquet(cache_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "source": f"{SOURCE_DATASET}:{'+'.join(SOURCE_COHORTS)}",
                "source_prompt_ids": source_ids,
                "n_source": len(source_ids),
                "n_sentences": len(cache),
            },
            indent=2,
        )
    )
    print(f"Wrote transform cache: {cache_path} ({len(cache)} sentences)")
    return cache


def render_cohort_text(sents: pd.DataFrame, seed: int, p: float) -> tuple[str, int]:
    """Reassemble one prompt at density p. Returns (text, n_rules_surviving)."""
    text_parts: list[str] = []
    n_rules = 0
    for row in sents.itertuples(index=False):
        keep = row.changed and sentence_uniform(seed, row.prompt_id, row.sent_idx) < p
        text_parts.append((row.core_aave if keep else row.core_orig) + row.sep)
        if keep:
            n_rules += row.n_rules
    return "".join(text_parts).strip(), n_rules


def make_prompt_row(
    cohort: str,
    prompt_id: str,
    author_id: str,
    text: str,
    lgbtq: int,
    markedness: int,
    codedness: float,
) -> dict:
    return {
        "prompt_id": f"{DATASET_NAME}_{cohort}_{prompt_id}",
        "author_id": author_id,
        "cohort": cohort,
        "text": text,
        "lgbtq": lgbtq,
        "markedness": markedness,
        "codedness": codedness,
        **PROMPT_DEFAULTS,
    }


def author_row(src: dict, cohort: str) -> dict:
    def as_list(v) -> list[str]:
        return [str(x) for x in v] if v is not None else []

    return {
        "author_id": src["author_id"],
        "cohort": cohort,
        "transgender": str(src["transgender"]),
        "gender": as_list(src["gender"]),
        "orientation": as_list(src["orientation"]),
        "pronouns": as_list(src["pronouns"]),
        "race": as_list(src["race"]),
        "age": str(src["age"]),
        "disability": str(src["disability"]),
        "education": str(src["education"]),
        "income": str(src["income"]),
    }


def build_manifest() -> DatasetManifest:
    cohorts = [
        CohortSpec(
            name="baseline",
            group="baseline",
            display_name="Original SAE prompts (y=0)",
            description=(
                "Standard-American-English source prompts (the baseline cohort of "
                "data/synthetic): plain, direct register, no dialect transform."
            ),
        )
    ]
    comparisons: list[ComparisonSpec] = []
    for cohort, p in DENSITIES.items():
        pct = round(p * 100)
        cohorts.append(
            CohortSpec(
                name=cohort,
                group="target",
                display_name=f"AAVE-transformed p={p:g} (y=1)",
                description=(
                    f"Same authors/content as baseline, with Multi-VALUE AAVE "
                    f"morphosyntax applied at rule-application density p={p:g} "
                    f"({pct}% of changed sentences kept transformed, seeded/nested)."
                ),
            )
        )
        comparisons.append(
            ComparisonSpec(
                name=f"{cohort}_vs_baseline",
                target_cohort=cohort,
                baseline_cohort="baseline",
                expectation=EXPECTATION[cohort],
                explorations=(cohort == "value_p100"),
                expected_accuracy=0.0,
                notes=(
                    f"sensitivity calibration, exact y; AAVE rule-application "
                    f"density p={p:g}. Tunable signal (no fixed published target)."
                ),
            )
        )
    for split in ("null_split_a", "null_split_b"):
        cohorts.append(
            CohortSpec(
                name=split,
                group="baseline",
                display_name=f"Baseline null split {split[-1].upper()}",
                description=(
                    "Seeded random half of the baseline authors, pure SAE "
                    "(no transform) — negative-control partner."
                ),
            )
        )
    comparisons.append(
        ComparisonSpec(
            name="null_split_a_vs_b",
            target_cohort="null_split_a",
            baseline_cohort="null_split_b",
            expectation="null",
            explorations=False,
            expected_accuracy=0.0,
            notes=(
                "negative control: random author-split of the untransformed "
                "baseline; expect C2ST~0.5, 0 BH survivors, null MMD."
            ),
        )
    )
    return DatasetManifest(
        name=DATASET_NAME,
        description=(
            "Multi-VALUE AAVE sensitivity dial: a neutral SAE corpus (data/synthetic "
            "baseline) transformed into African-American Vernacular English at five "
            "rule-application densities p in {0.05,0.10,0.25,0.50,1.00}. Exact y by "
            "construction (identical authors/content; only dialect differs). One "
            "comparison per p forms a sensitivity curve; a null author-split of the "
            "untransformed baseline is the negative control. Validates the pipeline's "
            "sensitivity to graded stylistic signal, not realism."
        ),
        cohorts=cohorts,
        comparisons=comparisons,
    )


def load_source(n_source: int | None, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool ALL synthetic cohorts as the SAE substrate; optionally subsample.

    prompt_ids and author_ids are disjoint across the synthetic cohorts, so
    pooling introduces no collisions. When ``n_source`` is set below the pool
    size, a seeded random sample of prompts is drawn and the authors table is
    restricted to the authors those prompts reference (so validation holds).
    """
    source = PromptDataset.load(Path(SOURCE_DATASET))
    src_prompts = (
        source.prompts[source.prompts["cohort"].isin(SOURCE_COHORTS)]
        .sort_values("prompt_id")
        .reset_index(drop=True)
    )
    if n_source is not None and n_source < len(src_prompts):
        rng = np.random.default_rng([seed, _crc("n_source")])
        picked = np.sort(rng.choice(len(src_prompts), size=n_source, replace=False))
        src_prompts = src_prompts.iloc[picked].reset_index(drop=True)
    used_authors = set(src_prompts["author_id"])
    src_authors = (
        source.authors[source.authors["author_id"].isin(used_authors)]
        .drop_duplicates("author_id")
        .reset_index(drop=True)
    )
    return src_prompts, src_authors


def build(
    seed: int, force: bool, n_source: int | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    src_prompts, src_authors = load_source(n_source, seed)

    cache = build_transform_cache(src_prompts, seed, force)

    prompt_rows: list[dict] = []
    author_rows: list[dict] = []

    # baseline cohort: original SAE text, pure.
    for rec in src_prompts.itertuples(index=False):
        prompt_rows.append(
            make_prompt_row(
                "baseline",
                rec.prompt_id,
                rec.author_id,
                rec.text,
                lgbtq=0,
                markedness=0,
                codedness=0.0,
            )
        )
    for a in src_authors.to_dict("records"):
        author_rows.append(author_row(a, "baseline"))

    # value_pXX cohorts: density-applied AAVE.
    for cohort, p in DENSITIES.items():
        for rec in src_prompts.itertuples(index=False):
            sents = cache[cache["prompt_id"] == rec.prompt_id]
            text, n_rules = render_cohort_text(sents, seed, p)
            n_words = max(len(text.split()), 1)
            codedness = round(min(1.0, n_rules / n_words), 4)
            markedness = int(n_rules > 0)
            prompt_rows.append(
                make_prompt_row(
                    cohort,
                    rec.prompt_id,
                    rec.author_id,
                    text,
                    lgbtq=1,
                    markedness=markedness,
                    codedness=codedness,
                )
            )
        for a in src_authors.to_dict("records"):
            author_rows.append(author_row(a, cohort))

    # null control: seeded split of the baseline authors, untransformed.
    author_ids = sorted(src_authors["author_id"])
    rng = np.random.default_rng([seed, _crc("null_split")])
    order = rng.permutation(len(author_ids))
    half = len(author_ids) // 2
    split_map = {
        "null_split_a": {author_ids[i] for i in order[:half]},
        "null_split_b": {author_ids[i] for i in order[half:]},
    }
    for split, ids in split_map.items():
        for rec in src_prompts.itertuples(index=False):
            if rec.author_id not in ids:
                continue
            prompt_rows.append(
                make_prompt_row(
                    split,
                    rec.prompt_id,
                    rec.author_id,
                    rec.text,
                    lgbtq=0,
                    markedness=0,
                    codedness=0.0,
                )
            )
        for a in src_authors.to_dict("records"):
            if a["author_id"] in ids:
                author_rows.append(author_row(a, split))

    return pd.DataFrame(prompt_rows), pd.DataFrame(author_rows)


def write_readme(out_dir: Path, seed: int, prompts: pd.DataFrame) -> None:
    counts = prompts.groupby("cohort").size().to_dict()
    n_words = prompts["text"].str.split().str.len()
    n_baseline = int((prompts["cohort"] == "baseline").sum())
    n_src_authors = int(prompts[prompts.cohort == "baseline"].author_id.nunique())
    lines = [
        "# multi_value_aave",
        "",
        "**Multi-VALUE African-American Vernacular English (AAVE) sensitivity dial.**",
        "A *transform*-based dataset (not a scraped corpus): the label y is exact by",
        "construction — the two sides share identical authors and content, differing",
        "only in whether AAVE morphosyntax was applied.",
        "",
        "## Provenance",
        "- **Transform**: `Dialects.AfricanAmericanVernacular()` from Multi-VALUE",
        "  (SALT-NLP/multi-value, `pip install value-nlp`, **Apache-2.0**).",
        "  Ziems et al., *VALUE* (ACL 2022, arXiv:2204.03031) and *Multi-VALUE*",
        "  (arXiv:2212.08011). Rules validated by dialect-speaker judgments upstream.",
        "- **Source corpus (SAE substrate)**: ALL cohorts of `data/synthetic`",
        "  (`target` + `baseline` + `target_twin`, read via `PromptDataset`) pooled",
        f"  — {n_src_authors} authors x {n_baseline} short SAE prompts. Every source",
        "  prompt is SAE-grammar English (the synthetic cohorts differ in",
        "  topical/register coding, NOT in grammar), so transforming any of them is",
        "  still an exact-label dial; the shared content cancels in every",
        "  `value_pXX_vs_baseline` comparison. Pooling all three cohorts triples the",
        "  per-cohort sample (was 48 from baseline alone), powering the sensitivity",
        "  curve. This is the project's synthetic fixture, not real user data;",
        "  author demographics (gender/race/age/...) are inherited verbatim as z/d",
        "  metadata and are NOT the y variable.",
        "",
        "## License / ethics",
        "- Multi-VALUE code+rules: Apache-2.0. Generated AAVE text is a rule-based",
        "  rendering, not real speakers' writing; do not treat it as authentic AAVE",
        "  data or as evidence about detecting self-reported dialect/identity. This",
        "  dataset only calibrates the pipeline's sensitivity to graded stylistic",
        "  signal (realism is explicitly out of scope).",
        "",
        "## The density dial (exact mechanism)",
        f"Seeded with `--seed {seed}`. Each SAE prompt is split into sentence units;",
        "each sentence is transformed once (a per-sentence `dialect.set_seed` makes",
        "the otherwise-stochastic transform reproducible). A sentence is *changed* iff",
        "its transform differs from the original. One seeded uniform u in [0,1) is",
        "drawn per sentence, **independent of p**; in cohort `value_pXX` a changed",
        "sentence is kept transformed iff u < p, else reverted to SAE. Because u does",
        "not depend on p, the transformed sets are **nested**:",
        "`p=0.05 subset p=0.10 subset p=0.25 subset p=0.50 subset p=1.00` (full AAVE).",
        'This realizes the plan\'s "revert a (1-p) fraction of changed sentences" as a',
        "smooth, monotone, content-preserving dial. (DialectFromFeatureList is also",
        "available but gives a lumpy dial — features fire at very different rates — so",
        "sentence-reversion was chosen for an even p-axis.)",
        "",
        "## Per-prompt annotations",
        "- `lgbtq` (generic y flag): 1 for every `value_pXX` prompt, 0 for baseline",
        "  and null splits.",
        "- `markedness`: 1 iff any AAVE rule survives in the density-applied text.",
        "- `codedness`: (# AAVE rules fired in surviving sentences) / (# words),",
        "  i.e. realized rule-application density per word, clamped to [0,1].",
        "- z/d/c beyond the inherited author demographics are left at defaults",
        '  (0 / "") — unrecorded; usage/topical sections skip gracefully.',
        "",
        "## Cohorts",
        "- `baseline` (y=0): original SAE prompts.",
        "- `value_p05/p10/p25/p50/p100` (y=1): AAVE at density p=0.05/.../1.00.",
        "- `null_split_a`, `null_split_b`: seeded random halves of the baseline",
        "  authors, both pure SAE (negative control).",
        "",
        "## Comparisons",
        "- `value_pXX_vs_baseline` (one per p): sensitivity curve. Prior expectation",
        '  "" for low p (0.05/0.10/0.25), "distinguishable" for high p (0.50/1.00).',
        "  No fixed published accuracy target — signal is tunable by design.",
        '- `null_split_a_vs_b`: negative control, expectation "null"',
        "  (expect C2ST~0.5, 0 BH survivors, null MMD).",
        "",
        "## Sampling / sizes",
        f"- Author structure preserved: the same {n_src_authors} source authors"
        " appear in every value cohort (exact-y).",
        f"- Bounded by the pooled source corpus ({n_src_authors} authors x"
        f" {n_baseline} prompts, `--n-source` capped); no realism subsampling was",
        "  applied — realism is out of scope for a calibration dial.",
        f"- Text length: {int(n_words.min())}-{int(n_words.max())} words "
        f"(median {int(n_words.median())}); all <= 120 words, so **no truncation or**",
        "  **segmentation** was needed.",
        "- Per-cohort prompt counts:",
    ]
    for cohort in ["baseline", *DENSITIES, "null_split_a", "null_split_b"]:
        lines.append(f"  - `{cohort}`: {counts.get(cohort, 0)}")
    lines += [
        "",
        "## Caveats",
        "- Rule-based dialect rendering can be ungrammatical or inconsistent; it",
        "  approximates AAVE morphosyntax, not natural production.",
        "- The signal is definitional (we injected it), so high-p separability",
        "  validates sensitivity only, and says nothing about real-world detection",
        "  of self-reported dialect or identity.",
        "- The SAE substrate is a small synthetic fixture; absolute effect sizes are",
        "  not comparable to the real-corpus datasets.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data") / DATASET_NAME)
    parser.add_argument(
        "--force", action="store_true", help="recompute the transform cache"
    )
    parser.add_argument(
        "--n-source",
        type=int,
        default=None,
        help=(
            f"cap the pooled SAE substrate to N prompts (seeded sample); "
            f"default = all {N_SOURCE_ALL} (target+baseline+twin of data/synthetic)"
        ),
    )
    args = parser.parse_args()

    prompts, authors = build(args.seed, args.force, args.n_source)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    (out_dir / "dataset.json").write_text(
        json.dumps(manifest.to_dict(), indent=4) + "\n"
    )
    prompts.to_parquet(out_dir / "prompts.parquet", index=False)
    authors.to_parquet(out_dir / "authors.parquet", index=False)
    write_readme(out_dir, args.seed, prompts)

    dataset = PromptDataset.load(out_dir)
    print(
        f"\nWrote {out_dir}/ (dataset.json, prompts.parquet, authors.parquet, README.md)"
    )
    print("PromptDataset.load: OK")
    print("\nPer-cohort prompt counts:")
    print(dataset.prompts.groupby("cohort").size().to_string())
    print("\nPer-cohort author counts:")
    print(dataset.authors.groupby("cohort").size().to_string())
    print("\n3 sample rows:")
    cols = ["prompt_id", "cohort", "lgbtq", "markedness", "codedness", "text"]
    sample = dataset.prompts[dataset.prompts.cohort == "value_p100"].head(3)
    for r in sample[cols].itertuples(index=False):
        print(
            f"  [{r.cohort} y={r.lgbtq} m={r.markedness} c={r.codedness}] "
            f"{r.text[:110]!r}"
        )


if __name__ == "__main__":
    main()
