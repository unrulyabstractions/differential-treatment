"""Annotate a dataset cohort's markedness/codedness labels with an LLM (paper 3.2).

Usage:
    uv run python scripts/annotate_prompt_set.py \\
        --dataset data/synthetic --cohort target \\
        --out data/synthetic_annotated [--model gpt-5-mini]

Writes a copy of the dataset directory whose prompts.parquet has the cohort's
markedness/codedness columns replaced by LLM estimates (lgbtq untouched), plus
<out>/rationales.json mapping prompt_id -> rationale. Requires OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.annotation.markedness_codedness_annotator import (  # noqa: E402
    annotate_markedness_codedness,
)
from src.common.dataset_tables import PromptDataset  # noqa: E402
from src.common.file_io import save_json  # noqa: E402
from src.common.logging_utils import log  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="dataset directory")
    parser.add_argument("--cohort", required=True, help="cohort to annotate")
    parser.add_argument(
        "--out", type=Path, required=True, help="output dataset directory"
    )
    parser.add_argument("--model", default="gpt-5-mini")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = PromptDataset.load(args.dataset)
    mask = dataset.prompts["cohort"] == args.cohort
    if not mask.any():
        raise SystemExit(f"No prompts in cohort '{args.cohort}'")
    rows = dataset.prompts[mask]
    log(f"Annotating {len(rows)} prompts from cohort '{args.cohort}' with {args.model}")
    estimates = annotate_markedness_codedness(list(rows["text"]), model_name=args.model)

    prompts = dataset.prompts.copy()
    prompts.loc[mask, "markedness"] = [e.markedness for e in estimates]
    prompts.loc[mask, "codedness"] = [e.codedness for e in estimates]

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.dataset / "dataset.json", args.out / "dataset.json")
    shutil.copy(args.dataset / "authors.parquet", args.out / "authors.parquet")
    prompts.to_parquet(args.out / "prompts.parquet", index=False)
    save_json(
        dict(zip(rows["prompt_id"], [e.rationale for e in estimates], strict=True)),
        args.out / "rationales.json",
    )
    PromptDataset.load(args.out)  # re-validate the written copy
    log(f"Annotated dataset written to {args.out}")


if __name__ == "__main__":
    main()
