"""Collect real community fitness prompts from the PullPush Reddit archive.

Builds LOCAL-ONLY prompt files (data/prompts/real_* is gitignored — community
data stewardship: collected user content never leaves this machine).

Curated post ids were research-verified against raw PullPush JSON: public
posts, fitness/nutrition questions, no explicit identity statements needed
(community shows through register/context), no self-identified minors, no
usernames collected. Instruction ids are left empty — stage 1's extraction
mode (`annotate_instructions: "extract"`) infers and matches them.

Usage:
    uv run python scripts/collect_reddit_prompts.py                 # LGBTQ+ pair (curated + sweeps)
    uv run python scripts/collect_reddit_prompts.py --pair data/group_pairs/women_vs_men.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dtreat.common.file_io import load_json, save_json
from dtreat.llm.api_retry import call_with_retry

OUT_DIR = Path(__file__).parent.parent / "data" / "prompts"
PULLPUSH_URL = "https://api.pullpush.io/reddit/search/submission/?ids={post_id}"

# (post_id, subreddit) — curated + verified by prior research pass
TARGET_POST_IDS = [
    ("1knf4jr", "FTMFitness"),
    ("1kogtps", "FTMFitness"),
    ("1kbw07m", "FTMFitness"),
    ("1kkd29v", "FTMFitness"),
    ("1ke2cej", "FTMFitness"),
    ("1koh5va", "askgaybros"),
    ("1kd4cd5", "askgaybros"),
    ("1k8j69e", "askgaybros"),
    ("1jodmy9", "askgaybros"),
    ("1kn2ohf", "butchlesbians"),
    ("1ixbwyd", "butchlesbians"),
    ("1kq05bb", "MtF"),
    ("1kjnu9y", "MtF"),
    ("1klv5p4", "MtF"),
]
BASELINE_POST_IDS = [
    ("1kobouf", "beginnerfitness"),
    ("1klxchi", "beginnerfitness"),
    ("1kha6bp", "beginnerfitness"),
    ("1kfvxgf", "beginnerfitness"),
    ("1fvdxva", "gainit"),
    ("1koeogh", "naturalbodybuilding"),
    ("1khqbn1", "naturalbodybuilding"),
]

# Keyword sweeps per side; every hit passes the same ethics/quality filters.
# Queries stay strictly fitness-scoped; swept posts must clear the fitness
# relevance filter and must not be marked over_18.
SWEEP_QUERIES = ["bulk", "cutting", "protein", "creatine", "plateau", "workout split", "macros", "meal plan"]
TARGET_SWEEP_SUBREDDITS = ["FTMFitness", "askgaybros", "MtF", "butchlesbians", "gaybros"]
BASELINE_SWEEP_SUBREDDITS = ["beginnerfitness", "naturalbodybuilding", "gainit", "bodybuilding"]
SWEEP_URL = (
    "https://api.pullpush.io/reddit/search/submission/"
    "?subreddit={subreddit}&q={query}&size=25"
)
MAX_PROMPTS_PER_SIDE = 60

FITNESS_RELEVANCE_KEYWORDS = (
    "bulk", "cut", "protein", "creatine", "muscle", "gym", "workout", "lift",
    "weight", "plateau", "macro", "meal", "cardio", "calorie", "diet",
)

# Never include content from posters who self-identify as minors, and skip
# removed/deleted or trivially short bodies.
EXCLUSION_MARKERS = ("i'm underage", "im underage", "i am underage", "i'm a minor", "i am a minor", "16m", "16f", "15m", "15f", "17m", "17f")


def fetch_json(url: str) -> dict | None:
    """Fetch with patient backoff; None (not fatal) when retries exhaust —
    a partially collected dataset is still written and usable."""
    def _do_fetch() -> dict:
        request = urllib.request.Request(
            url, headers={"User-Agent": "dtreat-local-research"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        result = call_with_retry(
            _do_fetch, max_retries=8, base_delay_s=20.0, max_delay_s=240.0,
            label="pullpush",
        )
    except Exception as error:
        print(f"  [skip] fetch failed after retries: {type(error).__name__}")
        return None
    time.sleep(3.0)
    return result


def usable_text(post: dict) -> str | None:
    title = (post.get("title") or "").strip()
    body = (post.get("selftext") or "").strip()
    if body.lower() in ("[removed]", "[deleted]"):
        return None
    text = f"{title}\n\n{body}".strip() if body else title
    lowered = text.lower()
    if any(marker in lowered for marker in EXCLUSION_MARKERS):
        return None
    if len(text) < 80:
        return None
    return text


def collect_by_ids(post_ids: list[tuple[str, str]], prefix: str) -> list[dict]:
    prompts = []
    for post_id, subreddit in post_ids:
        data = fetch_json(PULLPUSH_URL.format(post_id=post_id))
        if data is None:
            continue
        posts = data.get("data", [])
        text = usable_text(posts[0]) if posts else None
        if text is None:
            print(f"  [skip] {subreddit}/{post_id}: removed/excluded/missing")
            continue
        prompts.append(
            {
                "prompt_id": f"{prefix}_{post_id}",
                "text": text,
                "instruction_id": "",
                "source_permalink": f"reddit.com/r/{subreddit}/comments/{post_id}/",
            }
        )
        time.sleep(0.5)  # be polite to the archive
    return prompts


def _sweep_usable(post: dict) -> str | None:
    """Sweep hits face stricter filters than curated ids: on-topic and SFW."""
    if post.get("over_18"):
        return None
    text = usable_text(post)
    if text is None:
        return None
    lowered = text.lower()
    if not any(keyword in lowered for keyword in FITNESS_RELEVANCE_KEYWORDS):
        return None
    return text


def sweep_side(existing: list[dict], subreddits: list[str], prefix: str) -> list[dict]:
    """Top up one side with keyword sweeps until MAX_PROMPTS_PER_SIDE."""
    return sweep_side_capped(existing, subreddits, prefix, SWEEP_QUERIES, MAX_PROMPTS_PER_SIDE)


def sweep_side_capped(
    existing: list[dict],
    subreddits: list[str],
    prefix: str,
    queries: list[str],
    cap: int,
) -> list[dict]:
    """Keyword sweeps over subreddits until `cap` prompts collected."""
    seen_ids = {p["prompt_id"].split("_")[-1] for p in existing}
    prompts = list(existing)
    for query in queries:
        for subreddit in subreddits:
            if len(prompts) >= cap:
                return prompts
            data = fetch_json(SWEEP_URL.format(subreddit=subreddit, query=query))
            if data is None:
                continue
            for post in data.get("data", []):
                if len(prompts) >= cap:
                    break
                post_id = post.get("id", "")
                text = _sweep_usable(post)
                if not post_id or post_id in seen_ids or text is None:
                    continue
                seen_ids.add(post_id)
                prompts.append(
                    {
                        "prompt_id": f"{prefix}_{post_id}",
                        "text": text,
                        "instruction_id": "",
                        "source_permalink": f"reddit.com/r/{subreddit}/comments/{post_id}/",
                    }
                )
    return prompts


def collect_pair(spec_path: Path) -> None:
    """Collect both sides of a group pair from a committed pair spec.

    Specs hold only subreddit lists and queries (no user content), so they
    are safe to commit; the collected prompt files stay LOCAL ONLY.
    """
    spec = load_json(spec_path)
    pair = spec["pair_name"]
    cap = int(spec.get("max_prompts_per_side", MAX_PROMPTS_PER_SIDE))
    for side, community in (("target", spec["target_community"]),
                            ("baseline", spec["baseline_community"])):
        subreddits = spec[f"{side}_subreddits"]
        prefix = f"{side[0]}_{pair}"
        print(f"Collecting {pair}/{community} from {subreddits} (cap {cap})...")
        prompts = sweep_side_capped(
            [], subreddits, prefix, spec.get("sweep_queries", SWEEP_QUERIES), cap
        )
        print(f"  {len(prompts)} prompts")
        out_path = OUT_DIR / f"real_{pair}_{community}.json"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        save_json(
            {"community": community, "domain": spec["domain"], "prompts": prompts},
            out_path,
        )
        print(f"wrote {out_path} ({len(prompts)} prompts) — LOCAL ONLY")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", type=Path, default=None,
        help="group-pair spec JSON (default: built-in LGBTQ+ pair with curated ids)",
    )
    args = parser.parse_args()
    if args.pair:
        collect_pair(args.pair)
        return
    print("Collecting target-community prompts (curated ids + sweeps)...")
    target_prompts = collect_by_ids(TARGET_POST_IDS, "t_reddit")
    target_prompts = sweep_side(target_prompts, TARGET_SWEEP_SUBREDDITS, "t_reddit")
    print(f"  {len(target_prompts)} target prompts")
    print("Collecting baseline prompts (curated ids + sweeps)...")
    baseline_prompts = collect_by_ids(BASELINE_POST_IDS, "b_reddit")
    baseline_prompts = sweep_side(baseline_prompts, BASELINE_SWEEP_SUBREDDITS, "b_reddit")
    print(f"  {len(baseline_prompts)} baseline prompts")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(
        {"community": "lgbtq", "domain": "fitness_nutrition_advice", "prompts": target_prompts},
        OUT_DIR / "real_reddit_lgbtq.json",
    )
    save_json(
        {"community": "cishet", "domain": "fitness_nutrition_advice", "prompts": baseline_prompts},
        OUT_DIR / "real_reddit_cishet.json",
    )
    for name in ("real_reddit_lgbtq.json", "real_reddit_cishet.json"):
        reloaded = load_json(OUT_DIR / name)
        print(f"wrote {OUT_DIR / name} ({len(reloaded['prompts'])} prompts) — LOCAL ONLY")


if __name__ == "__main__":
    main()
