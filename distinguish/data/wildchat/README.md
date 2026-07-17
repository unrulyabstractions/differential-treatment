# WildChat country-contrast dataset (`wildchat`)

**Provenance.** First user turns of [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)
(Zhao et al., ICLR 2024), the public 837,989-conversation release (toxic and
journalist-flagged-PII conversations were removed by AI2 relative to the paper's
1M). Only shard `data/train-00000-of-00014.parquet` (59,857 conversations) is read; it is downloaded to a
scratch directory, never stored in this repo. **License: ODC-BY** (attribution
required). Regenerate with:
`uv run python scripts/convert_wildchat.py --seed 0 --n-authors 80`

**Task.** y = first-turn IP-geolocation country: target **United Kingdom** (y=1)
vs baseline **United States** (y=0). This is a PROXY label (location, not
self-reported identity or dialect) with **no published separability benchmark**
(`expected_accuracy` 0.0, expectation left exploratory). Validation-design role:
modality match (real chatbot prompts) + scale/asymptotics.

**Sampling (seed 0).** English-labeled conversations whose first turn is
non-redacted, non-toxic, has a country + hashed IP, and >= 8 words;
exact-duplicate texts dropped after truncation. **x = the first user turn,
truncated to its first 120 words** (24% of kept prompts
were truncated). 80 authors/side for target and baseline, plus
2x80 extra United States authors (disjoint from baseline)
randomly halved into `null_split_a`/`null_split_b` — the negative-control
comparison (expectation "null": C2ST ~= 0.5, zero BH survivors, null MMD).
<= 4 prompts per author, seeded picks; author structure preserved.

| cohort | prompts | authors |
|---|---|---|
| baseline | 145 | 80 |
| null_split_a | 152 | 80 |
| null_split_b | 153 | 80 |
| target | 162 | 80 |

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
United States and ~170 United Kingdom eligible authors; add shards
for more). Reruns overwrite `data/wildchat/`.
