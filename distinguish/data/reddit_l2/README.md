# Reddit-L2 (native vs non-native English)

Converted for the prompt-distinguishability pipeline (`D=(x,y,z,d,c)` tables).

## Provenance

- **Corpus**: Reddit-L2 (Rabinovich, Nisioi, Ordan & Wintner, *TACL* 2018; Goldin, Rabinovich & Wintner, *EMNLP* 2018, D18-1395). English Reddit comments; author first language (L1) is inferred from the country flair of European subreddits (r/europe, r/AskEurope).
- **Access**: the corpus is distributed via public Google Drive links on the now-archived Haifa CL project page (https://cl.haifa.ac.il/projects/L2/). The official channel asks for a courtesy email to the authors (ellarabi@gmail.com) — please do so for any substantial use. This converter downloads the public `reddit_full_posts_data.zip` (Drive id `18iJuQJi_rIrarZjXJF-gdVR8ZddH-r7f`) and extracts only four members (`europe_data/reddit.{Germany,UK,US}.tok.clean.csv` + `readMe.txt`) via HTTP range requests — no full 5.3 GB download and no `gdown` dependency. Raw files land in a scratch dir OUTSIDE the repo.
- **License / terms**: research use; cite the two papers above. Raw Reddit text is user-generated content under Reddit's terms; only derived, truncated snippets are stored here.

## What each row is

- **x (text unit)**: a *chunk* of one author's consecutive comments, greedily packed to at most **120 words** (min **20 words**). A single comment longer than 120 words is TRUNCATED to its first 120 words. Automod/removal boilerplate and exact within-author duplicate comments are dropped before packing. The source `europe_data` comments are pre-tokenized (space-separated); we additionally normalize escaped newlines, mangled `\uXXXX` escapes, and spaced HTML entities (`& gt ;` -> `>` etc.).
- **y (`lgbtq` flag = generic target flag)**: 1 for German-L1 cohorts, 0 for the native baseline.
- **z / d / c**: Reddit-L2 carries no self-reported identity, demographics, or interaction context, so all author identity/demographic fields and all prompt context fields are left at their unrecorded defaults (`""`, `[]`, `0`, `0.0`). The only known signal — L1 — is encoded by the cohort, not by a per-row column (topic_id stays 0 so the topical module treats it as unrecorded rather than a survey topic).

## Cohorts

| cohort | group | y | source flair | n_authors | n_prompts |
|---|---|---|---|---|---|
| target | target | 1 | Germany | 80 | 320 |
| baseline | baseline | 0 | UK + US | 80 | 320 |
| null_split_a | target | 1 | Germany (indep.) | 70 | 280 |
| null_split_b | target | 1 | Germany (indep.) | 70 | 280 |

The baseline draws 40 UK + 40 US authors. The two null-split cohorts come from a German author pool that is disjoint from the `target` cohort, then split in half by seed — a same-distribution (German-vs-German) negative control.

## Comparisons

- **target_vs_baseline** — expectation `distinguishable`, `expected_accuracy=0.91`. Positive control on the L1 axis. Goldin et al. 2018 report 91.07% in-domain binary native/non-native accuracy. The signal here is grammatical/stylistic, not topical (shared subreddits): syntactic + semantic should fire while topical JSD stays comparatively low. Flair-based labels are noisy, capping accuracy.
- **null_control** — expectation `null`, `expected_accuracy=0.0`. Seeded random split of the German group; expect C2ST ~0.5, no BH survivors, null MMD.

## Sampling

- Seed: `0`.
- Eligible authors = users with at least 4 usable chunks; up to 4 chunks kept per author (first N in file order).
- target: 80 German authors. baseline: 80 native authors (40 UK + 40 US). null pool: 140 German authors (disjoint from target), split into 70 + 70.
- Author structure is preserved (every prompt's `author_id` is present in `authors.parquet`).

## Caveats

- **Noisy labels**: L1 is inferred from self-selected subreddit country flair, not verified. Some flaired users may be heritage/near-native speakers; this caps separability and is the intended realism.
- **Truncation/segmentation**: chunking + the 120-word cap change the unit relative to the original per-comment / 100-sentence-chunk formats; absolute accuracy is not comparable to the papers' feed-level numbers.
- **Confounds**: even within shared subreddits, German-flair users may raise Germany-specific topics; report topical JSD alongside syntactic/semantic to attribute the signal.
- **Ethics**: real user comments. Only truncated snippets are stored; author ids are the corpus's opaque usernames, namespaced by flair.
