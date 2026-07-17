# TwitterAAE (converted)

Tweet text labelled by a demographic language model's posterior, converted to the
paper's D=(x,y,z,d,c) table format (`src/common/dataset_tables.py`). Regenerate
with `uv run python scripts/convert_twitteraae.py --seed 0 --chunk-mb 32`
(idempotent).

## Provenance and license

- Source: `http://slanglab.cs.umass.edu/TwitterAAE/TwitterAAE-full-v1.zip`
  (5.9 GB). **Research use only**; cite Blodgett, Green & O'Connor,
  "Demographic Dialectal Variation in Social Media: A Case Study of
  African-American English", EMNLP 2016 (and Blodgett et al. ACL 2018).
- We do **not** download the full 5.9 GB zip. The converter issues HTTP range
  requests to read only the first **32 MB (compressed)** of the
  `twitteraae_all` member (via the zip central directory offset), caches that
  prefix to the scratch --raw-dir, and stream-decompresses it. The member is
  user-id-sorted, so the prefix is a contiguous head slice of the corpus - fine
  for a smoke test, documented here.
- Each `twitteraae_all` line is 10 tab-separated fields: tweet id, timestamp,
  user id, geo-coordinates, Census blockgroup, tweet text (JSON string), and the
  four model posteriors **[African-American, Hispanic, Other, White]**.

## Cohorts and sampling (seed=0)

- y = the model's posterior alignment: `target` = African-American posterior
  **>= 0.8** (lgbtq flag 1 = generic y), `baseline` = White posterior
  **>= 0.8** (0). This is a lexical demographic model, NOT self-report.
- **One prompt per author, author_id = the streamed ROW INDEX.** The
  `twitteraae_all` file's user ids are partial/reused and we make no attempt to
  group by them; per the task spec each tweet is treated as its own author
  (`twitteraae_row_<n>`). There is therefore no within-author structure.
- x = tweet text, JSON-decoded, stripped, with interior whitespace
  (newlines/tabs/repeated spaces) collapsed to single spaces. Filters: dropped
  retweets (`RT `), tweets containing URLs, tweets under 6 words, and non-English tweets
  (< 90% ASCII characters - the corpus posterior-labels many foreign
  tweets); corpus-wide exact-duplicate texts removed. Tweets are short, but any
  text over **120 words is truncated** to the first 120 for the
  per-unit length cap.
- Seeded subsample: 80 tweets per side (AA pool 4000 candidates,
  White pool 4000 candidates in the prefix).
- `null_split_a` / `null_split_b`: negative control - 80+80
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

- **The label is circular.** End-to-end smoke test (validation-design role: ALL modules must fire). CIRCULAR by construction - smoke test ONLY: every module must fire (many BH survivors, huge MMD, C2ST >> 0.5); a silent module is broken. Document that success says NOTHING about detecting self-reported identity. The y label was assigned FROM the tweet text (posterior of Blodgett et al.'s lexical demographic model >= 0.8), so any lexical/semantic separability is definitional, not a discovery.
- **SMOKE TEST ONLY.** High separability here validates that the pipeline's
  modules fire on an obvious signal; it says nothing about detecting real,
  self-reported identity from prompts.
- `expected_accuracy` is set to **0.0** for `target_vs_baseline`: there is no
  meaningful published classification benchmark (the label is definitional), so
  a large deviation is not interpretable as over/under-performance.
- The prefix is a contiguous, user-sorted head slice, not a uniform sample of
  the 59 M-tweet corpus; treat cohort composition as illustrative.
- Tweets contain informal, sometimes offensive social-media language.
