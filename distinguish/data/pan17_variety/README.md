# pan17_variety

PAN-2017 author profiling, English Twitter feeds (training split), converted
by `scripts/convert_pan17_variety.py --seed 0`.

## Provenance & license
- Zenodo record 3745980, `pan17-author-profiling-training-dataset-2017-03-10.zip` — open access, no registration
  (verified 2026-07-06). Per-author XML feeds (100 tweets each) +
  `truth.txt` (`authorhash:::gender:::variety`).
- PAN shared-task data, research use; cite Rangel Pardo, Rosso, Potthast &
  Stein, "Overview of the 5th Author Profiling Task at PAN 2017", CLEF 2017.
- EN train pool used here: 600 US + 600 GB authors (of 3,600 EN total).

## Task
- **y** = English variety: `target` = united states (lgbtq flag 1),
  `baseline` = great britain (flag 0). The generic `lgbtq` column is the y
  flag, not an identity claim.
- **x** = one individual TWEET. The published anchor (expected_accuracy
  0.715) is per-FEED over 100 tweets — per-tweet separability WILL be lower;
  compare per-author aggregates. Best published EN 6-way variety accuracy:
  0.9004; best EN gender: 0.8233 (CLEF 2017 overview, Tables 3-4).
- Gender labels are available and stored in `authors.parquet`
  (`gender=["woman"|"man"]`) for slices, but this dataset keeps ONE primary
  y (variety); a gender contrast would be a separate dataset.

## Sampling (seeded, author structure preserved)
- Tweet filter: whitespace-normalized; retweets (`RT @...`) dropped; after
  removing URLs/@mentions must keep >= 4 words and >= 20
  chars; per-author exact duplicates dropped. Interior whitespace
  (newlines/tabs/repeated spaces) is collapsed to single spaces. Tweets are
  truncated to 120 words as a guard (no real tweet reaches it — no text was
  actually truncated or segmented).
- 80 authors/side sampled (seed=0) from authors with
  >= 4 qualifying tweets; 4 tweets each
  -> 320 prompts per cohort.
- Negative control: the sampled US authors are randomly split (same seed)
  into `null_split_a` / `null_split_b`; their prompts are duplicated under
  `*_ns` prompt ids. Comparison `null_split` has expectation "null"
  (expect C2ST ~ 0.5, zero BH survivors, null MMD).

## Caveats
- z/d/c are unannotated except author gender: markedness/codedness/topic/
  usage fields are 0/"" ("unrecorded") -> implicit & usage modules skip.
- Variety labels come from PAN's collection procedure (user-declared
  location), not self-reported identity; treat as weak ground truth.
- Tweets are 2017-era Twitter register: URLs, @mentions and hashtags remain
  in the text (only used for the low-content filter).
