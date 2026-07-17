# Blog Authorship Corpus (gender contrast)

Converted by `scripts/convert_blog_authorship.py` to the paper's
`D=(x,y,z,d,c)` Parquet format. **Not** synthetic — real blog posts.

## Provenance

- Source: HF `tasksource/blog_authorship_corpus` (`blogtext.csv`, 681,284 posts / 19,320 bloggers).
  Original corpus: Schler, Koppel, Argamon & Pennebaker, *Effects of Age
  and Gender on Blogging* (AAAI Spring Symposium 2006).
- License: original corpus is **non-commercial research use only**
  (Schler et al.); the HF mirror is tagged apache-2.0. Treat as
  non-commercial research data.

## The contrast (y)

- `target` = female bloggers (y=1, `lgbtq=1`); `baseline` = male
  bloggers (y=0). `lgbtq` here is the generic target flag, not sexual
  orientation (the corpus has no orientation field).
- Published 2-way gender accuracy **80.1%** (Schler et al. 2006,
  Multi-Class Real Winnow, style+content) -> `expected_accuracy=0.801`.

## x-unit: a POST SEGMENT (reported per validation design)

- Each post is cleaned (the corpus's `urlLink` placeholder tokens are
  stripped and runs of whitespace collapsed) and TRUNCATED to its first
  **<= 120 words** (whitespace-tokenized) = one segment,
  one segment per post. Posts with < 20 clean words are dropped.
- This is much shorter than the full posts / per-author feature vectors
  behind the 80.1%; per-prompt separability will run lower. Aggregate to
  the author (PromptSet groups by author) for the fair comparison.
- **Topic vs style caveat:** the corpus's `topic` field (blogger
  industry, e.g. Student/Technology) and any survey `c`-context are NOT
  stored (`domain=''`, `topic_id=0`, all ordinals 0) because they do not
  map onto the paper's MH/GSH/REL survey catalog. Topical analysis is
  therefore derived from the text itself; attributing distinguishability
  to topic (teens write about school) vs style is part of the finding.

## Sampling (seeded, author structure preserved)

- Seed: 0. Authors kept only if they have >= 2 clean posts; up to **4 posts per
  author** are sampled (seeded shuffle).
- `target`: 80 female authors; `baseline`:
  80 male authors; balanced across the corpus's three
  age bands (13-17 / 23-27 / 33-48) so age is not a gender confound.
- Negative control: 60 + 60 held-out
  female authors (disjoint from `target`), randomly split into
  `null_split_a` / `null_split_b`.

## Author (z, d) fields

- `gender`: female->["woman"], male->["man"] (self-reported binary).
- `age`: numeric age mapped to a canonical bracket (13-17 kept as-is;
  else 18-24 / 25-34 / 35-44 / 45-54). The corpus only contains ages
  13-17, 23-27, 33-48 (Schler's design). **Slice caveat:** the
  under35/35plus age slice matches bracket *strings*, so 13-17 authors
  fall on the 35plus side -- rely on the balanced sampling, not that
  slice, for age control.
- `transgender`, `orientation`, `pronouns`, `race`, `disability`,
  `education`, `income`: unrecorded (""/[]); the corpus lacks them.

## Cohort counts (this build)

| cohort | authors | prompts |
|---|---|---|
| target | 80 | 305 |
| baseline | 80 | 305 |
| null_split_a | 60 | 226 |
| null_split_b | 60 | 225 |

## Comparisons

- `target_vs_baseline`: female vs male, expectation *distinguishable*,
  `expected_accuracy=0.801`.
- `null_control`: random female author split, expectation *null*.

Load with `PromptDataset.load(Path('data/blog_authorship'))`.
