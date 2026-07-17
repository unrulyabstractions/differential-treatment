# multi_value_aave

**Multi-VALUE African-American Vernacular English (AAVE) sensitivity dial.**
A *transform*-based dataset (not a scraped corpus): the label y is exact by
construction — the two sides share identical authors and content, differing
only in whether AAVE morphosyntax was applied.

## Provenance
- **Transform**: `Dialects.AfricanAmericanVernacular()` from Multi-VALUE
  (SALT-NLP/multi-value, `pip install value-nlp`, **Apache-2.0**).
  Ziems et al., *VALUE* (ACL 2022, arXiv:2204.03031) and *Multi-VALUE*
  (arXiv:2212.08011). Rules validated by dialect-speaker judgments upstream.
- **Source corpus (SAE substrate)**: ALL cohorts of `data/synthetic`
  (`target` + `baseline` + `target_twin`, read via `PromptDataset`) pooled
  — 36 authors x 144 short SAE prompts. Every source
  prompt is SAE-grammar English (the synthetic cohorts differ in
  topical/register coding, NOT in grammar), so transforming any of them is
  still an exact-label dial; the shared content cancels in every
  `value_pXX_vs_baseline` comparison. Pooling all three cohorts triples the
  per-cohort sample (was 48 from baseline alone), powering the sensitivity
  curve. This is the project's synthetic fixture, not real user data;
  author demographics (gender/race/age/...) are inherited verbatim as z/d
  metadata and are NOT the y variable.

## License / ethics
- Multi-VALUE code+rules: Apache-2.0. Generated AAVE text is a rule-based
  rendering, not real speakers' writing; do not treat it as authentic AAVE
  data or as evidence about detecting self-reported dialect/identity. This
  dataset only calibrates the pipeline's sensitivity to graded stylistic
  signal (realism is explicitly out of scope).

## The density dial (exact mechanism)
Seeded with `--seed 0`. Each SAE prompt is split into sentence units;
each sentence is transformed once (a per-sentence `dialect.set_seed` makes
the otherwise-stochastic transform reproducible). A sentence is *changed* iff
its transform differs from the original. One seeded uniform u in [0,1) is
drawn per sentence, **independent of p**; in cohort `value_pXX` a changed
sentence is kept transformed iff u < p, else reverted to SAE. Because u does
not depend on p, the transformed sets are **nested**:
`p=0.05 subset p=0.10 subset p=0.25 subset p=0.50 subset p=1.00` (full AAVE).
This realizes the plan's "revert a (1-p) fraction of changed sentences" as a
smooth, monotone, content-preserving dial. (DialectFromFeatureList is also
available but gives a lumpy dial — features fire at very different rates — so
sentence-reversion was chosen for an even p-axis.)

## Per-prompt annotations
- `lgbtq` (generic y flag): 1 for every `value_pXX` prompt, 0 for baseline
  and null splits.
- `markedness`: 1 iff any AAVE rule survives in the density-applied text.
- `codedness`: (# AAVE rules fired in surviving sentences) / (# words),
  i.e. realized rule-application density per word, clamped to [0,1].
- z/d/c beyond the inherited author demographics are left at defaults
  (0 / "") — unrecorded; usage/topical sections skip gracefully.

## Cohorts
- `baseline` (y=0): original SAE prompts.
- `value_p05/p10/p25/p50/p100` (y=1): AAVE at density p=0.05/.../1.00.
- `null_split_a`, `null_split_b`: seeded random halves of the baseline
  authors, both pure SAE (negative control).

## Comparisons
- `value_pXX_vs_baseline` (one per p): sensitivity curve. Prior expectation
  "" for low p (0.05/0.10/0.25), "distinguishable" for high p (0.50/1.00).
  No fixed published accuracy target — signal is tunable by design.
- `null_split_a_vs_b`: negative control, expectation "null"
  (expect C2ST~0.5, 0 BH survivors, null MMD).

## Sampling / sizes
- Author structure preserved: the same 36 source authors appear in every value cohort (exact-y).
- Bounded by the pooled source corpus (36 authors x 144 prompts, `--n-source` capped); no realism subsampling was
  applied — realism is out of scope for a calibration dial.
- Text length: 35-93 words (median 60); all <= 120 words, so **no truncation or**
  **segmentation** was needed.
- Per-cohort prompt counts:
  - `baseline`: 144
  - `value_p05`: 144
  - `value_p10`: 144
  - `value_p25`: 144
  - `value_p50`: 144
  - `value_p100`: 144
  - `null_split_a`: 72
  - `null_split_b`: 72

## Caveats
- Rule-based dialect rendering can be ungrammatical or inconsistent; it
  approximates AAVE morphosyntax, not natural production.
- The signal is definitional (we injected it), so high-p separability
  validates sensitivity only, and says nothing about real-world detection
  of self-reported dialect or identity.
- The SAE substrate is a small synthetic fixture; absolute effect sizes are
  not comparable to the real-corpus datasets.
