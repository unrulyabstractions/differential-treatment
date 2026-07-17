# Findings: validating the pipeline on real datasets

We ran the pipeline on 8 open datasets whose groups are established as separable
(or, for the sensitivity dial and smoke test, separable by construction),
following the validation design in `ITERATION4_PLAN.md`. Configuration: local +
Cohere embedders, linear C2ST, marked words, NeuroBiber, embedding topical/
interactional; ~60–100 authors/side, text units ≤120 words.

**Important scope correction (verified by reading every source paper).** Our
held-out C2ST accuracy is *not* the same quantity as the "published" numbers, and
for most of these datasets the paper reports **no separability metric at all**.
The published figures use different tasks (6-way vs our binary), units (whole
blog / 100-tweet feed vs our single segment/tweet), features (hand-crafted
stylometry vs neural embeddings), and estimators (in-corpus k-fold profiling
accuracy vs a held-out two-sample C2ST). So the validation is **not** "recover
the published number." What actually validates the pipeline is: (a) **above-chance
C2ST wherever the literature establishes the groups differ, paired with a matched
negative control at ≈0.50**; (b) the correct **qualitative module behaviors**
(topical low on Reddit, embeddings blind to dialect, etc.); (c) the correct
**relative orderings**. The table below states each paper's actual reported figure
and why it is or isn't comparable.

## Results and the true published comparators

| Dataset | Contrast (our unit) | What the paper actually reports | Our C2ST | Comparability |
|---|---|---|---|---|
| PAN-2017 | US vs GB, per tweet | **no binary figure**; 6-way EN variety per-author acc (chance 16.7%): best 90.0% SVM, median 79.7%. The 71.5% we'd cited is the cross-system *mean*, not an SVM | **0.71/0.81/0.79** | NOT COMPARABLE (binary vs 6-way, tweet vs feed). Above chance 0.50 w/ clean null = US/GB style is separable here; **not** a benchmark match, and the 90% SVM exceeds us |
| Reddit-L2 | German-L1 vs native, per chunk | native-vs-*pooled*-nonnative (not German): 91.07% *content* features (char-3gram); func-word+POS is lower — 90.8% (TACL) / FW 80.3%, POS 69.1% | 0.65/0.74/0.72 | RELATED, below the ceiling as expected; different task/unit/features. "func+POS" was our mislabel — 91% is content features |
| Blog | female vs male, per segment | 80.1% whole-blog gender acc (MCRW, 502 style+1000 content feats, 10-fold CV, unit ≈ one ~7.9k-word blog) | 0.57/0.60/0.58 | RELATED; ~65× more text/instance + different model/estimator → 80.1% is a loose ceiling, not a target |
| TwitterAAE | AAE vs White-aligned, per tweet | **none** — corpora are model-defined (§2.3), so separability is circular by construction | 0.69/0.81/0.79 | NOT COMPARABLE; self-consistency smoke test only |
| Multi-VALUE | SAE vs AAVE (p=1) | **none** — only task-degradation (e.g. CoQA F1 −6.7/−8.4%) + rule acceptability | 0.79 @ p=1 | NOT COMPARABLE; internal detection vs the deterministic transform label (exact) |
| PRISM | female vs male, per prompt | **none** — nearest is a topic~demographics regression (73/638 coeffs sig) | 0.52 (n.s.) | NOT COMPARABLE; exploratory, ~chance |
| ThoughtTrace | female vs male, per turn | **none** (no benchmark) | 0.54 (sig) | NOT COMPARABLE; exploratory, weak-but-detectable |
| WildChat | US vs GB, per prompt | **none** — only IP-geolocation country *prevalence* (US 21.6%/UK 3.8%) | 0.52 (n.s.) | NOT COMPARABLE; proxy label, exploratory null |

The load-bearing check is the **negative control**, which held near-chance on
C2ST everywhere (random within-group author split): PAN 0.45, Blog 0.50, Reddit
0.48, TwitterAAE 0.50, ThoughtTrace 0.51 — all n.s. Above-chance target-vs-
baseline C2ST against a 0.50 null is the honest statement of "these groups are
distinguishable," independent of any published number.

## Cross-cutting findings (module sensitivities)

1. **C2ST is the robust separability metric; MMD-Fuse over-rejects on
   heterogeneous populations.** Blog's negative control (female split A vs
   split B) fired 4/12 tests — *all* MMD-Fuse and topical, while C2ST stayed at
   0.50. A random split of a demographically heterogeneous group (blog age bands
   13–17/23–27/33–47 mixed) is not distributionally identical, so the sensitive
   two-sample MMD test rejects on sub-population imbalance, whereas C2ST measures
   label *predictability* under CV and correctly stays at chance. **Trust C2ST
   over MMD for separability on real, heterogeneous data.**

2. **Detection is a meaning-vs-grammar axis — the sensitivity curve.** The
   Multi-VALUE dial applies AAVE morphosyntax at densities p ∈ {0.05…1.0}, so
   `sensitivity_curve.png` plots each module's detection evidence vs p (144
   prompts/side for power). The pattern is clean: **NeuroBiber (syntactic) rises
   monotonically** with p and clears α at p≈0.5 (p=8e-7 at full density); the
   **C2ST discriminative catch-all fires only at full surface change** (p=1.0,
   acc 0.67–0.79); **semantic MMD is blind at every density** (meaning is
   preserved, and embeddings encode meaning); lexical/topical/interactional stay
   flat. AAVE markers do surface raw-z-significant lexically ("ain't" z=2.69) but
   none survive BH — it is morphosyntactic, not lexical. So the modules split by
   *what kind* of difference they see (grammar vs meaning), not surface-vs-
   embedding. Negative control clean (0/10). This is the exact-label calibration
   the dial was built for.

3. **NeuroBiber's binary presence features miss rate-based grammatical signal.**
   Reddit-L2 (native vs non-native English) is 91%-separable by function-word
   and POS-trigram *rates* — but NeuroBiber was silent (0/96, p=0.24), because
   its 96 features encode *presence/absence* of Biber categories, and both
   groups use the same feature inventory at different rates. The embedding C2ST
   (0.74) caught the signal the binary features washed out.

4. **Lexical calibration must be tuned per corpus (now automatic).** A fixed
   calibration constant that suppresses register words on the exaggerated
   synthetic fixture (C=0.035) over-suppresses subtler real corpora, while a
   larger C that surfaces real signal lets function words flag on the synthetic.
   The default is now `calibration_constant="auto"`: a per-corpus binary search
   for the largest C at which no register word (stopwords + contractions + top
   English-frequency vocab) clears raw |z|≥1.96 — MotS's own `find_optimal_alpha`
   methodology. Verified: no function/contraction word flags on any corpus
   (synthetic OR real), real content words surface (Reddit-L2: german/germany/uk),
   and negative controls stay at 0. The resolved C is stored and marked on the
   constant-sweep plot.

5. **BH-FDR is underpowered on sparse per-word data; raw z surfaces the signal.**
   TwitterAAE's label is definitionally lexical, yet lexical marked words gave 0
   under BH — but the AAE markers (`amp`, `gt` = `&amp;`/`&gt;` tweet artifacts,
   `shit`) surface under MotS's own raw |z|≥1.96 rule. With ~1200 tokens/side and
   256-word vocabularies, BH over the whole vocabulary has no power. The pipeline
   now **always reports the raw-z count and words alongside BH** so "0 significant"
   is never uninformative (`n_significant_raw_z` in every lexical result). MotS
   itself uses raw z; the added BH (per the paper draft) is the conservative
   choice, appropriate only at larger sample sizes.

6. **The survey-topic catalog does not fit non-survey datasets.** The embedding
   topical backend assigns every prompt to the nearest of the paper's 15 MH/GSH/
   REL survey topics — meaningless for blog posts or tweets, so real-data topical
   JSD is near-null and uninformative. Use the **TopicGPT backend** (generates a
   taxonomy from the corpus) for non-survey datasets; the fixed survey catalog is
   only appropriate for the survey-derived data it was built for.

7. **Subtle regional style is separable above chance — with a clean null — but
   this is not a benchmark win.** PAN-2017 US-vs-GB per-tweet C2ST is 0.71–0.81
   against a matched null at 0.45, so the pipeline does detect subtle regional
   register. Correction after reading the source: this does **not** "match or
   exceed" the shared task — that task was 6-way per-author (chance 16.7%, best
   SVM 90.0%), a different and harder problem our binary per-tweet number cannot
   be compared to. The credibility claim is the weaker, true one: subtle
   coded-style *is* detectable above a clean chance baseline, so detecting
   implicit LGBTQ+ signaling is a plausible target — not that we beat 2017
   stylometry.

8. **Gender is a weak signal in real chatbot prompts.** PRISM and ThoughtTrace
   (gold self-reported gender, real LLM conversations) gave C2ST 0.52–0.54 —
   barely above chance, with only the sensitive MMD test firing. People's
   chatbot prompts encode gender far less than their blogs or tweets do. The
   power-analysis cohorts (n=12/24 authors/side) lost significance entirely,
   quantifying the sample size the modality demands.

## Residual-stream scaling study (§3.3.4 / §3.3.5 extension)

We swept the residual-stream C2ST representation across a **six-model scale
ladder** (gemma-2-2b/9b/27b, Qwen2.5-14B/32B/72B) on all seven real datasets plus
a within-target **null_control** (target set split in half by author — separability
there can only be overfitting). Representation: **whole-prompt mean-pooled** hidden
state at 75% depth (this beat the paper's change-of-turn pooling by ~+0.03 C2ST in
a separate sweep). Full matrix in `docs/residual_scaling_matrix.json`.

| model | twitteraae | reddit | pan17 | blog | prism | thoughttrace | wildchat | **null** | real-μ |
|---|---|---|---|---|---|---|---|---|---|
| gemma-2B | 0.863 | 0.766 | 0.775 | 0.577 | 0.562 | 0.574 | 0.573 | 0.500 | 0.670 |
| gemma-9B | 0.838 | 0.781 | 0.823 | 0.602 | 0.578 | 0.562 | 0.564 | 0.537 | 0.678 |
| Qwen-14B | 0.844 | 0.805 | 0.792 | 0.607 | 0.584 | 0.571 | 0.577 | 0.512 | 0.683 |
| gemma-27B | 0.875 | 0.792 | 0.823 | **0.615** | 0.581 | 0.571 | 0.583 | 0.562 | **0.692** |
| Qwen-32B | 0.838 | 0.809 | 0.787 | 0.593 | 0.545 | 0.537 | 0.590 | 0.525 | 0.671 |
| Qwen-72B | **0.887** | **0.841** | 0.792 | 0.610 | 0.584 | 0.558 | 0.567 | 0.463 | 0.691 |

All accuracies are held-out author-grouped 5-fold CV; significance is a 10-permutation
author-label null (p floored at 0.091 = observed beat all 10). Findings:

1. **No high-dim overfitting at any scale.** Every `null_control` sits at chance
   (0.46–0.56, all p≥0.45) — the residuals (up to 8192-D on ~640 samples) never
   manufacture signal. This is the load-bearing calibration check; without it the
   accuracies would be uninterpretable.
2. **Scale buys little, and it's front-loaded.** real-μ: 2B 0.670 → 72B 0.691, only
   **+0.02 for 36× the parameters**, non-monotonic (32B dips below 14B). **gemma-27B
   (0.692) ties the 72B (0.691)** — family/quality matters as much as size.
3. **The hard cases are scale-invariant.** blog + the three real-prompt sets
   (prism/thoughttrace/wildchat) stay ~0.54–0.62 across the *entire* ladder
   (per-dataset spread ≤0.04). On thoughttrace the **best model is gemma-2B**
   (0.574); on prism it's Qwen-14B. Where the distinction is genuinely faint, more
   parameters give no advantage.
4. **Scale's one payoff is at the near-null margin.** On prism/thoughttrace the 32B
   was non-significant (p=0.18–0.27), but the 72B nudges both to marginally
   significant (p=0.091) — scale converts "indistinguishable" to "weakly detectable"
   without raising accuracy much.
5. **Residual > text embedders on the hard sets.** The real-prompt datasets scored
   0.52/0.54/0.52 with text embeddings (above) but ~0.56–0.58 with residual streams
   — the hidden state carries a little more of the faint signal.
6. **Reproducible.** The 72B was measured on two independent boxes and returned
   byte-identical accuracies (0.8875/0.8406/0.7922 on the first three).

The attributional probe (§3.3.5) uses this same representation on gemma-2-2b and,
on the synthetic fixture, Claude Opus 4.8 named the discriminating concept as
"LGBTQ+ identity vocabulary (queer/trans/gender/coming out)" from the
exact per-token contributions (probe acc 1.00).

## What validates and what does not

- **Above-chance where signal exists, chance where it doesn't.** PAN/Reddit/Blog
  give significant above-0.50 C2ST; their matched within-group null splits sit at
  0.45–0.50 (n.s.). This — not proximity to any published number — is the
  validation. The published figures are different quantities (verified per paper);
  we do **not** claim to recover or beat them.
- **Negative controls clean on C2ST** across all datasets; MMD/topical show the
  documented heterogeneity over-rejection.
- **Smoke test passes** (TwitterAAE): every module fires except lexical-under-BH,
  which the raw-z reporting now exposes. Its label is model-defined, so this is
  self-consistency, not external validation.
- **Sensitivity dial works** (Multi-VALUE): reveals module-specific blindness
  against an exact, deterministic transform label (no external comparator).
- **Real-prompt exploratory sets** (PRISM/ThoughtTrace/WildChat) show that
  demographic signal in short prompts is genuinely faint — an ecologically
  important null-leaning result, not a pipeline failure. None has a published
  separability comparator; these are exploratory, not validation.
- **What would be over-claiming** (now removed): "recovers published separability,"
  "matches 71.5%," "exceeds the 2017 SVM." Each compared different tasks/units/
  estimators; every source paper was read to confirm the real reported quantity.

Raw run outputs and per-dataset READMEs are under `data/<name>/README.md`;
analysis working notes in the session scratchpad.
