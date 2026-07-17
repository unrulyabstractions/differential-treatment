# Synthetic test-fixture dataset

**These are exaggerated, fully synthetic test fixtures — not research data.** Every
prompt, author profile, and survey answer was written by an AI assistant purely to
exercise the distinguishability pipeline. The target cohorts are deliberate
caricatures of community-coded language and do **not** represent any real community,
population, or user; the baseline cohort is an equally deliberate caricature of a
terse cis-het register. The identity, demographic, and usage annotations are
invented to create designed contrasts, not measured ones. Do not use these files to
support any empirical claim.

## Format: D = (x, y, z, d, c)

One dataset = one directory in the Parquet-table format of
`src/common/dataset_tables.py`, loaded with
`PromptDataset.load(Path("data/synthetic"))`:

    data/synthetic/
    ├── dataset.json       # manifest: cohorts + comparisons (DatasetManifest.to_dict())
    ├── prompts.parquet    # one row per prompt: x, y, c + cohort + author_id
    └── authors.parquet    # one row per author: z, d + cohort

Every row carries a `cohort` label (`target` / `baseline` / `target_twin`), and
`dataset.prompt_set("<cohort>")` rebuilds the in-memory `PromptSet`
(`src/common/prompt_set_schema.py`) for one cohort, mirroring the paper's dataset
(Tables 1-2). Author-level attributes (z, d) are columns of `authors.parquet`
(the multi-select fields `gender`, `orientation`, `pronouns`, `race` are
`list<string>` columns); each `prompts.parquet` row carries the text x, labels y,
and interaction context c as flat columns. `"*"` = prefer not to answer; ordinal
`0` = unrecorded (never used here — every ordinal is filled in). Schemas:
`src/common/dataset_annotations.py`.

| y (per prompt) | meaning |
|---|---|
| `lgbtq` | 1 = target group (LGBTQ+), 0 = baseline |
| `markedness` | 1 = explicit identity signal in the text (see reading below) |
| `codedness` | implicit identity-signal strength in [0, 1] |

| z (per author) | d (per author) |
|---|---|
| `transgender` ("0"/"1"/"*") | `race` (multi-select) |
| `gender` (multi-select) | `age` bracket |
| `orientation` (multi-select) | `disability` ("0"/"1"/"*") |
| `pronouns` (multi-select) | `education`, `income` brackets |

| c (per prompt) | meaning |
|---|---|
| `topic_id`, `domain` | fixed survey catalog (1-15; MH/GSH/REL) |
| `provenance` | "real" (recalled real prompt) or "hyp" (hypothetical) |
| `adoption` | 1-5, how long ago the author adopted chatbots (per author) |
| `general_freq` | F 1-8, overall chatbot use (per author) |
| `llm_freq` | F 1-8, chatbot use for this domain (per author x domain) |
| `professional_freq` | F 1-8, professional help for this domain (per author x domain) |
| `aversion`, `satisfaction` | A 1-5 attitudes (per author x domain) |

Survey-style answers are constant within an author (`adoption`, `general_freq`)
or within an author x domain cell (the four domain scales).

## Cohorts and comparisons

`dataset.json` (`DatasetManifest.to_dict()`) names the three cohorts and the two
comparisons the dataset supports: `target_vs_baseline` (target vs baseline,
expectation "distinguishable", explorations on) and `target_vs_twin` (target vs
target_twin, expectation "null", explorations off).

| Cohort | `group` | Display name | Prompts | Authors | MH / GSH / REL |
|---|---|---|---|---|---|
| `target` | target | "Target (LGBTQ+)" | 48 | `t01..t12` (4 each) | 22 / 15 / 11 |
| `baseline` | baseline | "Baseline (cis-het)" | 48 | `b01..b12` (4 each) | 32 / 6 / 10 |
| `target_twin` | target | "Target twin (null check)" | 48 | `tw01..tw12` (4 each) | 20 / 16 / 12 |

Topic ids follow the paper's fixed 15-topic survey catalog (Table 2c); the source
of truth is `src/topical/survey_topic_catalog.py`. Per-topic counts:

| topic_id | Domain | Description | target | baseline | twin |
|---|---|---|---|---|---|
| 1 | MH | isolation, anxiety, depression, or panic in social settings | 10 | 22 | 10 |
| 2 | MH | coping with a major life transition or shift in self-view | 4 | 3 | 3 |
| 3 | MH | whether to start seeing a therapist | 2 | 3 | 1 |
| 4 | MH | values conflicting with family or community expectations | 6 | 4 | 6 |
| 5 | MH | improving body image | 0 | 0 | 0 |
| 6 | GSH | understanding one's gender or sexual orientation | 5 | 0 | 6 |
| 7 | GSH | STI symptoms, causes, or treatments | 2 | 3 | 3 |
| 8 | GSH | understanding gender-affirming care | 3 | 0 | 2 |
| 9 | GSH | side effects of gender/sexual-health medications (PrEP, estrogen) | 2 | 3 | 2 |
| 10 | GSH | finding knowledgeable healthcare providers | 3 | 0 | 3 |
| 11 | REL | communication, intimacy, or boundaries in a new relationship | 3 | 5 | 3 |
| 12 | REL | exploring a new relationship dynamic | 2 | 0 | 2 |
| 13 | REL | communicating needs to a partner | 4 | 5 | 6 |
| 14 | REL | media featuring relationships like one's own | 1 | 0 | 1 |
| 15 | REL | preparing for a new sexual experience | 1 | 0 | 0 |

Note: the catalog has no topics for friendship/community belonging, family
conflict, or breakups, so target/twin's community-, family-, and breakup-themed
prompts honestly land in MH topics 1, 4, and 2. MH is therefore the largest
single domain in every cohort, but target/twin keep a much heavier GSH + REL share
(26 and 28 of 48) than baseline (16 of 48).

## Authors (z, d)

Target/twin author identities were written to be consistent with each author's
four texts (e.g. the author whose prompts discuss starting estrogen is annotated
as a trans woman): a mix of trans men/women, nonbinary folks, cis queer/bi women,
questioning (`transgender: "*"`), one ace-spectrum author, ages skewing 18-54,
varied race (occasionally multi-select), education and income spreads, 3
disabled authors and occasional `"*"` per cohort. Baseline authors are straight cis
men and women (`transgender: "0"`, she/her or he/him) with varied demographics.

**Provenance and voice.** ~70% of each cohort's prompts are `"real"` (recalled) and
~30% `"hyp"`, spread across authors. Hypothetical prompts are not always
self-voiced: where a frozen text conflicts with its author's profile (e.g. a
married baseline man's one "my husband ..." prompt), that prompt is annotated
`"hyp"` — a prompt the respondent imagined rather than recalled. `"real"`
prompts may also be recalled from an earlier life stage (e.g. b08's
during-the-marriage prompt next to her post-divorce one).

## Labels (y): markedness and codedness profile

Markedness reading (annotated per text, honestly): 1 iff the text **on its own
explicitly states the author's identity** — a label applied to self (including
self-inclusive "we" labels like "the only out queer couple") or the author's own
transition/hormone status ("I've been on T for five months"). Community
references, bare outness references without a label, and coded slang (binder,
egg cracked, polycule) stay 0 and are carried by `codedness` instead. For the
**baseline**, an explicit different-gender partner reference ("my wife" from a
male-profiled author) counts as an explicit heteronormative identity marker, so
baseline markedness is *high* while its LGBTQ+ codedness is ~0.

| Cohort | markedness = 1 | mean codedness |
|---|---|---|
| target | 15 / 48 | 0.85 (range 0.55-0.95) |
| baseline | 41 / 48 | 0.01 (range 0.00-0.10) |
| target_twin | 15 / 48 | 0.85 (range 0.70-0.95) |

## Intended contrast: target vs baseline (every dimension should fire)

- **Lexical**: disjoint signature lexicons of ten words per side, each repeated
  across many prompts and authors so the calibrated marked-words test has real
  power. Target signature words (total occurrences in target / twin / baseline):
  partner 26/23/2, community 33/29/0, queer 27/27/0, chosen 25/24/1,
  family 31/30/1, pride 26/24/1, folks 25/24/1, gender 26/24/1, trans 26/25/0,
  out 33/32/3 — plus lower-frequency flavor words (HRT 7, T 5, estrogen 4,
  binder, drag, polycule, situationship, church-hurt). Baseline signature words
  (baseline / target / twin): work 38/2/2, gym 26/0/0, wife 25/0/0,
  budget 23/0/0, schedule 22/1/1, husband 21/1/1, routine 21/1/1,
  girlfriend 19/1/1, dating 19/1/1, marriage 17/1/1. Cross-set occurrences of
  1-3 are deliberate natural crossovers.
- **Syntactic**: target is long, hedged, first-person, emotionally disclosing
  (~63 words/prompt, contractions, "I guess", "kind of", "maybe"); baseline is
  direct, imperative, low-emotion, and concrete (~47 words/prompt, "Give me…",
  "List…", "Rank…", digits over number words, no contractions, minimal
  negation and wh-questions).
- **Semantic / distributional**: the register and vocabulary gaps put the two
  sets in clearly separable embedding regions; a classifier two-sample test
  should beat chance by a wide margin.
- **Topical**: topic distributions differ sharply (domain mixes 22/15/11 vs
  32/6/10; baseline puts 22 of 48 prompts on topic 1 and has zero mass on
  topics 6, 8, 10, 12, 14, and 15), so topic-distribution divergence (JSD)
  should be significant. Baseline's few GSH prompts stay generic-clinical.
- **Usage / attitudes (paper RQ2, `c` annotations)**: designed contrast in the
  survey scales — target authors use chatbots more for these domains, consult
  professionals less, and are more averse, while satisfaction is designed to be
  a null (near-identical means). Prompt-level means:

  | scale | target | baseline | twin | design |
  |---|---|---|---|---|
  | `llm_freq` | 5.88 | 3.90 | 5.88 | target higher (fires) |
  | `professional_freq` | 1.94 | 3.88 | 1.94 | target lower (fires) |
  | `aversion` | 3.44 | 2.38 | 3.44 | target higher (fires) |
  | `satisfaction` | 3.52 | 3.52 | 3.58 | designed n.s. |
  | `general_freq` | 5.92 | 4.92 | 5.92 | target slightly higher |
  | `adoption` | 3.67 | 3.33 | 3.67 | no designed contrast |

**Expected pipeline outcome**: target vs baseline → distinguishable on all
content dimensions with small p-values, plus the designed usage contrasts above
(satisfaction excepted).

## Intended null: target vs target_twin (no dimension should fire)

`target_twin` is drawn from the same distribution as `target`: same voice, same
hedged register, same coded lexicon at matched frequencies (every signature
word within ±15% of target's count; crossover words matched exactly),
near-identical per-topic counts (max per-topic difference: 2, on topic 13),
near-identical length profile (~66 vs ~63 words/prompt), matched markedness
(15 vs 15) and codedness (0.85 vs 0.85), and matched usage-scale distributions
(same ranges; prompt-level means within ~2% on every scale, exact on most —
see table above) — but entirely new scenarios (not paraphrases), disjoint
author ids, and freshly authored (not copied) author profiles and survey
answers.

**Expected pipeline outcome**: target vs twin → null on the calibrated tests.
Observed (seed 0, on these same texts under the fixtures' previous ids):
lexical 0 significant words (p≈0.99), syntactic 0/96 features (p=1), semantic
text MMD-Fuse p=0.34, topical JSD 0.045 (p=0.93). The two *strongest* tests do
register marginal signal — residual-stream MMD-Fuse p≈0.043 and linear C2ST
accuracy 0.625 (p≈0.015, vs 0.98 on the real contrast) — because a
hand-written twin cannot be perfectly exchangeable with target: the scenarios
are new even though voice, lexicon, topics, and lengths are matched. Treat
those two as the framework's sensitivity floor for this fixture pair, not as a
bug; anything beyond marginal (p ≪ 0.01, accuracy ≫ 0.7) on this pair *would*
indicate a defect. The usage scales should be n.s. across the board on this
pair.
