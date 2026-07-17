# Iteration 4 plan (user request, 2026-07-06 evening)

Session state for continuation. Repo at commit d573e05 (iteration 3 complete,
verified). COHERE_API_KEY exists in ~/.zshrc (available to uv run via profile).
wordfreq covers "i'm" (1.4e-3) — its significance is a calibration-procedure
question, not missing data.

## Work items (user's 10 points)

1. **implicit/** replaces codedness/ under each section: breakdown of BOTH
   y-annotations — markedness (explicit) and codedness (implicit): reruns at
   codedness thresholds AND markedness==0/1 splits, one evidence chart each.
2. **Lexical calibration overhaul**:
   - Fetch + read the More of the Same code (Mickel et al. 2025, arXiv
     2503.00333 — find GitHub) and align our calibrated marked words with their
     Algorithm 3 (hyperparameter tuning so common/register words like "i'm"
     don't flag); document deviation if any.
   - Calibration-justification plots: null-pair z QQ plot vs N(0,1),
     n_significant vs prior_strength sweep, weight sweep — saved under
     lexical/calibration/.
   - Word-cloud plots (marked words per side, size ~ |z|).
   - Slices must report the actual marked WORDS per slice (full section result
     JSON + plots per slice, not just verdicts).
   - Confirm BH: scipy false_discovery_control(method="bh") — show in docs.
3. **Real datasets in data/**: converter scripts (scripts/convert_*.py or
   src/datasets_import/) to our parquet format for: blog_authorship (HF,
   gender), reddit_l2 (L1), prism (HF), pan17 (registration — skip if blocked),
   twitteraae (rehydration — document limits; maybe skip), multi_value (GitHub
   transform on a corpus = exact-y sensitivity calibration), thoughttrace (HF),
   okcupid (JSE, ethics note), wildchat (HF, country), toefl11 (LDC — document
   skip). Sub-sample each to ~200-400 prompts/side with author structure;
   schema must tolerate missing z/d/c (make sections skip usage/topical
   gracefully when context is unannotated — "0 = unrecorded" already).
   Run the pipeline on every converted dataset; compare separability with the
   published numbers in the user's table (Blog 80.1% gender; Reddit-L2 90.8%;
   TwitterAAE near-definitional; Multi-VALUE tunable). Analyze + iterate.
4. **Syntactic**: add per-group histogram distributions (per-prompt feature
   counts by group); palette/color improvement pass across ALL plots.
5. **Cohere**: run with key from ~/.zshrc (source in runner env) — variants
   activate; verify embed-v4.0 output dims.
6. **Topical / proper TopicGPT**: read TopicGPT paper (arXiv 2311.01449) +
   github (chtmp223/topicGPT); implement generation phase with seed topics
   ORTHOGONAL to survey topics (style/intent-level seeds applying to any
   question), refinement + assignment; new backend "topicgpt:<model>";
   compare with survey-topic taxonomy.
7. **slices/ structure**: slices/gender/{women,men,nonbinary}/,
   slices/race/{...}/, slices/age/{under35,35plus}/... each slice dir gets the
   FULL section output (json + plots), plus the per-facet evidence chart.
8. **Analysis pass**: after real-dataset runs, read each dataset's paper,
   reason whether results make sense, find bugs, iterate.
9. **Comparison-vs-comparison plots**: run-root plot juxtaposing
   target_vs_baseline vs target_vs_twin evidence per test (paired bars).
10. No shortcuts; verify everything with image tokens + verifier agent.

## Suggested sequencing
A. Research (background agents): More-of-the-Same code; TopicGPT paper+repo;
   dataset access/format notes for all 10.
B. Core restructure: explorations -> implicit/ + slices/{facet}/{value}/ with
   full outputs; run-comparison plot; syntactic histograms; palette pass.
C. Lexical calibration: align with MotS code, calibration plots, word clouds.
D. TopicGPT backend.
E. Dataset importers + runs (subsampled; heavy runs may use vast.ai GPU —
   cloud/ scripts ready; balance ~$3.6).
F. Full verification + docs + commits at each milestone.

## Research: TopicGPT

Sources read directly: paper page https://arxiv.org/abs/2311.01449 ("TopicGPT: A
Prompt-based Topic Modeling Framework", Pham, Hoyle, Sun, Resnik, Iyyer, NAACL 2024,
v2 2024-04-01) and full source of https://github.com/chtmp223/topicGPT cloned to
scratchpad (`topicgpt_python` v0.2.8 in-repo; PyPI ships 0.2.7). All file paths below
refer to that repo unless prefixed with `src/`.

### 1. The actual method (from source, not the paper prose)

TopicGPT = two phases plus refinement, all plain-text prompting with a line-based
topic format. The canonical topic line everywhere is:

```
[<level>] <Label>: <Description>
```

e.g. `[1] Trade: Mentions the exchange of capital, goods, and services.` Stored trees
add a count: `[1] Environment (Count: 4): Involves the management ...` (that is the
`.md` taxonomy file format, `TopicTree.to_file`).

#### Phase 1 — topic generation (`generation_1.py::generate_topic_lvl1`)

- **Seed topics** (`prompt/seed_1.md`): a markdown file of `[1] Label: Description`
  lines — the demo uses exactly ONE seed (`[1] Trade: ...`). Seeds anchor the
  *granularity and style* of generated topics via in-context examples; they are
  loaded into the initial `TopicTree`.
- **Prompt** (`prompt/generation_1.txt`, placeholders `{Topics}`, `{Document}`): shows
  the current topic list, two worked examples (one "add new topic", one "return
  existing duplicate"), then instructions: topics must be GENERALIZABLE,
  single-concept, level-numbered, output format `[Topic Level] Topic Label: Topic
  Description`, or `None` if no topic. One call **per document**, sequential.
- **How new topics are added** (`generation_1.py::generate_topics`): parse each
  response line with `regex r"^\[(\d+)\] ([\w\s\+_#-]+):(.+)"`; skip lines with
  level != 1; `TopicTree.find_duplicates(name, lvl)` does **case-insensitive exact
  label match** — duplicates just increment the existing node's `count`, novel labels
  are appended to the tree, and the updated topic list feeds the next document's
  prompt. An `early_stop` counter (default 100 in the function, 1000 via CLI) aborts
  after that many consecutive duplicate-only documents. Note: the topic list injected
  into the prompt is labels only — `topic.split(":")[0]` strips descriptions
  (`prompt_formatting`, which also handles context overflow by truncating the doc or
  sub-selecting topics by SBERT `all-MiniLM-L6-v2` cosine similarity).
- Output: taxonomy `.md` (`generation_1.md`) + per-doc responses `.jsonl`.

#### Refinement (`refinement.py::refine_topics`) — merging + pruning

- **Pair selection** (`topic_pairs`): embed all topic strings with SBERT
  `all-MiniLM-L6-v2`, take pairs with cosine similarity **> 0.5** (threshold), at most
  **2 pairs per round** (`num_pair=2`), never re-prompting a pair seen before
  (`all_pairs` memory).
- **Merge prompt** (`prompt/refinement.txt`, placeholder `{Topics}`): "merge topics
  that are paraphrases or near duplicates ... Return `None` if no modification is
  needed." Expected merge-line format (parsed by
  `r"^\[(\d+)\]([\w\s',_+#\-]+)[^:]*:([\w\s,\.\-\/;']+) \(([^)]+)\)$"`):
  `[1] NewLabel: new description ([1] Original A, [1] Original B)`.
- **Tree update** (`TopicTree.update_tree`): originals removed, merged node gets the
  summed count; a `mapping` dict {original label -> new label} is persisted
  (`refinement_mapping.json`) and used to rewrite the Phase-1 responses file
  (`update_generation_file` -> `refined_responses` column).
- **Pruning** (`remove_topics`): drop level-1 topics whose *generation-phase* count is
  **< 1% of total count** (`threshold=0.01`), gated by `remove: True` in `config.yml`.
- Loop until no new similar pairs remain (guaranteed to terminate because each pair is
  prompted at most once).

#### Phase 2 — assignment (`assignment.py`) + self-correction (`correction.py`)

- **Assignment prompt** (`prompt/assignment.txt`, placeholders `{tree}`,
  `{Document}`): full topic list (with descriptions) + 2 worked examples; required
  output `[Topic Level] Topic Label: Assignment reasoning ("Supporting quote")`; hard
  rules "You MUST NOT make up new topics / quotes"; ends with "Double check that your
  assignment exists in the hierarchy!". One call **per document** (vLLM path batches).
  If the tree exceeds context, only the topics most SBERT-similar to the doc are shown.
- **Self-correction** (`correction.py::correct_topics`): `topic_parser` re-extracts
  labels with `r"^\[\d+\] ([\w\s\-'\&]+):"` and flags docs whose response has (a) no
  parseable topic ("error") or (b) a label not in the tree ("hallucinated"). Those
  docs are re-prompted with `prompt/correction.txt` — identical to assignment but with
  an extra `{Message}`: *"Previously, this document was assigned to: {old response}.
  Please reassign it to an existing topic in the hierarchy."* One correction pass,
  then re-parse; leftovers are only warned about. Correction default sampling is
  temperature 0.6 / top_p 0.9 (deliberately non-greedy so the retry differs).
- **Output format**: JSONL rows carrying the input columns plus `prompted_docs`
  (possibly truncated doc) and `responses` (the raw `[1] Label: reasoning ("quote")`
  string — downstream metrics re-parse labels from it).

#### Package structure & reuse verdict

- Pip-installable: **yes** — `pip install topicgpt_python` (PyPI name
  `topicgpt-python`, latest published **0.2.7**; the GitHub tree is 0.2.8 which moves
  vllm to an extra). API = 6 top-level functions re-exported from
  `topicgpt_python/__init__.py`: `sample_data`, `generate_topic_lvl1`,
  `generate_topic_lvl2`, `refine_topics`, `assign_topics`, `correct_topics` — all
  file-path-in/file-path-out (jsonl/md), no in-memory API.
- **Do NOT depend on it.** Concrete blockers verified in source:
  1. `utils.py::APIClient.iterative_prompt` always sends
     `temperature`/`top_p` (defaults 0.0/1.0) — **gpt-5-mini rejects any non-default
     temperature**, so every call would 400. Our own `src/topical/topic_assignment.py`
     already deliberately omits temperature for this reason.
  2. PyPI 0.2.7 hard-requires `vllm>=0.6.3.post1` (plus `vertexai`, `anthropic`,
     `google-generativeai`, `sentence-transformers`) — a huge, GPU-flavored dep tree.
  3. Sequential 1-doc-per-call prompting with `time.sleep(60)` retry backoff; no JSON
     output contract; file-based I/O only; module-level SBERT load on import.
- **Worth reusing (copy/adapt, not import):** the four prompt texts
  (`prompt/generation_1.txt`, `refinement.txt`, `assignment.txt`, `correction.txt`),
  the `[1] Label: Description` line format + parse regexes, the dedup/early-stop
  generation loop, the sim>0.5 pair-selection + merge + <1% prune refinement rules,
  and the "reassign with previous-answer message" correction idea (our existing
  indexed-JSON retry loop in `_assign_batch` is already a stronger version of it).
- Small upstream bug worth avoiding: `TopicTree.from_seed_file` parses the whole
  `Label: Description` line into `node.name`, so seed topics never match
  `find_duplicates` against generated labels — split seeds on the first `:` in our
  implementation.

### 2. Design: `topicgpt:<model>` topical backend

Current state (read first): `src/topical/topical_dimension.py` runs every spec in
`TopicalConfig.assignment_backends` (`"embedding" | "openai:<model>"`) against the
fixed 15-topic `SURVEY_TOPICS` catalog, computes topic/domain JSD on
interleave-pooled assignments, and permutation-tests at author level.
`src/topical/topic_assignment.py::_assign_by_openai` already implements
TopicGPT-Phase-2-equivalent batched assignment (indexed JSON batches of 40, no
temperature, validation + one self-correction retry). The new backend adds
**Phase 1**: derive the catalog from the pooled corpus instead of using the survey
catalog.

#### Spec plumbing

- New spec `"topicgpt:<model>"` (e.g. `"topicgpt:gpt-5-mini"`) in
  `assignment_backends`. Add `"topicgpt": "OPENAI_API_KEY"` to `_PROVIDER_KEYS` in
  `src/inference/embedding_store.py` so `embedder_unavailable_reason` keeps gating it
  on the key (the spec is never used as an *embedding* spec, so `EmbeddingStore._compute`
  is unaffected); `assignment_model_name` in `topic_assignment.py` learns the new
  prefix.

#### New module `src/topical/topicgpt_taxonomy.py`

```python
@dataclass
class GeneratedTopic(BaseSchema):
    topic_id: int          # 1..K, assigned in generation order after refinement
    label: str             # e.g. "Advice Seeking"
    description: str
    generation_count: int  # docs that proposed/matched it in Phase 1

def build_taxonomy(texts, model_name, config, context) -> list[GeneratedTopic]:
    """Phase 1 + refinement on the pooled (interleaved) texts."""
```

- **Generation loop** (port of `generate_topics`): one chat call per pooled prompt,
  no temperature/max_tokens overrides (gpt-5-mini), system message = adapted
  `generation_1.txt` with `{Topics}` = current `[1] Label: Description` lines
  (keep descriptions — dropping them was an upstream accident) and `{Document}` =
  the user prompt. Parse `re.compile(r"^\[(\d+)\]\s*([^:]+):\s*(.+)$")` per line;
  accept level-1 only; case-insensitive label dedup increments `generation_count`;
  early-stop after `topicgpt_early_stop` consecutive all-duplicate docs. Adapt the
  instruction text: replace the bills examples with two prompt-domain examples and
  add one sentence — "Topics must describe the *communicative intent or style* of the
  user's request (what the user is trying to get from the assistant), NOT its subject
  matter." This is the lever that keeps the taxonomy orthogonal to survey topics.
- **Refinement** (port of `merge_topics` + `remove_topics`): embed
  `f"{label}: {description}"` via `context.embedding_store.get_text_embeddings(...,
  config.embedding_model)` (reuses the existing all-MiniLM cache — no new SBERT dep);
  pairs with cosine > `topicgpt_merge_similarity` (default **0.5**), max 2 per round,
  each pair prompted once, `refinement.txt` prompt, parse
  `[1] New: desc ([1] A, [1] B)` merges, cap at `topicgpt_max_refine_rounds`
  (default 10); then prune topics with `generation_count <
  topicgpt_prune_fraction * total` (default **0.01**) — but never prune a seed topic.
- **Determinism/caching**: gpt-5-mini has no greedy mode, so Phase 1 is
  nondeterministic. Persist the built taxonomy to the run dir
  (`topical/topicgpt_taxonomy.json`) and into the result schema so every run is
  auditable; build it ONCE per run on the interleaved pool (target+baseline via the
  existing `interleave_texts`) so taxonomy construction cannot leak set identity.

#### Assignment (Phase 2) — reuse, parameterized

Generalize `_assign_by_openai`/`_assign_batch` to take `catalog_lines: list[str]` and
`valid_ids: set[int]` instead of reading `SURVEY_TOPICS` (survey backend passes the
survey catalog, topicgpt backend passes the generated one). The existing
validate-and-retry loop *is* the self-correction step; keep `_OPENAI_BATCH_SIZE = 40`,
`_OPENAI_MAX_ATTEMPTS = 2`.

#### Dimension integration (`topical_dimension.py`)

- `_run_backend` gains a catalog argument: topic counts via
  `np.bincount(ids, minlength=K+1)[1:]`; JSD + author-level permutation test unchanged.
- Domain JSD is survey-specific: for `topicgpt:*` backends emit `domain_jsd =
  float("nan")` and empty `domain_rows` (viz already tolerates missing rows;
  verdict detail text drops the domain clause).
- `TopicalBackendResult` gains `catalog: list[GeneratedTopic] | None = None`
  (None for survey backends) so `topic_rows.short_name` can be the generated label
  and reports/plots stay self-describing.
- **Comparison with survey taxonomy** (work item 6): both backends run in the same
  section, so the report already juxtaposes their JSDs/p-values; add one extra plot
  `topical/topicgpt_shares.png` (per-generated-topic shares, same style as survey
  shares) and optionally a survey×topicgpt cross-tab heatmap from the pooled
  assignments (both id vectors already exist in memory).

#### Config additions (`TopicalConfig` in `src/common/run_config.py`)

```python
topicgpt_seed_topics: str = "style"      # named seed set below
topicgpt_merge_similarity: float = 0.5   # upstream default
topicgpt_prune_fraction: float = 0.01    # upstream default (<1% pruned)
topicgpt_early_stop: int = 25            # consecutive dup-only docs (corpus ~400-800)
topicgpt_max_generation_docs: int = 400  # cap Phase-1 cost; interleaving keeps balance
topicgpt_max_refine_rounds: int = 10
```

`configs/config.json`: `"assignment_backends": ["embedding", "openai:gpt-5-mini",
"topicgpt:gpt-5-mini"]`. Cost: Phase 1 ≤ 400 small calls + ~20 refinement calls;
Phase 2 identical to the existing openai backend (N/40 calls) — well within gpt-5-mini
budget.

#### Seed topics — ORTHOGONAL to the survey topics (style/intent/stance level)

The survey catalog is *subject-matter* (what the question is about: MH/GSH/REL). The
topicgpt seeds must therefore describe *how/why the user is asking* — properties any
question about any subject can have, so the two taxonomies measure independent axes.
Draft seed file (format identical to `prompt/seed_1.md`), 8 seeds:

```
[1] Factual Inquiry: Asks for objective information, facts, or an explanation of how something works.
[1] Advice Seeking: Requests personalized guidance or recommendations about what the asker should do.
[1] Emotional Disclosure: Shares feelings, worries, or personal struggles, seeking support, empathy, or validation.
[1] Decision Support: Lays out options or a dilemma and asks for help weighing or choosing between them.
[1] Procedural Guidance: Asks for concrete step-by-step instructions for accomplishing a specific task.
[1] Creative Composition: Asks the assistant to draft, write, or rehearse text on the asker's behalf (e.g. a message to send someone).
[1] Opinion Elicitation: Asks for the assistant's own judgment, stance, or evaluation of something.
[1] Self-Understanding: Seeks help interpreting the asker's own identity, feelings, behavior, or experiences.
```

These deliberately span intent (inquiry/advice/procedure), stance (opinion
elicitation), and style/affect (emotional disclosure, self-understanding) — none
encodes MH/GSH/REL content, so JSD on this taxonomy is evidence about *register and
intent* differences between prompt sets, complementary to the survey-topic JSD.
Phase 1 may still add topics (that is the point of TopicGPT); the seeds + the added
instruction sentence anchor the level of abstraction.

#### Implementation order

1. `src/topical/topicgpt_taxonomy.py` (prompts as module constants, `build_taxonomy`).
2. Parameterize catalog in `topic_assignment.py`; add `topicgpt:` spec + key gating.
3. Generalize `_run_backend` + schemas; NaN domain JSD for generated catalogs.
4. Config fields + `configs/config.json` + `configs/README.md` backend docs.
5. Plot + report hooks; taxonomy JSON artifact; tests with a faked OpenAI client
   (deterministic canned responses) covering dedup, merge parsing, pruning, and the
   assignment retry path.

## Research: More of the Same code

Sources (all read directly, 2026-07-06):

- Code: https://github.com/jennm/more-of-the-same (confirmed as the official repo in the paper's NeurIPS checklist). Key files: `calibrated_marked_words.py` (the method), `og_marked_words.py` (Cheng et al. Marked Personas baseline), `utils.py` (`compute_calibrated_marked_words` driver), `find_optimal_alpha.ipynb` + `alpha_values_table.md` (the alpha sweep). Cloned to scratchpad `more-of-the-same/`. No LICENSE file in the repo.
- Paper: arXiv:2503.00333 (NeurIPS 2025). Appendix A.2 (pp. 20-21) describes the method; Algorithm 2 = Marked Personas (Cheng et al.), Algorithm 3 = "Calculation of regularizing terms" (the calibration), Algorithm 4 = Calibrated Marked Words (p. 22). Tables 3/5 (pp. 23-24) = alpha sweep; B.4 (p. 33) = statistical significance discussion.

### The exact method (from `calibrated_marked_words.py::get_log_odds`, defaults `p=0.15`, `prior_type='hybrid'`, `lower=True`)

Inputs per comparison: `df1` = target-group texts, `df2` = unmarked-group texts, `df0` = **all** texts in scope (all groups, not just the two being compared) — `df0`'s counts are the "topic prior" `P_topic`.

1. **Tokenization**: lowercase, whitespace split, then strip every non-letter char (`replace('[^a-zA-Z\s]','',regex=True)`). So `i'm` -> `im`, `eco-friendly` -> `ecofriendly`, digits vanish; empty tokens deleted.
2. **Hybrid prior (count space, not probability space)**: `P_english` = raw lowercased token counts of the **NLTK Brown corpus** (~1.16M tokens). For each word *already in the topic prior's vocabulary*:
   `P[w] = int(p * P_topic[w] + (1-p) * P_english[w])` with `p = 0.15` (weight on topic). Words in Brown but absent from the generated corpus are NOT added. Then every `P[w]` is rounded `int(P[w]+0.5)`, and any word that appears in `counts1`/`counts2` with `P[w]==0` is floored to `P[w]=1`. Because Brown dwarfs the topic corpus, for in-Brown words the hybrid prior is effectively >95% English in probability terms; the topic side mainly matters for words Brown has never seen.
3. **Constants**: `c_english = 0.225`, `c_topic = 0.45` (in the hybrid branch — the `c_english=.0001625`/`c_topic=0.005078125` near the top of the function are dead code, overwritten). `c = p*c_topic + (1-p)*c_english = 0.25875`, so `1/c ≈ 3.865`. (`prior_type='english'` alone uses `c=.225`; topic-only uses `c=.45`.)
4. **Calibration word set W** (this is what Algorithm 3 calls "calibration words"): `W = (top20(P) ∩ top20(counts1) ∩ top20(counts2)) ∪ HARDCODED_STOPWORDS`, where top20 = the 20 highest-count words (min-heaps, `num_common_words = 20`) and the hardcoded set is the 27 words `{the, of, and, to, a, in, for, is, on, that, by, this, with, i, you, it, not, or, be, are, from, at, as, your, am, an, my}`.
5. **Regularizing terms (Algorithm 3)**: with `w_p = Σ_{w∈W} P[w]`, `w_g1 = Σ_{w∈W} counts1[w]`, `w_g2 = Σ_{w∈W} counts2[w]`:
   `r1 = c * w_p / w_g1`, `r2 = c * w_p / w_g2`.
   The prior actually used for side *i* is `P[w]/r_i` — i.e. **per-side, data-driven prior strength**. Summed over W, side *i*'s prior mass on calibration words is exactly `w_gi / c ≈ 3.865 × w_gi`.
6. **Log-odds z (Algorithm 4 / Monroe et al. 2008)**, with `n1 = Σ counts1`, `n2 = Σ counts2`, `n_P = Σ P` (post-mixing totals):
   - `l1 = (y1_w + P[w]/r1) / (n1 + n_P/r1 - y1_w - P[w]/r1)` and symmetrically `l2` with `r2, n2`
   - `σ²_w = 1/(y1_w + P[w]/r1) + 1/(y2_w + P[w]/r2)`
   - `δ_w = (log l1 - log l2) / σ_w`  — note the returned "delta" is **already the z-score**.
   (Paper typo alert: Algorithm 2/4 pseudocode writes `n_U` in the `l1` denominator; the code uses the group's own total `n1`. Follow the code.)
7. **Significance rule**: raw two-sided threshold, `z > 1.96` marked for target, `z < -1.96` marked for the unmarked group (paper text says "z-score ≥ 1.96"; code is strict `>`). **No multiple-testing correction of any kind** — Appendix B.4's only corrections-adjacent stats are Welch t-tests on the downstream SRB scores. Every word of the full-corpus vocabulary is tested, including singletons (no min-count filter; missing prior floored to 1).
8. **Aggregation** (`calibrated_marked_words()`): when the target is contrasted against multiple unmarked axes (e.g. race and gender), a word must clear 1.96 in **all** comparisons (`Counter` count `>= len(unmarked_val)`), and its reported score is the **sum of z-scores** across comparisons. For their gender runs `unmarked_val = ['M']`, so this is a no-op. `utils.compute_calibrated_marked_words` additionally: builds the unmarked ("M") word list as the intersection of "M vs g" runs across all other genders g (must appear in `n_genders - 1` runs), and **removes person names** (list from `data/names_with_dem.csv`) from the output.

Why "the"/"be"/"im" stop flagging: each calibration word gets a pseudo-count ≈ `3.865 × (its share of P) × w_gi`, i.e. several times that group's own observed count, and both sides are shrunk toward the *same* hybrid rate. δ collapses ∝ 1/α while σ only shrinks ∝ 1/√α, so z → 0 for common words; content words have tiny `P[w]`, keep nearly raw counts, and still clear 1.96. And because `r_i` scales with each corpus's size (`w_gi`), the shrinkage is corpus-size-adaptive — exactly the failure mode of Cheng et al.'s uncalibrated prior with unequal corpus sizes.

### Their hyperparameters and how they chose them (Appendix A.2 + `find_optimal_alpha.ipynb`)

| Hyperparameter | Value | How chosen |
|---|---|---|
| `C_english` | 0.225 | Binary search on [0,1], English-only prior, "maximize # significant words while excluding common words" |
| `C_topic` | 0.45 | Same binary search, topic-only prior |
| mixing `α` (code `p`, weight on topic) | **0.15** | Sweep 0→1 in 0.05 steps (notebook: `i/20`) on gpt-4o-mini software-engineer generations with specified gender (a ~0.0079% sample of their data, 50% gender-balanced); criterion: minimize common words + names among significant words while preserving gender-related words |
| effective `C` | 0.25875 | `= α·C_topic + (1-α)·C_english` |
| top-k for W | 20 | hardcoded (`num_common_words`) |
| z threshold | 1.96 (two-sided) | fixed; no correction |
| English corpus | NLTK Brown, raw lowercased counts | fixed |

### Comparison with `src/lexical/marked_words_analyzer.py` (defaults from `configs/config.json` lines 15-19 / `src/common/run_config.py` lines 45-52: `min_word_count=2`, `reference_corpus="wordfreq:en"`, `english_prior_weight=0.5`, `prior_strength=500.0`, `fdr_alpha=0.05`)

Precise deviations, most important first:

1. **We have no calibration step — the defining "Calibrated" feature is missing.** We use one fixed, symmetric `alpha = prior_strength * pi` (500 total pseudo-counts) for both sides. They compute per-side strengths `P[w]/r_i` from the data; per side the prior mass on common words alone is `w_gi/0.25875 ≈ 3.865 × w_gi` (typically tens of thousands of pseudo-counts), and it automatically rescales with each corpus's size. With our fixed 500, imbalanced target/baseline sizes or high-frequency function-word rate differences can still flag "the"-class words, and the suppression strength is arbitrary rather than tuned.
2. **Hybrid mixing space differs.** Ours: probability-space mix `pi = 0.5*p_ref + 0.5*p_corpus` (each renormalized over the min-count vocabulary). Theirs: raw-count mix `0.15*topic_counts + 0.85*brown_counts`, which in probability terms is overwhelmingly English (Brown ≈ 1.16M tokens vs topic corpora of ~10^4). Our 0.5 English weight is far more topic-heavy than their effective prior.
3. **Topic-prior corpus scope**: theirs is the full dataframe in scope (all groups, including groups outside the pair being compared); ours is target+baseline only. Only matters when >2 groups exist.
4. **Significance rule**: ours = two-sided p + Benjamini-Hochberg at 0.05 (stricter and more principled); theirs = raw |z| > 1.96, no correction, plus the multi-axis intersection/z-sum. A faithful replication needs a `raw_z` mode.
5. **Vocabulary**: ours tests words with combined count ≥ 2; theirs tests every word in the full corpus (singletons included, prior floored at count 1). Also our reference floor is 1e-9 relative frequency vs their integer floor of 1 count.
6. **Reference corpus**: wordfreq:en (modern, well-normalized) vs NLTK Brown raw counts (1961; words absent from Brown get prior `int(0.15*count)` which truncates small counts to 0 before the +0.5 rounding, then floors to 1). wordfreq is arguably better, but word lists won't be comparable one-for-one.
7. **Tokenization**: ours keeps internal apostrophes and splits on any non-letter (`i'm` stays `i'm`; `eco-friendly` -> `eco`,`friendly`); theirs deletes non-letters inside whitespace tokens (`im`, `ecofriendly`, `aidriven`). Their published marked-word tables can't be matched exactly without a compatibility tokenizer.
8. **Name handling**: they strip person names from outputs post-hoc; we don't (probably fine for prompt-distinguishability, but noise like model-favorite names will surface as marked words).
9. API nit: their returned "delta" is the z-score; our `MarkedWord.log_odds` is the raw prior-adjusted delta with `z_score` separate — keep ours, it is strictly more informative.

### What to change (implementation-ready)

Add a calibrated-prior mode to `compute_marked_words_table` (keep BH-FDR as our default significance; add `significance="raw_z"` with threshold 1.96 only for replication runs):

```python
CALIBRATION_STOPWORDS = frozenset({
    "the", "of", "and", "to", "a", "in", "for", "is", "on", "that", "by",
    "this", "with", "i", "you", "it", "not", "or", "be", "are", "from",
    "at", "as", "your", "am", "an", "my",
})
C_ENGLISH, C_TOPIC, TOPIC_WEIGHT = 0.225, 0.45, 0.15
CALIBRATION_C = TOPIC_WEIGHT * C_TOPIC + (1 - TOPIC_WEIGHT) * C_ENGLISH  # 0.25875


def calibrated_side_alphas(vocabulary, prior_counts, y_t, y_b, top_k=20):
    """Mickel et al. Algorithm 3: per-side Dirichlet pseudo-counts."""
    def top(v):
        return set(np.asarray(vocabulary)[np.argsort(-v)[:top_k]])

    calibration = (top(prior_counts) & top(y_t) & top(y_b)) | CALIBRATION_STOPWORDS
    in_w = np.isin(vocabulary, sorted(calibration))
    w_p = prior_counts[in_w].sum()
    r_t = CALIBRATION_C * w_p / y_t[in_w].sum()
    r_b = CALIBRATION_C * w_p / y_b[in_w].sum()
    return prior_counts / r_t, prior_counts / r_b  # alpha_t, alpha_b
```

with `prior_counts = TOPIC_WEIGHT * (y_t + y_b) + (1 - TOPIC_WEIGHT) * english_counts` in **count space** (for wordfreq, scale to Brown-like magnitude: `english_counts = word_frequency(w, "en") * 1_161_192`), and the Monroe delta/variance generalized to per-side priors:

```python
delta = np.log((y_t + a_t) / (n_t + a_t.sum() - y_t - a_t)) \
      - np.log((y_b + a_b) / (n_b + a_b.sum() - y_b - a_b))
variance = 1.0 / (y_t + a_t) + 1.0 / (y_b + a_b)
```

Concrete change list:

1. `src/lexical/marked_words_analyzer.py`: support per-side `alpha_t`/`alpha_b` in the delta/variance block (lines 121-132) — the current single-`alpha` path is the `prior_strength` mode; add `prior_calibration: "fixed" | "mickel"` (default `"mickel"`), with `"mickel"` replacing `prior_strength` by the Algorithm-3 regularizers above.
2. Add `significance: "bh_fdr" | "raw_z"` (default `"bh_fdr"`, `raw_z` uses 1.96) so we can reproduce their tables when needed.
3. Optional replication fidelity knobs (off by default): count-space hybrid mixing with `topic_weight=0.15`; a `tokenizer="mickel"` variant that deletes non-letters inside whitespace tokens; testing singletons (`min_word_count=1`).
4. Reconsider `english_prior_weight=0.5` if we keep the probability-space mix: their effective English share is ≈0.95+, and our 0.5 leaves common-word suppression mostly to `prior_strength`, which is 2-3 orders of magnitude weaker than their calibrated mass.

Verification note: all formulas/constants above were read directly from the cloned source (`calibrated_marked_words.py` lines 17-133, `utils.py`, `find_optimal_alpha.ipynb`) and cross-checked against the paper PDF pages 20-24 and 33 (viewed as images); the C constant was recomputed (`0.25875`, `1/C = 3.8647`).

## Research: datasets

Research agent findings (2026-07-06). Every access claim below was tested live
today (HTTP status / API / actual download) unless marked otherwise. Target
schema recap (src/common/dataset_tables.py): `prompts.parquet` needs
`prompt_id, author_id, cohort, text, lgbtq, markedness, codedness, topic_id,
domain, provenance, adoption, general_freq, llm_freq, professional_freq,
aversion, satisfaction`; `authors.parquet` needs `author_id, cohort,
transgender, gender, orientation, pronouns, race, age, disability, education,
income`. `gender/orientation/pronouns/race` must be **lists** (parquet arrays;
`_as_list` accepts None), the rest strings/ints. `lgbtq` is the generic y
target flag: **1 for every prompt in the target cohort, 0 in baseline/twin**.
Unknown annotations: int 0 / float 0.0 / "" / [] ("0 = unrecorded").

### Feasibility summary (tested 2026-07-06)

| dataset | source | gating | verdict |
|---|---|---|---|
| blog_authorship | HF `tasksource/blog_authorship_corpus` | none (public CSV) | FEASIBLE TODAY |
| reddit_l2 | public Google Drive links (Haifa CL page) | none (links live; official README says "email ellarabi@gmail.com") | FEASIBLE TODAY (see note) |
| prism | HF `HannahRoseKirk/prism-alignment` | none | FEASIBLE TODAY |
| pan17 | Zenodo record 3745980 | **open** — no registration (verified by downloading + unzipping the 52 MB train zip) | FEASIBLE TODAY |
| twitteraae | slanglab.cs.umass.edu direct zip | none (HTTP 200 verified) | FEASIBLE TODAY (5.9 GB download) |
| multi_value | GitHub SALT-NLP/multi-value (`pip install value-nlp`) | none, Apache-2.0 | FEASIBLE TODAY (transform, not corpus) |
| thoughttrace | HF `SCAI-JHU/ThoughtTrace` | none, CC-BY-4.0 | FEASIBLE TODAY |
| okcupid | GitHub rudeboybert/JSE_OkCupid | none — **but essays are deliberately decoupled from labels** | BLOCKED for our purpose (skip; ethics note) |
| wildchat | HF `allenai/WildChat-1M` | none (ODC-BY since 2024-06; verified public signed URL) | FEASIBLE TODAY |
| toefl11 | LDC LDC2014T06 | LDC account + license fee | BLOCKED (document skip) |

### Shared converter scaffolding (src/datasets_import/)

```python
PROMPT_DEFAULTS = dict(lgbtq=0, markedness=0, codedness=0.0, topic_id=0,
                       domain="", provenance="", adoption=0, general_freq=0,
                       llm_freq=0, professional_freq=0, aversion=0, satisfaction=0)
AUTHOR_DEFAULTS = dict(transgender="", gender=[], orientation=[], pronouns=[],
                       race=[], age="", disability="", education="", income="")

def make_tables(rows, dataset_name):
    """rows: list of dicts with keys author_id, cohort, text, + optional overrides.
    Assigns prompt_id=f"{dataset_name}_{i:05d}", fills defaults, dedups authors."""
```
Common subsampling recipe (all datasets): filter texts to 100–2000 chars,
drop exact duplicates, then sample **60–100 authors per cohort, up to 4 texts
per author** (seed=0) → 240–400 prompts/side, preserving author structure
(validate() requires every prompt's author_id in authors.parquet). Twin-null
pair: split the *target* authors into two disjoint halves → cohorts `target` /
`target_twin` (mirrors data/synthetic/dataset.json: comparisons
`target_vs_baseline` expectation "distinguishable", `target_vs_twin`
expectation "null", explorations false).

### 1. Blog Authorship Corpus — FEASIBLE TODAY

- **Use** `tasksource/blog_authorship_corpus` (plain CSV mirror; the canonical
  `barilan/blog_authorship_corpus` is a *loading-script* dataset that modern
  `datasets` refuses to run — dataset viewer rejects it too).
- **File**: single `blogtext.csv`, 800 MB, 681,284 rows (posts), ~19K bloggers.
  Direct URL: `https://huggingface.co/datasets/tasksource/blog_authorship_corpus/resolve/main/blogtext.csv`
- **Fields** (verified via datasets-server first-rows): `id` (int, blogger id —
  repeats across posts = author id), `gender` ("male"/"female"), `age` (int),
  `topic` (industry, e.g. "Student" — map to `domain`), `sign`, `date`, `text`.
- **License**: original corpus "may be used for non-commercial research"
  (Schler et al.); mirror tagged apache-2.0.
- **Recipe**: `pd.read_csv(url)` (or `hf download tasksource/blog_authorship_corpus`),
  filter 200–2000 chars, cohorts `female` (target, lgbtq=1... i.e. y=1) vs
  `male`; `authors.gender=["female"]`, `authors.age=str(age)`,
  `prompts.domain=topic`, `provenance="blog_authorship"`.
- **Published sanity check**: Schler, Koppel, Argamon & Pennebaker 2006
  ("Effects of Age and Gender on Blogging", AAAI SSS) — **80.1% gender
  accuracy** (Multi-Class Real Winnow, style+content features) — matches the
  user's table. Age 3-way: 76.2%.

### 2. Reddit-L2 (native vs non-native) — FEASIBLE TODAY

- **Official channel**: github.com/ellarabi/reddit-l2 README says "data
  available per request — ellarabi@gmail.com". BUT the (now-archived) corpus
  page `cl.haifa.ac.il/projects/L2/` published public Google Drive links, and
  both are **still live** (verified: Drive returns the virus-scan interstitial,
  i.e. file exists and is publicly downloadable):
  - The Reddit-L2 corpus: `https://drive.google.com/file/d/1RlKn2AFWjrlnh_FPZK_PcdNVthwzrpyj/view`
  - Goldin et al. processed chunks folder: `https://drive.google.com/drive/folders/1Lk7BIhpU1YFG2yeHNhS1DQbqGb69JDZi`
  - second corpus file: `https://drive.google.com/file/d/18iJuQJi_rIrarZjXJF-gdVR8ZddH-r7f/view`
  Download with `gdown --fuzzy <url>` (folder: `gdown --folder`). Courtesy
  email to the authors is still the polite move; note this in the converter
  docstring.
- **Contents** (Rabinovich, Nisioi, Ordan & Wintner TACL 2018; Goldin,
  Rabinovich & Wintner EMNLP 2018, D18-1395): English Reddit posts, ~230M
  sentences / 3.5B tokens; author country from subreddit flair (r/Europe,
  r/AskEurope, r/EuropeanCulture) as L1 proxy. Goldin subset: 34,511 users, 23
  L1s / 29 countries, sentences tagged with **user id, subreddit, country**,
  grouped in 100-sentence per-user chunks.
- **Recipe**: cohorts `nonnative` (target: all non-English-speaking-country
  flairs) vs `native` (UK/US/Ireland/Australia/NZ flairs). Unit: concatenate
  ~5 consecutive sentences of one user into one prompt (single Reddit
  sentences are too short; 100-sentence chunks too long). `author_id=user`,
  put country in `authors.race=[]`-NO — keep country in the cohort description
  and per-L1 slices via a sidecar column? Schema has no free column: encode L1
  in `topic_id` (int-mapped) or run per-L1 datasets later. `provenance="reddit_l2"`.
- **Published sanity check**: Goldin et al. 2018 binary native/non-native
  accuracy, in-domain: **91.07%** with content features (char-3grams + token
  unigrams + spelling; Table 2 — the user's 90.8% matches this ballpark),
  **93.40%** with all features incl. Reddit metadata (Table 7); 23-way NLI
  62.06% (char 3-grams) / 86.05% (all features).

### 3. PRISM — FEASIBLE TODAY

- **HF id**: `HannahRoseKirk/prism-alignment` (ungated; verified). Files:
  `survey.jsonl` (1,500 users), `conversations.jsonl` (8,011),
  `utterances.jsonl` (68,371), `metadata.jsonl`.
- **License**: human-written text (incl. prompts) **CC-BY-4.0**; model
  responses CC-BY-NC-4.0 (we only need the human prompts).
- **Fields** (verified via datasets-server): survey → `user_id, age, gender,
  employment_status, education, marital_status, english_proficiency, religion,
  ethnicity, location, lm_familiarity, lm_frequency_use, lm_usecases,
  stated_prefs, self_description, system_string`; utterances → `utterance_id,
  user_id, conversation_id, turn, user_prompt, model_response, score`;
  conversations → `opening_prompt, conversation_history, open_feedback`.
- **Recipe**: `load_dataset("HannahRoseKirk/prism-alignment", "utterances")` +
  `"survey"`, join on `user_id`, keep turn==0 user_prompts (or
  conversations.opening_prompt). Cohorts by a survey demographic — gender
  (female vs male) is the cleanest first run; religion/ethnicity/age as
  slices. Bonus: `lm_frequency_use` maps onto our `llm_freq` (ordinal-code the
  categories), `lm_usecases` → `domain`. Fill authors from survey: gender,
  age, education, ethnicity→race, religion→(no column; drop). No published
  separability number — treat as exploratory (expectation "").

### 4. PAN-2017 author profiling — FEASIBLE TODAY (no registration!)

- **URL**: https://zenodo.org/records/3745980 — `access_right: open`
  (verified via Zenodo API **and** by downloading + unzipping; the folklore
  that PAN data needs registration does not apply to this record).
  Files: `pan17-author-profiling-training-dataset-2017-03-10.zip` (53 MB),
  `...test-dataset-2017-03-16.zip` (35 MB). Direct:
  `https://zenodo.org/records/3745980/files/pan17-author-profiling-training-dataset-2017-03-10.zip?download=1`
- **Format** (verified by extraction): per language dir (`en/ es/ pt/ ar/`),
  one XML per author (`<author lang="en"><documents><document><![CDATA[tweet]]>...`,
  100 tweets/author) + `truth.txt` lines `authorhash:::female:::canada`.
  EN train: **3,600 authors** (gender-balanced), varieties: australia, canada,
  great britain, ireland, new zealand, united states.
- **Recipe**: stdlib `xml.etree` + regex on truth.txt. Single tweets are short
  → concatenate 5 tweets per prompt (→ up to 20 prompts/author; take 3-4).
  Cohorts `female` vs `male` (en); variety kept for slices via `topic_id`
  int-map or a second dataset (`pan17_variety`: us vs great britain).
  `provenance="pan17"`. License: PAN terms = research use.
- **Published sanity check** (overview: Rangel Pardo, Rosso, Potthast & Stein,
  CLEF 2017, ceur-ws.org/Vol-1866/invited_paper_11.pdf, Tables 3-4, verified):
  best **gender EN accuracy 0.8233** (Basile et al. "N-GrAM", SVM char 3-5
  grams + word 1-2 grams; avg over languages 0.8253); best variety EN 0.9004
  (Tellez et al.). So ~82% is the anchor, consistent with the user's table.

### 5. TwitterAAE — FEASIBLE TODAY (text included, NO rehydration needed)

- **URL**: `http://slanglab.cs.umass.edu/TwitterAAE/TwitterAAE-full-v1.zip`
  (verified HTTP 200; ~5.9 GB; note the /TwitterAAE/ path — the bare-host path
  404s). Page: http://slanglab.cs.umass.edu/TwitterAAE/
- **Contents** (zip central directory listed via ranged request):
  `TwitterAAE-full-v1/README.txt`, `twitteraae_all`, `twitteraae_all_aa`,
  `twitteraae_limited`, `twitteraae_limited_aa`. TSV lines carry tweet id,
  timestamp, user id, geolocation, census blockgroup, **tweet text**, and the
  4 posterior proportions from Blodgett/Green/O'Connor's (EMNLP 2016)
  mixed-membership model in order [African-American, Hispanic, Other, White].
  ~59M tweets ("_aa" files = AA-posterior-heavy subsets; "limited" = stricter
  filtering). Terms: research use only + cite EMNLP 2016 / ACL 2018.
- **Recipe**: download once to scratch, stream-parse. Standard convention in
  the literature: cohort `aa_aligned` = posterior AA ≥ 0.8, `white_aligned` =
  posterior White ≥ 0.8. `author_id=user id`; cap 2 tweets/author; tweets are
  short → optionally concatenate 3 per prompt. `provenance="twitteraae"`.
- **Sanity check**: labels are **near-definitional** (the y label *is* the
  output of a language model trained on census-geolocation seeds — circular
  with any lexical separability test). Expect very high separability; use as
  positive control and say so in the manifest description. Published: 99%+
  separable dialect classifiers are trivial here; Blodgett et al. report their
  demographic model's held-out correctness, not a classification benchmark.

### 6. Multi-VALUE — FEASIBLE TODAY (transform, not corpus → exact-y calibration)

- **Repo**: https://github.com/SALT-NLP/multi-value — license **Apache-2.0**
  (verified LICENSE), `pip install value-nlp`. Needs spaCy en pipeline +
  nltk wordnet (`python -m spacy download en_core_web_sm; nltk.download('wordnet')`).
- **API** (verified in src/multivalue/Dialects.py): 50 dialect classes, e.g.
  `AfricanAmericanVernacular, AppalachianDialect, ChicanoDialect,
  IndianDialect, ColloquialSingaporeDialect, ...` plus `DialectFromVector` /
  `DialectFromFeatureList` for custom feature subsets (189 features total).
  ```python
  from multivalue import Dialects
  aave = Dialects.AfricanAmericanVernacular()
  out = aave.transform("I talked with them yesterday")
  rules = aave.executed_rules   # which of the 189 features fired
  ```
- **Recipe** (this is our *exact-y sensitivity dial*): take an existing
  baseline cohort's texts (blog male cohort or PRISM prompts), build
  `target = transform(text)` and `baseline = text` with **identical authors
  and content** → y is the ONLY difference, by construction. Tune effect size
  two ways: (a) `DialectFromFeatureList(feature_list=[...])` with k features
  (k∈{5,20,all}); (b) transform only a fraction p∈{0.25,0.5,1.0} of the
  target cohort's prompts. Log `executed_rules` count into `markedness`
  (0/1: any rule fired) and density into `codedness`. Separability is tunable
  ~50%→high, per the user's table ("tunable"); the VALUE papers (Ziems et al.
  ACL 2022, arXiv 2204.03031; Multi-VALUE arXiv 2212.08011) validate rules
  with dialect-speaker judgments.

### 7. ThoughtTrace — FEASIBLE TODAY

- **HF id**: `SCAI-JHU/ThoughtTrace` (ungated, **CC-BY-4.0**, verified).
  Single file `ThoughtTrace.jsonl` (~30 MB), 2,155 rows = conversations from
  1,058 users across 20 models. Paper: arXiv 2605.20087; site:
  thoughttrace-project.github.io.
- **Fields** (verified via first-rows): `id` = `"user{N}_task{M}_conversation{K}"`
  → **author_id = id.split("_")[0]**; `messages` = list of JSON turns
  `{content, timestamp, type: "user"/"assistant", id, reasons:[{content,label}],
  reactions...}` — user turns carry self-reported *reasons* (e.g. label
  `context_grounding_and_constraints`); `task_summary`, `task_expectation`;
  `survey_answers` = `[{age, gender, education, occupation, frequency, purposes}]`.
- **Recipe**: explode user-type messages → prompts (first user turn per
  conversation to start). authors: gender=[gender.lower()], age=str, education,
  occupation→(no column; put in nothing), `frequency` (LLM-use frequency,
  "5") → ordinal into `llm_freq`, `purposes` → `domain`. Cohorts by gender or
  education (Undergraduate vs Graduate). `provenance="thoughttrace"`. No
  published separability benchmark (paper's numbers are about *thought*
  prediction: +41.7% user-behavior prediction) — expectation "".

### 8. OKCupid profiles (JSE) — BLOCKED for our purpose (skip + ethics note)

- **Repo**: https://github.com/rudeboybert/JSE_OkCupid (Kim & Escobedo-Land,
  JSE 2015). Files today: `profiles_revised.csv.zip` (1.2 MB; 59,946 users;
  sex, orientation, age(+noise), ethnicity, education, income, ... — **no
  essays**) and `essays_revised_and_shuffled.csv.zip` (52 MB; essay0-essay9).
- **The blocker** (verified in okcupid_codebook_revised.txt): *"the essay data
  has been randomized by rows to **decouple** them from the profiles data"* —
  i.e. essay text can no longer be joined to sex/orientation labels. The repo
  history was squashed to a single 2021 commit, so the pre-revision linked
  file is not recoverable from git. Unrevised copies float around (e.g.
  Kaggle `andrewmvd/okcupid-profiles`, registration required), but the
  maintainers decoupled the data *deliberately* for privacy; using a mirror
  would override that decision for exactly the identity-inference use they
  worried about. **Recommendation: skip; write this paragraph into the
  converter stub as the ethics note** (plan item 3 anticipated this).

### 9. WildChat-1M — FEASIBLE TODAY

- **HF id**: `allenai/WildChat-1M` — **not gated** (verified: anonymous
  `resolve/` request gets a public signed CDN URL; `gated: False` via API).
  License **ODC-BY** (changed from AI2 ImpACT 2024-06-26, retroactive). The
  toxic-inclusive `allenai/WildChat-1M-Full` *is* application-gated — don't use.
- **Size/format**: 14 parquet shards, 3.36 GB, **837,989 conversations** in
  the public release (card says "1 million"; toxic + journalist-flagged-PII
  conversations were removed 2024-07/2024-10 — document this delta).
- **Fields** (verified): `conversation_hash, model, timestamp, turn, language,
  conversation: [{content, role, country, state, hashed_ip, language,
  redacted, timestamp, turn_identifier, header{user-agent, accept-language}}]`,
  plus openai_moderation/detoxify scores. **author_id proxy = hashed_ip** of
  the first user turn (imperfect: IPs shift; good enough for author grouping).
- **Recipe**: download shard 0 only (`data/train-00000-of-00014.parquet`,
  ~240 MB) — it's row-random enough for subsampling. Keep first user turn
  where `language=="English"` and `redacted==False`; cohorts by
  `conversation[0].country` — e.g. target `India` vs baseline `United States`
  (large English-speaking cohorts → L1/variety signal), or user's plan:
  country as the generic y. `general_freq`/`llm_freq` stay 0.
  `provenance="wildchat"`. No published separability number (expectation "").
- ```python
  df = pd.read_parquet(hf_hub_download("allenai/WildChat-1M",
        "data/train-00000-of-00014.parquet", repo_type="dataset"))
  first = df.conversation.str[0]  # dicts with content/country/hashed_ip
  ```

### 10. TOEFL11 (LDC) — BLOCKED (document skip)

- **Catalog**: https://catalog.ldc.upenn.edu/LDC2014T06 — "ETS Corpus of
  Non-Native Written English": 12,100 TOEFL essays, 11 L1s (Arabic, Chinese,
  French, German, Hindi, Italian, Japanese, Korean, Spanish, Telugu, Turkish),
  8 prompts, low/medium/high proficiency, raw + tokenized UTF-8.
- **Gating**: requires an LDC account + signed license; fee visible only
  after login ("Login for the applicable fee") — not downloadable today.
  **Skip**; Reddit-L2 covers the native-language axis with public data.
- **Published sanity check** (for the doc): NLI Shared Task 2013 (Tetreault,
  Blanchard & Cahill, BEA-8) winning system **83.6%** 11-way accuracy (Jarvis
  et al., word/POS/lemma n-grams + L2-SVM); later cross-validation SOTA 85.2%
  (Malmasi & Dras 2017, meta-classifier).

### Published-numbers sanity table (for post-run comparison, item 3/8)

| dataset | task | published | source |
|---|---|---|---|
| blog_authorship | gender, 2-way | 80.1% | Schler et al. 2006 (AAAI SSS) |
| reddit_l2 | native vs non-native | 91.07% content-feats / 93.40% all-feats (in-domain) | Goldin et al. EMNLP 2018, Tables 2/7 |
| pan17 (en) | gender, 2-way | 82.33% | Rangel et al. CLEF 2017 overview, Table 3 |
| pan17 (en) | variety, 6-way | 90.04% | ibid., Table 4 |
| twitteraae | AA- vs White-aligned | near-definitional (labels = model posteriors) | Blodgett et al. EMNLP 2016 |
| multi_value | transformed vs original | tunable by rule density | Ziems et al. 2022/2023 |
| toefl11 | L1, 11-way | 83.6% (shared task) / 85.2% (CV SOTA) | Tetreault et al. 2013; Malmasi & Dras 2017 |
| prism / wildchat / thoughttrace | demographic axes | no standard benchmark — exploratory | — |

Caveats for expectation-setting: our pipeline unit is a *prompt* (short),
while blog/PAN/TOEFL numbers are per-author over 100 tweets or full posts —
per-prompt separability will run lower than the table; per-author aggregation
(mean over prompts, as our PromptSet groups by author) is the right comparison.


## Validation design (user spec, 2026-07-06 — binding for Iter4-E)

Notation: x = text unit, y ∈ {0,1} (1 = target), a = author ID.

### Roles
| Role | Datasets |
|---|---|
| Known-target positive control | Blog(1), Reddit-L2(2), TOEFL11(10, LDC-blocked) |
| Subtle-signal / coded-language analog | PAN17(4), Multi-VALUE(6), OKCupid(8, see caveat) |
| Modality match (real prompts) + power analysis | PRISM(3), ThoughtTrace(7), WildChat(9) |
| End-to-end smoke test (ALL modules must fire) | TwitterAAE(5) |
| Sensitivity calibration (tunable signal) | Multi-VALUE(6) |
| Negative control | random author-split of any single group -> expect C2ST≈0.5, 0 BH survivors, null MMD |

### Per-dataset requirements beyond conversion
1. **Blog**: x = post SEGMENT (report the unit); y = gender (also age-band contrast).
   C2ST should land near 80.1% (large deviation either way = pipeline problem, not
   discovery). Report topical JSD prominently: attribution of distinguishability to
   topic vs style is itself the finding (teens write about school).
2. **Reddit-L2**: y = native (UK/US flair) vs one non-native L1. Signal is
   grammatical/stylistic NOT topical (shared subreddits) — expect syntactic +
   semantic to fire, topical to stay comparatively low; 91% published ceiling,
   flair noise caps it.
3. **PRISM**: x = first-turn prompt; condition on conversation type (unguided /
   values / controversy) — at minimum restrict to one type or add type as a slice.
   POWER ANALYSIS: subsample to our expected n (e.g. 12/24/48 authors per side,
   ~4 prompts each) as extra manifest comparisons; check MMD/C2ST significance
   retention. Exploratory (no published target).
4. **PAN17**: EN only (PT is near-trivial orthography). x = tweet (matches prompt
   granularity; note published 71.5% is per-FEED — expect per-tweet numbers lower).
   y = variety pair (US vs GB) and gender. Best coded-language credibility test.
5. **TwitterAAE**: posterior ≥0.8 (and/or 0.999) subsets; y=AAE vs White-aligned.
   CIRCULAR by construction — smoke test ONLY: every module must fire (many BH
   survivors, huge MMD, C2ST >> 0.5); a silent module is broken. Document that
   success says NOTHING about detecting self-reported identity.
6. **Multi-VALUE**: transform a neutral corpus (can reuse baseline cohort texts);
   y = transformed?. Cohorts at rule-application density p ∈ {0.05,0.1,0.25,0.5,1.0}
   -> one comparison per p -> SENSITIVITY CURVE plot (detection p-value/accuracy vs
   feature density, per module). Exact labels; validates sensitivity not realism.
7. **ThoughtTrace**: x = user turn; y = gender or median-split age/education.
   Same design as PRISM; smaller, observer-aware; no benchmark yet.
8. **OKCupid**: DIRECT analog of the research question (orientation, coded
   register). CAVEAT from our own verification: the permissioned JSE 2021 revision
   deliberately row-shuffled essays to decouple them from sex/orientation — the
   linkage the analysis needs is intentionally absent; the unshuffled variant is
   the unethical scrape. Surface to user; implement converter only if a
   permissioned linked release is found. Do NOT use the Kirkegaard scrape.
9. **WildChat**: EN-only prompts (else language is the trivial signal); y = country
   pair; hashed IP ≈ a (imperfect). Use for scale/asymptotics (MMD power, C2ST
   convergence) — large-n subsets.
10. **TOEFL11**: LDC-blocked; documented skip (Reddit-L2 covers the L1 axis).

### Converter contract additions
- Every converted dataset's manifest MUST include a negative-control comparison:
  random author-split of ONE group (seeded), expectation "null".
- ComparisonSpec gains optional fields: expected_accuracy (published separability
  target, 0.0 = none) and notes (role + caveats, e.g. TwitterAAE circularity).
- Real datasets leave z/d/c unannotated where unknown (0/""); usage/topical/slices
  must skip gracefully; markedness/codedness default 0 -> implicit/ explorations
  auto-skip (record reason).
- Subsample with author structure preserved (~60-100 authors/side, <=4 x-units
  each) EXCEPT WildChat scale runs; document every sampling choice in the
  dataset's README.

## Analysis predictions (write-ahead, to check against real runs)

Per-dataset expected pattern (from the papers + validation roles):
- reddit_l2 (German L1 vs native; Goldin 91%): signal is GRAMMATICAL not topical.
  Expect syntactic STRONG, semantic/distributional strong (C2ST high but < 91%:
  per-chunk not per-author + flair noise), lexical MODERATE (L1-transfer words),
  topical LOW/near-null (shared subreddits). If topical fires strongly -> topic
  leakage / bug.
- blog_authorship (F vs M; Schler 80.1% per-author): per-CHUNK C2ST should land
  BELOW 0.80. Topic correlates with gender -> topical fires; lexical fires; the
  topic-vs-style attribution is the finding.
- pan17_variety (US vs GB tweets; 71.5% per-FEED): hardest positive control.
  per-TWEET signal weaker; expect lexical catches regional spelling, but
  semantic/distributional per-tweet modest/near-chance. Honest if some tests
  don't clear alpha.
- twitteraae (AAE vs White-aligned, posterior>=0.8; CIRCULAR): smoke test —
  EVERY module must fire strongly (many marked words, huge MMD, C2ST>>0.5). A
  silent module = bug.
- multi_value_aave (exact y, density p): SENSITIVITY CURVE — detection should
  rise monotonically with p; find the p where each module first clears alpha.
- prism/thoughttrace (gold gender, real prompts, no benchmark): exploratory;
  modest signal expected; power-analysis cohorts (n12/n24) show significance
  loss at small n.
- wildchat (country proxy, EN-only): topic-driven country differences; proxy
  label caps it.
- ALL null_control / null_split comparisons: must be null (C2ST~0.5, 0 marked
  words, null MMD). Any significant = false positive to investigate.

## Iteration 4: COMPLETE (verified)

All 10 points delivered and independently verified:
1. implicit/ (codedness sweep + markedness splits) ✓
2. Lexical calibration overhaul (MotS Algorithm 3): register words no longer
   flag (verifier: i'm z=1.75 n.s., all of {i'm,a,do,give,work} not BH-sig,
   content signatures all sig); calibration plots (null QQ, constant sweep,
   rank); word clouds; per-slice marked words; BH confirmed
   (scipy false_discovery_control); raw-z always reported ✓
3. 8 real datasets imported + run + analyzed vs published (docs/FINDINGS.md);
   PAN-17 C2ST 0.71-0.81 matches published 71.5% ✓
4. Syntactic per-group histograms + palette pass ✓
5. Cohere active (verifier: 1536-dim, not skipped, all 6 semantic variants) ✓
6. Proper TopicGPT (8 orthogonal intent-level topics generated) ✓
7. slices/{gender,race,age,transgender}/{value}/ full outputs ✓
8. Analysis pass with write-ahead predictions; module-sensitivity findings ✓
9. comparison_matrix.png (target_vs_baseline vs target_vs_twin, 25 tests) ✓
10. Full verification: verifier 62+ checks 0 failures, all recomputed ✓

Verifier verdict: "All artifacts independently verified. The specific
regression (register word BH-significant) is absent."

## Re-verification pass (comprehensive, adversarial)

Three independent verifier agents recomputed from raw data (no reported values trusted):
- **recompute-real (VERIFIED, 10 checks):** real-dataset C2ST recomputed from
  parquet matches FINDINGS exactly — PAN 0.7141 (published 0.715), Reddit 0.6484,
  Blog 0.5705, TwitterAAE 0.6937; all 4 null controls 0.45-0.50 n.s.; null halves
  author-disjoint; topical JSD hand-recomputed (scipy jensenshannon**2 base2) exact.
- **determinism-audit (VERIFIED, 15 checks):** every section byte-identical across
  two same-seed runs (RNG seeded); permutation/BH/GroupKFold correctness confirmed;
  auto-calibration binary search monotone-correct; all 5 prior bug-fixes present.
- **schema-config (MIXED -> now fixed):** all 7 result schemas round-trip; no NaN/Inf
  leaks (allow_nan=False clean); found configs/config.json had DRIFTED from
  _default_slices() after the slice bug-fixes (invariant "config.json == defaults"
  broke on 3 slice values). REGENERATED config.json; invariant re-verified.

## Deep verification of previously-unverified paths (executed/audited)

Two workflows re-derived the gaps I could not previously vouch for:
- CONVERTER FIDELITY (8 datasets re-derived from raw source): all 8 faithful;
  1 minor doc gap (interior-whitespace collapse undocumented in 4 READMEs) — fixed.
- EXECUTION/METHODOLOGY AUDIT (ran or read-audited each path): 8 confirmed bugs,
  all fixed + verified by execution:
  * annotation module (never run before): silent label MISALIGNMENT (order-trust
    -> indexed-id binding + reindex + reject mismatch); content=None crash; no
    retry -> corrective retry. Live-verified 4/4.
  * ModernBERT C2ST non-deterministic (unseeded fold init) -> torch.manual_seed;
    0.8958==0.8958 across runs.
  * C2ST chance hard-coded 0.5 -> majority-class (PRISM 0.5162).
  * MMD-Fuse alpha not threaded (rejected vs significant could diverge) -> fixed.
  * vast_destroy/vast_launch read only first paginated page -> full page walk.
GPU path: cloud scripts read-audited (pagination fixes applied); a live paid run
was NOT performed (costs against the ~$3.6 balance) — flagged to the user.

## Paper conformance (read from source PDF, 2026-07-07) + conditional distinguishability

CONFORMANCE (§3.3 methods vs our impl): all methods match the paper's spec.
- 3.3.1 lexical: Calibrated Marked Words (Mickel) + BH — differentially verified
  vs the MotS reference (r=0.99996). ✓
- 3.3.2 syntactic: NeuroBiber, NO calibration, simple log-odds — ✓ (ground-truthed).
- 3.3.3 semantic: MMD-Fuse; text emb (OpenAI 3-small + Cohere embed-4) + residual
  stream at change-of-turn, layer at 75% depth, AVERAGE over token positions;
  sentinel <|im_end|>\n<|im_start|>assistant — ✓ (ground-truthed layer+positions).
  DEVIATION: paper uses Qwen3.6-27B/Llama3.3-70B/Gemma4-31B; we use 1.7B/1B/2B
  (local compute). Method identical; model scale differs — documented.
- 3.3.4 distributional: C2ST, linear + ModernBERT, author-level CV + author-label
  permutation (up to 15 prompts/author) — ✓ (calibration bug found+fixed).
- 3.3.5 topical: TopicGPT + JSD — ✓ (orthogonal seeds address the paper's own open
  margin note "[Should we define orthogonal topics?]").
- 3.3.6 interactional: speech act / disclosure depth / anthropomorphization — ✓.

CONDITIONAL DISTINGUISHABILITY (paper §6 Discussion names it; user directive):
Marginal (aggregate) distinguishability pools all prompts. Conditional
distinguishability measures distinguishability WITHIN strata of a content
variable Z (c^dom domain MH/GSH/REL by default; also c^top topic, c^prov). This
separates "what they talk about" (topic choice) from "how they talk about it"
(coded style) — the H1/H2 thesis. Two regimes:
  - marginal separable but conditional ~null  => difference is topic-choice.
  - conditional separable (survives) or conditional > marginal (Simpson: hidden
    separability revealed by conditioning) => genuine coded signal beyond topic.
Aggregation per (test, variant) verdict:
  - conditional_statistic = Σ_z (n_z/N) · statistic_z   (stratum-size weighted).
  - conditional_p = Fisher combine of per-stratum p (-2 Σ ln p_z ~ χ²_{2k}).
  - n_significant_strata; interpretation label vs the marginal verdict.
Each dimension gets conditional/{variable}/ (per-stratum full outputs, like
slices) + an aggregate; run root gets aggregate-vs-conditional juxtaposition.
