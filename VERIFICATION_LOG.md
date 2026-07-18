# Verification Log

## 2026-07-17 — Reuse-map transitive-closure check against base repo (queering-nlp-bias)

WHAT: Missing-link dependency claims for the copy/adapt reuse plan (files in
/Users/unrulyabstractions/work/queering-nlp-bias referenced by copy/adapt-verdict files
but absent from the reuse set).

HOW: Ran `ls` on every claimed missing/unreviewed file and `grep -n`/`grep -ln` on every
consumer file to confirm the actual import lines (not relying on reader summaries).
Confirmed directly:
- torch at module level: base_schema.py:11, random_seed.py:6, num_types.py:12,
  entropy_diversity/core_impl.py:15, device_utils.py:9 — VERIFIED
- device_utils imported by profiling_timer.py:34, profiling_decorators.py:9,
  model_runner.py:11, scoring_pipeline.py:19 — VERIFIED
- src/common/text is a PACKAGE (eos_handling.py, thinking_filter.py, text_display.py),
  not a file; lazily imported by scorer.py:36 and run_full_experiment.py:277 — VERIFIED
  (reader maps had listed it as a single module; corrected)
- generation_method_registry.py imports ArmGenerationResult from experiment_types
  (lines 26, 55) despite experiment_types being skip-verdict — VERIFIED
- CHECKPOINT_DIR = Path("/tmp/scoring_checkpoints") at scoring_pipeline.py:39 — VERIFIED
- auto_export used by logging/__init__.py, profiler/__init__.py, backends/__init__.py — VERIFIED
- Existence of unreviewed deps: token_tree.py, token_trajectory.py, logit_kde.py
  (imported at judgment_scoring_helpers.py:17), greedy_output.py, greedy_paths.py,
  gen_logging_utils.py, sample_compliance_data.py (imported at
  test_scoring_pipeline.py:8), tests/integration/conftest.py, tests/unit/conftest.py,
  estimate_normativity.py — VERIFIED
- skip-verdict files imported by adapt-verdict files (generated_trajectory,
  api_tokenizer, embedding_runner, scoring_data, generation_helpers at
  generation_output.py:197/212, default_config in 6 consumers) — VERIFIED

RESULT: VERIFIED (all missing-link claims in the final structured answer were confirmed
against the filesystem). NOT independently re-verified: the per-file reuse verdicts and
line counts reported by the upstream readers (taken as input); the contents of the
unreviewed files above (existence confirmed only — flagged as must-read-before-copy in
the final answer).

## 2026-07-17 — Ported common layer matches paper's worked example

WHAT: `src/dtreat/common/` (base_schema, file_io, auto_export, random_seed,
console_logging, discrete_information) + package auto-export wiring.

HOW: `uv sync` then executed `import dtreat` in a live interpreter and called the
information measures on the paper's §4.5 worked example; observed stdout directly:
D_pi = 2.37 bits (paper: 2.37), I_1 = 0.46, I_3 = 0.19, I_4 = 0.08 (paper values
0.46/0.19/0.08); auto-export resolved 42 symbols.

RESULT: VERIFIED (information math matches the paper's worked example exactly).
NOT yet verified: BaseSchema roundtrip serialization under the numpy adaptation —
covered by unit tests later in this build.

## 2026-07-17 — LLM client layer live exercise

WHAT: `src/dtreat/llm/` (chat types, backends incl. mock with planted bias,
retry, pricing, ChatClient cache/trace/stats, parallel executor).

HOW: Ran a live script through `uv run`: mock target produced deterministic
output twice for same (prompt, seed); cued vs baseline prompts produced visibly
different planted behaviors (observed stdout: cued → "worth asking whether this
goal", baseline → "300 calories over maintenance"); mock helper returned the 5
case-study axes as parseable JSON; mock judge parsed the sentinel protocol and
returned correct YES verdicts; cache hits = 3 exactly as predicted (r2, r3, one
parallel job sharing seed=1); 24 trace lines for 24 calls; 20 parallel jobs, 0
failures.

RESULT: VERIFIED (mock path). NOT verified: anthropic/openai backends against
live APIs (no calls made yet — covered by L2 smoke tests later).

## 2026-07-17 — End-to-end mock experiments (biased + null)

WHAT: Full 5-stage pipeline runs `out/runs/mock_biased/` and `out/runs/mock_null/`
(240 responses each, 5 axes, 2000 prompt-level permutations).

HOW: Ran `uv run dtreat run-all` for both configs and READ the actual
`analysis_summary.md` artifact for mock_biased in full (re-opened the file):
- biased: 4/5 axes significant, exactly the 4 planted-biased axes; planted-null
  `mention_sleep` correctly NOT significant (p=0.065); all Δ signs match the
  planted profile (gives_number −0.48, reconsider_goal +0.48, warn_fat +0.26,
  recommend_purchase +0.22); D_pi(sig)=0.54 bits; C2ST 0.833 [0.731, 0.902]
  vs 0.5 majority; refusals 0/120 both sides.
- null: observed stdout 0/5 significant, C2ST 0.486 ≈ chance (summary file not
  separately re-opened — stdout only).
- `dtreat validate` on both runs: 0 problems, 0 warnings (observed output).
Note: target rates attenuate toward baseline (0.46 vs planted 0.30) because
some target prompts carry no cue words — expected cue-detection realism, not a
bug; direction/significance recovery unaffected.

RESULT: VERIFIED for mock_biased artifacts (read directly) and validation
outputs; mock_null summary VERIFIED via stdout + validate only.

## 2026-07-17 — Diagnostics CLI + debug server + UI

WHAT: `dtreat status/inspect/trace/estimate-cost`, FastAPI debug server
(12 endpoints), static UI (7 tabs with SVG charts).

HOW: Executed each CLI command and observed correct output (status shows all 6
artifacts; inspect computed per-axis community rates matching the analysis;
trace aggregated 481 calls; estimate-cost projected $0.04 for live_smoke).
Started `dtreat serve`, curl-checked all 12 endpoints → all HTTP 200 with real
payloads (permutation-null endpoint returned 2000-perm null deltas).
VIEWED actual headless-Chrome screenshots with image tokens for tabs:
overview shell, stage1 (matched instruction bars, TV=0 tile), stage3 (response
browser + filters), stage4 (rate bars + prompt×axis heatmap), stage5 (diverging
Δ chart with ns-outline, MI ranking, full table, markdown). Palette validated
with the dataviz validator: 2-series categorical passes ALL checks light+dark
(CVD ΔE 26.5, normal 29.0). Fixed two visual defects found in screenshots
(Δ-label collision on long negative bars; heatmap column-label clipping) and
re-verified stage5 by screenshot after the fix.

RESULT: VERIFIED (stage2/trace/overview tabs rendered but reviewed less
closely; heatmap clip fix verified only indirectly via code — stage4 not
re-screenshot after the margin change).

## 2026-07-17 — Test suite at graded realism + live experiment

WHAT: 56 tests (unit + L0 mock e2e + L1 synthetic stats + L2 live smoke) and
the full live experiment `out/runs/live_smoke/` (gpt-4o-mini, ~$0.03).

HOW: Ran `uv run pytest -q` (observed: 53 passed, 3 deselected) and
`uv run pytest tests/integration -v` (observed all 15 integration tests
individually PASSED, including planted-bias recovery, null-stays-quiet, noisy
judge, per-axis judge mode, resumability, FDR calibration, power curve, C2ST
calibration). Ran `uv run pytest -m live` with real keys: 3 passed (OpenAI
helper axes parse, OpenAI judge protocol verdicts exactly correct, Anthropic
completion + cost accounting). Ran `dtreat run-all -c configs/live_smoke.json`
and READ the resulting analysis_summary.md in full: 96 real responses, 0
unparsed verdicts, 0/5 axes significant at this scale, C2ST 0.517 ≈ chance —
internally consistent and honestly non-significant.

RESULT: VERIFIED.

## 2026-07-17 — Independent verifier agent over all experiment artifacts

WHAT: Adversarial re-verification of mock_biased, mock_null, live_smoke runs,
the paper's worked example, and the test suite, by a separate verifier agent
recomputing everything from raw artifacts.

HOW (as reported by the verifier, with its own computed numbers): recomputed
per-community rates from raw verdict JSONL for all axes in all three runs —
exact matches to reports (e.g. gives_number 0.45833/0.93333); delta
consistency within 5e-8; recomputed D_pi(sig)=0.53965128 = reported;
D_pi(all)=0.42281253 = reported; mock_null 0 significant, C2ST 0.48611;
live_smoke all-96 records model gpt-4o-mini, min response length 968 chars,
read 3 real response texts; paper example KL=2.367251; pytest 53 passed.

FINDING → FIX: verifier found `c2st.above_chance` serialized as string "True"
(numpy bool_ leaking through _canon). Fixed base_schema._canon (np.bool_ →
bool) + explicit float casts in run_c2st; re-ran `dtreat analyze` for all
three runs; re-opened all three reports programmatically and observed
above_chance now True/False/False as proper JSON booleans. ruff: all checks
passed; pytest re-run: 53 passed.

RESULT: VERIFIED (all three runs, math, and tests independently confirmed;
the one defect found was fixed and the fix re-verified against all three
regenerated artifacts).

## 2026-07-17 — Hourly audit #1 (scheduled iteration)

WHAT: Full-repo stability audit + closure of the two items earlier marked
not-fully-verified (mock_null summary artifact; stage-4 heatmap after the
column-label margin fix).

HOW: `uv run pytest -q` observed 53 passed / 3 deselected; `ruff check`
observed "All checks passed"; debug server responding HTTP 200. READ
mock_null analysis_summary.md in full: 0/5 significant, D_pi(all) = 0.00
bits, C2ST 0.486 ≈ chance, refusals 0 — coherent null artifact. VIEWED a
fresh stage-4 screenshot with image tokens: all five heatmap column labels
fully visible, no clipping; per-axis rate bars and tiles correct. Paper
section-by-section coverage re-checked against the implementation (Eq 1–14,
§4.1–4.5.3 all implemented; §5.2 modular participation supported via
hand-authored stage artifacts; §5.3 judge calibration is future work in the
paper, not pipeline scope). TaskList: all 14 tasks completed.

RESULT: VERIFIED — audit #1 clean, no gaps found. One more clean audit
before stopping the loop per the two-consecutive-stable rule.

## 2026-07-17 — Hourly audit #2 (final)

WHAT: Stability re-audit + cold-start determinism check.

HOW: `uv run pytest -q` observed 53 passed / 3 deselected; `ruff check`
observed "All checks passed"; ran `dtreat run-all` into a FRESH run dir
(out/runs/audit2_cold, deleted after the check) and observed stdout identical
to the verified mock_biased results — 4/5 significant axes, D_pi 0.54 bits,
C2ST 0.833 — confirming the pipeline is deterministic end to end;
`dtreat validate` on the fresh run: 0 problems, 0 warnings; debug server
HTTP 200. TaskList empty.

RESULT: VERIFIED — two consecutive clean audits; iteration loop stopped.

## 2026-07-17 — Instruction extraction/matching + judge panel + calibration

WHAT: New stage-1 extraction (`annotate_instructions: extract`) + frequency
matching; stage-4 multi-judge panels with rubrics; `dtreat calibrate-judge`.

HOW: Ran `dtreat run-all -c configs/mock_panel.json` and observed stdout
directly: extraction with mock:annotator, matching kept 21/side and dropped 6
prompts (recorded), TV distance 0.000 exactly, comparability PASSED; panel of
2 judges scored 210 responses. Ran `dtreat calibrate-judge`: observed
mock:judge vs mock:judge:noisy raw agreement 0.954, Cohen kappa 0.908
(n=1050) — matching the planted 5% flip noise model (expected raw ≈ 0.95);
self-consistency flip rates 0.000 (deterministic judge) and 0.060 (noisy,
consistent with 2·0.05·0.95 ≈ 0.095 over 50 verdicts). Added 16 unit tests
(Cohen kappa textbook value 0.4 exact, Fleiss cases, aggregation rules,
matching determinism) + 2 integration tests — observed 71 passed total.

RESULT: VERIFIED (mock path, artifacts + stdout observed). NOT yet verified:
panel behavior with real API judges; distinguish bridge (in progress).

## 2026-07-17 — distinguish/ bridge + input-vs-output comparison E2E

WHAT: Vendored distinguish/ pipeline bridged from stage 1; IO comparison in
stage 5 + UI.

HOW: Ran `dtreat distinguish -c configs/mock_panel.json` (observed stdout:
20 input tests, 3 significant, best input C2ST 0.881, 0 skipped variants,
artifact written); re-ran `dtreat analyze` and READ the summary's new
"Input legibility vs output treatment" section: input 0.881 → output 0.760,
signal usage 68%, coherent interpretation. VIEWED the stage-5 UI screenshot
with image tokens: IO card renders with paired bars + interpretation; also
viewed stage-4 panel screenshot (48 tie verdicts honestly surfaced, matching
the noisy judge's 5% flip rate). One export bug found and fixed en route
(topic_id must be an integer index; usage ints use 0 sentinel).

RESULT: VERIFIED.

## 2026-07-17 — Adversarial review workflow (64 agents) + fixes

WHAT: 5-dimension review (correctness/interfaces/duplication/conventions/
test-gaps) with per-finding adversarial verification; 57 confirmed findings.

FIXES APPLIED + how each was verified:
- load_json corruption (HIGH): repair regexes now run only after parse
  failure. Regression test executes both directions (valid content with
  ",,"/", ]" preserved byte-for-byte; genuinely broken file still repaired).
- Stage-4 resumability (HIGH): empty-verdict records no longer freeze
  responses out of the audit; regression test plants a poisoned artifact and
  observes the response re-judged (this test also exposed + fixed a mock
  judge protocol bug: bare YES for single-axis per_response prompts).
- stages↔pipeline package cycle (HIGH): experiment_config + run_directory_paths
  moved to dtreat/common; auto_export now WARNS on import failure instead of
  silently swallowing — which immediately caught two leftover relative
  imports during verification.
- Frequency-matching empty guard, cross-set prompt_id uniqueness, path-
  traversal prefix bypass (test: runs-evil sibling rejected), C2ST tiny-class
  crash (test), calibration fake-perfection on pre-panel artifacts (note
  instead), cost-estimate extract mode, cache tmp-file race, seed-hypothesis
  precedence, provider-prefixed pricing, stale quarantine cleanup, dead code
  removal (6 functions, zero callers confirmed by grep), CLAUDE.md/README
  sync.
- Deferred (logged, not fixed): arg-object refactors for long signatures,
  server/diagnostics dedup, file splits over 150 lines, remaining test-gap
  suggestions.

HOW VERIFIED: 78 tests pass (7 new regression tests), ruff clean, and
`dtreat run-all -c configs/mock_biased.json` reproduces the previously
verified numbers bit-identically (4/5 significant, D_pi 0.54 bits, C2ST
0.833) — observed stdout.

RESULT: VERIFIED.

## 2026-07-17 — Real-data experiment (real Reddit prompts, live models)

WHAT: `out/runs/real_reddit/` — 12+14 real community prompts (PullPush,
curated + verified permalinks, LOCAL ONLY/gitignored, checked with
git check-ignore + status), gpt-4o-mini target/helper/annotator,
gpt-4o-mini + gpt-4.1-mini judge panel, extraction + frequency matching,
distinguish bridge, calibration. Total cost ≈ $0.06.

HOW + findings (artifacts READ directly):
- First run FAILED CORRECTLY: canonicalizer produced 26 ids for 26 phrases
  (no merging), matching emptied both sets, and the new guard aborted stage 1
  with a diagnostic — the review fix caught a real failure the same day.
  Fixed by giving canonicalization a hard group budget
  (max_instruction_groups=8) + aggressive-merge instructions; re-run kept
  7 prompts/side at TV=0.
- Full pipeline: 42 responses, 0 refusals; panel scored with 30 tie/unparsed
  verdicts (12% — real judges disagree more than mocks); READ
  analysis_summary.md in full: 0/6 axes significant at FDR 0.05; largest
  trends customized_advice Δ=−0.51 (p=0.078) and long_term_focus Δ=−0.55
  (p=0.030, q=0.183) — target-community prompts trend toward LESS customized,
  LESS sustainability-focused advice, correctly not claimed significant at
  n=7 prompts/side; C2ST 0.667 [0.300, 0.903] → not above chance.
- Calibration (observed stdout): gpt-4o-mini vs gpt-4.1-mini raw 0.881,
  Cohen kappa 0.701 (n=252); flip rates 0.000 / 0.021 at temperature 0.
- Distinguish bridge (observed stdout + summary READ): input C2ST 0.714,
  5/20 input tests significant; IO comparison: 78% signal usage.
- FIX during verification: IO interpretation overclaimed "the model acts on
  it" from point estimates; added an output-evidence gate — re-ran analyze
  and READ the regenerated hedged interpretation ("collect more prompts
  before drawing conclusions").

RESULT: VERIFIED by direct artifact reads + observed stdout; independent
verifier agent recomputation in progress (result logged when it returns).

## 2026-07-17 — Independent verifier recomputation of real_reddit (final)

WHAT: All six verification items for out/runs/real_reddit, recomputed from
raw artifacts by a separate verifier agent (resumed after a transient API
drop mid-run).

HOW (verifier's computed numbers): stage-1 kept 7/side, instruction multisets
identical, TV = 0.0 recomputed, kept ∪ dropped exactly equals the 26
collected ids; 42 responses = 14×3, all gpt-4o-mini, texts 1551–2144 chars,
spot-read as real advice; 252 judge-verdict pairs complete for both judges,
pooled raw agreement 222/252 = 0.880952 and Cohen kappa 0.701493 recomputed
exactly (marginal-product pe = 0.60119); aggregation rule exact over all 252
pairs (agreement→verdict, disagreement→unparsed, 0 mismatches); all 6 axes'
n/rates/deltas match the report to <1e-6 (customized_advice 6/14=0.42857 vs
16/17=0.94118; long_term_focus 4/19=0.21053 vs 13/17=0.76471); 0/6
significant (min q=0.183); signal usage arithmetic checks
((0.6667−0.5)/(0.7143−0.5)=0.7778); interpretation confirmed hedged;
input report 5/20 tests significant, best C2ST 0.71428571 cross-checked
against the distinguish run's own distributional.json; real_reddit_*.json
confirmed untracked (git ls-files + check-ignore:24).
Nuance noted: input_distinguishability.json holds 26 verdict entries of
which 20 are tests with p-values (n_tests=20); 6 (modernbert + usage scales)
carry no p-value.

RESULT: VERIFIED — second consecutive clean audit; iteration loop stopped.

## 2026-07-17 — Scaled real-data study (out/runs/real_study) + independent verification

WHAT: The full research study on 60 real prompts/community: 4 helper
conditions (24 union axes), 216 responses, 2-judge panel scoring, 5-judge
cross-provider judge study, calibration, input-side distinguishability, IO
comparison. Two infrastructure defects found by study v1 and fixed
(canonicalization token truncation collapsing matching to 5/60 per side;
Gemini thinking starving judge replies to ~27 tokens).

HOW: Ran the suite twice (v1 diagnosed via llm_trace.jsonl: canonicalization
output = exactly its 1500-token cap; Gemini previews showed valid JSON cut at
~27 output tokens; claude-3-5-haiku 404 = retired). v2 observed stdout + READ
analysis_summary.md in full + extracted judge_study.json numbers. Independent
verifier agent then recomputed EVERY claim from raw artifacts: matching
36/side with TV = 0.0 exact and 48 drops accounted; union 24 axes with 1:1
condition mapping and both collision renames confirmed; 216 = 72×3 responses;
panel raw agreement 4727/5184 = 0.911844 and kappa 0.743630 exact (sklearn
cross-check identical); tailored_advice +0.340909, identity_integration
+0.272727, practical_advice −0.095874 to full precision; exactly 21
significant, all positive; C2ST block consistent (158 dropped incomplete);
judge-study intersection = 12 axes exact, kappa matrix 10 pairs (0.6140 min
gpt-4o-mini↔gemini-3.5-flash, 0.7882 max gpt-4.1-mini↔claude-haiku-4-5);
input side 13/20 significant, best C2ST 0.88888889 cross-checked against the
distinguish run's own distributional.json.
Verifier nuance adopted: manifest `unparsed_verdicts` is panel TIES, not
parse failures — schema docstring + UI label clarified.

RESULT: VERIFIED (all seven verification items, exact recomputation).

## 2026-07-17 — Cross-group pair runs (women_vs_men, over40_vs_young) + incident

WHAT: Two new group pairs collected (60/side each) and run end-to-end in the
same fitness domain; cross-run comparison; independent verifier over both.

VERIFIER RESULTS (recomputed from raw artifacts): over40_vs_young fully
VERIFIED (40/side TV=0.0; 240=80×3 responses; kappa 0.84313454 exact; 0/8
significant; C2ST 0.54166667 not above chance; input 7/20 significant, best
0.6125). women_vs_men stages 1/3/4 + input side VERIFIED (kappa 0.84447932
exact; input 10/20, best 0.7375); its ORIGINAL 8-axis stage-5 report (0/8
significant; behavior C2ST 0.700, ci_low 0.54570 > majority 0.50376,
above_chance TRUE) was verified on disk by the verifier BEFORE being
overwritten.

INCIDENT (my fault): I launched a helper-study union re-run on the same run
dir while the verifier was reading it, destroying the 8-axis stage-5 report
(out/ has no git backup). The verifier's report preserves the verified
numbers. Guards added: helper-study now archives replaced artifacts before
overwriting; and I will not mutate a run under active verification again.

BUG FOUND BY THE INCIDENT + FIXED: stage-4 resumability treated stored
records as valid for a CHANGED axis set (helper-study union axes were never
judged — 24/32 axes had zero data, three conditions showed info=0.000
exactly). A first manifest-comparison fix failed because the buggy
intermediate run had already overwritten the manifest with the new axis list;
the durable fix is per-record axis-coverage (a record is reusable only if
verdicts ∪ tie-list cover the current axes). Re-run verified: observed
"240 stored records do not cover the current axis set — re-judging", then
32/32 axes with data. Corrected women union results READ from the report:
0/32 significant (best q=0.434: emphasis_on_recovery Δ=−0.18,
personalized_affirmations Δ=−0.05); union C2ST unreliable (212/240 rows
dropped by ties) → signal-usage now gated on output evidence (a noisy C2ST
had produced a nonsense 164% ratio; comparison now shows usage only for
runs with real output evidence).

RESULT: over40 run VERIFIED; women run VERIFIED across stages with the
8-axis stage-5 conclusions preserved via the verifier's direct observation
and the archived comparison; corrected 32-axis artifacts verified by direct
read; 78 tests + lint green after all fixes.

## 2026-07-18 — Pipeline correction round + full-method women run

WHAT: Removed the counterfactual arm entirely (user correction: implement the
paper's pipeline ONLY); restored Fig-1 semantics — first as a strict chain,
then per user clarification as the DAG it is (stages 2∥3 join at 4), with
responses executed before hypotheses so behavior-grounded generation observes
real responses. Stage 2 now runs ALL generation methods by default
(a-priori/zero_context, two_stage, literature bundled, literature_rag with
live arXiv retrieval, prompt-grounded, response_grounded, seeds) with
multi-source provenance and a per-method table in every analysis report.
Helper prompts now include the instruction-type inventory (§4.2 fidelity).

HOW: Mock run-all observed executing 1→3→2→4→5 with response_grounded
producing axes on a fresh run; RAG retrieval live-tested (top hit for the
lgbtq/fitness pair: WinoQueer — relevant; earlier naive query returned
particle physics and was fixed); 78 tests + ruff green. Real women_vs_men
full-method run READ from artifacts: 48-axis union; per-method table shows
response_grounded leading (0.177 bits ≈ 2× literature_rag 0.093, ≈3×
a-priori methods); best candidate axis technical_language (proposed only by
response_grounded): Δ=−0.17, p=0.0020, q=0.096 under 48-way FDR — the
strongest women-pair signal to date, still shy of 0.05.

RESULT: VERIFIED (mock E2E + artifacts read directly; live run numbers from
its own report).

## 2026-07-18 — Polish iterations (deadline-bounded loop)

WHAT: Stage polish round: camelCase axis-id fix, fuzzy + stemmed union
dedup, C2ST midpoint tie-fill, helper diversity rule, judge-prompt
experiment, JSON missing-comma repair, UI provenance cards.

HOW + findings (all observed directly):
- Judge-prompt experiment vs exact mock ground truth (320 real-judge
  judgments): baseline protocol 98.5% accuracy (gpt-4o-mini), fancier
  variants equal or worse, 0 unparsed anywhere — protocol validated by
  measurement.
- Fresh women run: C2ST midpoint fill drops 0 rows (was 158+); union-level
  behavior C2ST 0.689 [0.577, 0.783] ABOVE CHANCE — the gender pair's diffuse
  differentiation is now measurable at distribution level.
- Live-caught bug: literature_rag produced 0 axes because the model omitted
  a JSON comma; extractor now repairs missing commas at value boundaries
  (regression test with the exact pattern; the actual cached reply re-parses
  to 8 entries); zero-parse warning added.
- Complete 6-method run after the fix: response_grounded leads AGAIN
  (0.167 bits vs 0.177 in the prior independent axis draw — the
  behavior-grounded advantage REPLICATES at ~2-3× a-priori methods);
  union C2ST 0.757; still 0 significant single axes across ~90 distinct
  axes tried over three draws — women/men differentiation is robustly
  diffuse for gpt-4o-mini.
- Dedup now stems inflections (recommendation_of_supplements ~
  recommends_supplements merge; unit-tested). 80 tests + lint green.
- UI provenance verified by screenshot: multi-source chips show cross-method
  merges (recommendation_of_supplements: zero_context + literature_rag);
  46-axis Δ chart renders; label gutter widened and RE-VERIFIED by screenshot
  (full ids untruncated). Summary readability: ns tail collapses (women
  summary: 8 table rows + "…and 38 further non-significant axes" line,
  observed in the regenerated artifact).

RESULT: VERIFIED.
## 2026-07-17 — Web research: seed prompts for fitness/nutrition bias-audit dataset
- WHAT: Verbatim Reddit question prompts (LGBTQ+ vs general/cis-het fitness communities) collected for the report delivered in-chat (not written to a file). Sources: PullPush API (api.pullpush.io) queries over r/FTMFitness, r/askgaybros, r/gaybros, r/MtF, r/butchlesbians, r/actuallesbians, r/beginnerfitness, r/naturalbodybuilding, r/gainit, r/GYM, r/Fitness, r/bodybuilding.
- HOW: For ~20 quotes used in the report, re-fetched the raw PullPush JSON with curl and printed title/selftext/permalink directly (bypassing WebFetch's summarizer model) and compared strings: FTMFitness (5 posts), askgaybros (5), beginnerfitness (4), MtF (3), butchlesbians (3), gainit (1), naturalbodybuilding (2). All matched verbatim. RESULT: VERIFIED.
- Quotes taken only via WebFetch extraction (small-model mediated, not curl-checked): r/gaybros (2), r/bodybuilding (1), r/GYM Arnold-split (1), r/Fitness titles, r/gainit girlfriend-cue posts. RESULT: UNVERIFIED as exact-verbatim (flagged as such in report).
- Claims about Reddit API terms/robots.txt/Reddit-for-Researchers, WildChat/LMSYS licenses: from search-result snippets (TechCrunch, support.reddithelp.com, HuggingFace listings); reddit.com/redditinc.com/fitness.stackexchange.com not fetchable from this environment. RESULT: UNVERIFIED at primary-source level (flagged in report).

## 2026-07-17 — Correctness-bug review of src/dtreat (subagent, read-only)

WHAT: Bug-hunt findings reported to the review orchestrator (no source files modified).

HOW: Read all named files in full (permutation_significance, discrete_information,
judge_agreement_metrics, classifier_two_sample_test, chat_client,
parallel_chat_execution, base_schema, file_io, response_scoring_stage,
judge_calibration_stage, treatment_analysis_stage, response_collection_stage,
prompt_collection stages, backends, pipeline modules). Executed scratchpad scripts
via `uv run` and observed stdout directly:
- load_json regex corruption: `{"text": "CSV row: a,,b and list [1, 2, ]"}`
  loaded back as `"CSV row: a,b and list [1, 2]"` — BROKEN (confirmed bug).
- Stage-4 resume: `_aggregate_scored` with a failed judge produced an empty-verdict
  ScoredResponse; simulated re-run computed `to_score = []` — BROKEN (confirmed bug).
- Cross-set duplicate prompt_id: ran real `run_response_collection` on a crafted
  artifact; observed "collected: 2/4", both records labeled cishet — BROKEN (confirmed).
- run_c2st with minority class of 1 (n=12): observed sklearn ValueError raised, not
  None — BROKEN (contract violation confirmed).
- BaseSchema NaN roundtrip: to_dict → {'x': 'NaN'}, from_dict → str 'NaN' — confirmed.
- price_for_model("anthropic:claude-sonnet-4-5") → None (cost $0) — confirmed.
- Counter-checks that PASSED (no bug): Cohen kappa matches sklearn (0.0 on constructed
  example); BH q-values match the textbook example ([0.01, 0.04, 0.084, ...]);
  Wilson interval (0.8, n=50) → (0.6696, 0.8876); permutation p sane under null;
  tuple[float, ...] roundtrip OK.
NOT verified empirically (code-reading only, logic traced in full): cache tmp-file
write race (chat_client.py:135), seed-hypothesis truncation (hypothesis stage:112),
stale quarantine files on clean re-runs.

RESULT: VERIFIED (each reported finding either demonstrated by executed script or
explicitly labeled code-reading-only in the report).

## 2026-07-17 — Adversarial verification: stages<->pipeline package cycle finding

WHAT: Interfaces finding "Package cycle: stages import pipeline while pipeline
imports stages" (experiment_config.py:24) — verified for the review orchestrator
(read-only; no source files modified).

HOW: Read experiment_config.py, cli_entrypoint.py, stage_registry.py,
auto_export.py, and the __init__.py files in full; grepped all
`dtreat.pipeline` imports (8 stage modules + 3 diagnostics + 1 server module
import upward; stage_registry + cli_entrypoint import stages downward). Then
executed in .venv and observed stdout directly:
- `import dtreat.pipeline` → 'dtreat.pipeline.cli_entrypoint' NOT in
  sys.modules; `main` absent from dtreat.pipeline.__all__ — the cycle already
  silently drops cli_entrypoint TODAY.
- Instrumented importlib.import_module: captured the swallowed error —
  "cannot import name 'print_cost_estimate' from partially initialized module
  'dtreat.diagnostics.cost_estimation' (most likely due to a circular import)".
- `from dtreat.pipeline import main` → ImportError (demonstrated failure);
  direct `from dtreat.pipeline.cli_entrypoint import main` succeeds (fresh
  re-import after init), which is why the console script still works.
- Meta-path tracer confirmed stage modules import dtreat.pipeline submodules
  while dtreat.pipeline's __init__ is mid-execution (_initializing=True).

RESULT: VERIFIED — finding is real (is_real=true), with one path correction:
the demonstrated breakage runs through diagnostics (dtreat/__init__ →
diagnostics.cost_estimation → pipeline/__init__ → cli_entrypoint → half-
initialized cost_estimation), and diagnostics<->pipeline / server<->pipeline
cycles exist alongside stages<->pipeline. The proposed fix (move
experiment_config + run_directory_paths below pipeline) resolves all three.
## 2026-07-17 — code-review finding verification: `_axis_results` 9-arg signature
- WHAT: interfaces finding claiming `_axis_results` (src/dtreat/stages/treatment_analysis/treatment_analysis_stage.py:146) takes 9 positional untyped params carrying two cohesive array bundles.
- HOW: read treatment_analysis_stage.py, permutation_significance.py, and server/run_data_api.py in full; confirmed signature at line 146, call site at lines 69-71, the (sums, counts, is_target) tuple returned at line 143 and re-created at run_data_api.py:183, and the (deltas, p_values, q_values, significant) bundle produced at lines 64-67.
- RESULT: VERIFIED — finding is real as stated (minor precision note on which arrays are mutually swappable recorded in the review output).
## 2026-07-17 — code-review finding verification: judge call chain whole-config + loose args
- WHAT: interfaces finding on src/dtreat/stages/response_scoring/response_scoring_stage.py:129 (judge_all_responses takes full ExperimentConfig; helpers at 153/181 take 7/6 untyped positional params; _aggregate_scored triple-nested dict param at 251; forced config clone at judge_calibration_stage.py:199-200).
- HOW: read response_scoring_stage.py (283 lines), judge_calibration_stage.py (289 lines), and pipeline/experiment_config.py in full; confirmed the 6 consumed config fields (deployment_context:144, judge_mode:146, judge_temperature:164/197, judge_max_tokens:164, seed:165/199, max_workers:170/204) against the ~25-field ExperimentConfig; confirmed the from_dict(to_dict()) clone + seed mutation verbatim at calibration lines 199-200; confirmed CLAUDE.md lines 43-44 convention wording.
- RESULT: VERIFIED — finding is real as stated (precision notes: _aggregate_scored def is at line 247 with the nested-dict param at 251; judge_max_tokens is consumed only in per_response mode; the CLAUDE.md convention says "across module boundaries", which the module-crossing nested-dict return of judge_all_responses violates more directly than the module-private _aggregate_scored signature).

## 2026-07-17 — Code-review finding verification: run_c2st crash on skewed classes

WHAT: Claim that run_c2st (src/dtreat/stages/treatment_analysis/classifier_two_sample_test.py:35)
raises ValueError instead of returning None when a class is tiny, crashing stage 5.

HOW: Read classifier_two_sample_test.py and treatment_analysis_stage.py in full; ran a
repro script with the project venv (.venv, sklearn 1.9.0) calling run_c2st directly:
- 12 samples, minority class of 1 -> ValueError "The least populated classes in y have
  only 1 member" (guard n>=10/2-classes passes, stratified split raises at line 35)
- 10 samples, test_fraction=0.1 -> ValueError "The test_size = 1 should be greater or
  equal to the number of classes = 2" (sklearn uses ceil(n*test_fraction) for test size)
- balanced 20-sample control returns a result (normal path intact)
Confirmed run_c2st is called unguarded from _c2st_from_scored (treatment_analysis_stage.py:203),
which is called from run_treatment_analysis (line 82) with no try/except, so the
ValueError propagates and crashes stage 5.

RESULT: VERIFIED — finding is real.

## 2026-07-17 — Code-review finding verification: NaN/Inf floats break BaseSchema roundtrip

WHAT: Claim that _canon (src/dtreat/common/base_schema.py:65-69) serializes non-finite
floats as "NaN"/"Inf"/"-Inf" strings while _convert_value (lines 162-213) has no inverse,
so to_dict -> from_dict leaves a str in a float-typed field; reachable via
kl_divergence_bits returning inf when epsilon=0 is configured.

HOW: Read base_schema.py, file_io.py, discrete_information.py, treatment_analysis_stage.py,
experiment_config.py, analysis_report_schemas.py, run_data_api.py,
distinguish_report_schemas.py in full; grepped all from_dict/from_json/save_json call
sites. Ran an empirical repro with the project venv (.venv):
- S(x=nan).to_dict() -> {'x': 'NaN'}; S.from_dict -> 'NaN' (str). Same for inf -> 'Inf'.
- kl_divergence_bits(normalize_profile([.5,0],0), normalize_profile([0,.5],0)) -> inf.
- save_json(S(x=inf).to_dict()) writes "Inf"; S.from_json returns str; arithmetic on it
  raises TypeError and f"{x:.2f}" raises ValueError.
Confirmed ExperimentConfig.validate() (lines 92-110) never checks epsilon, so epsilon=0
passes validation, and save_json(report.to_dict()) at treatment_analysis_stage.py:94
persists "Inf". Precision note: no in-repo code calls AnalysisReport.from_json today —
current consumers (server run_data_api.py:68-74,160; diagnostics artifact_inspection.py:102)
read the raw dict and propagate the "Inf" string through the debug API instead of crashing.
The live same-pipeline roundtrip exercising the broken contract is
InputDistinguishabilityReport (written distinguish_bridge_stage.py:211, read via from_json
at line 228, then compared arithmetically at treatment_analysis_stage.py:269).

RESULT: VERIFIED — core defect is real and empirically confirmed; the named crash site
(AnalysisReport.from_json) is one step hypothetical since it currently has no caller.
