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
## 2026-07-17 — Web research: seed prompts for fitness/nutrition bias-audit dataset
- WHAT: Verbatim Reddit question prompts (LGBTQ+ vs general/cis-het fitness communities) collected for the report delivered in-chat (not written to a file). Sources: PullPush API (api.pullpush.io) queries over r/FTMFitness, r/askgaybros, r/gaybros, r/MtF, r/butchlesbians, r/actuallesbians, r/beginnerfitness, r/naturalbodybuilding, r/gainit, r/GYM, r/Fitness, r/bodybuilding.
- HOW: For ~20 quotes used in the report, re-fetched the raw PullPush JSON with curl and printed title/selftext/permalink directly (bypassing WebFetch's summarizer model) and compared strings: FTMFitness (5 posts), askgaybros (5), beginnerfitness (4), MtF (3), butchlesbians (3), gainit (1), naturalbodybuilding (2). All matched verbatim. RESULT: VERIFIED.
- Quotes taken only via WebFetch extraction (small-model mediated, not curl-checked): r/gaybros (2), r/bodybuilding (1), r/GYM Arnold-split (1), r/Fitness titles, r/gainit girlfriend-cue posts. RESULT: UNVERIFIED as exact-verbatim (flagged as such in report).
- Claims about Reddit API terms/robots.txt/Reddit-for-Researchers, WildChat/LMSYS licenses: from search-result snippets (TechCrunch, support.reddithelp.com, HuggingFace listings); reddit.com/redditinc.com/fitness.stackexchange.com not fetchable from this environment. RESULT: UNVERIFIED at primary-source level (flagged in report).
