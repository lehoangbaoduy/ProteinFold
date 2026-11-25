# ProteinFold — Implementation Plan & Progress

Living document: always reflects current status. Spec: [PRD.md](PRD.md) · Brief: [HANDOFF.md](HANDOFF.md)

**Legend:** ☐ todo · ◐ in progress · ☑ done (verified) · ⊘ blocked · — skipped (with reason)

**Process per task (master prompt):** harness-os pipeline: Constitution → Specification → Test Suite (observed RED) → Generate Code (observed GREEN) → Code Review (correctness, performance, security). Then update this file.

---

## Current status

| | |
|---|---|
| **Phase** | 1 ☑ · 2 ☑ · 3 ◐ (runner ☑, penalty/repair ☑, tuning ☑, main 2D + 3D running) · 4 ◐ (stats ☑, figures ☑, tables ☑; sensitivity pending; animation dropped (D30)) · 5 ◐ (web export ☑ `4d16bc2`; scaffold + input in review) |
| **Last update** | 2026-11-24 |
| **Next action** | (1) Finish review of harness run 33 (web scaffold + input + share codec) → commit. (2) `main_2d` still running in the background (284/990 at 19:56; resumable). (3) Next web units: Pyodide worker + wheel/self-hosted Pyodide (T5.2/T5.3), live views (T5.6/T5.7), paper-results tab (T5.10), about (T5.11), Playwright checks (T5.12–T5.14). (4) After main_2d: commit results, `make figures tables web-data`. |

### Blockers
| ID | Blocker | Needed from | Since |
|---|---|---|---|
| B1 | ~~Harness-os not initialized~~. Resolved: the user ran `harness init` (commit `d6b01ee`). | User | resolved 2026-11-24 |
| B2 | The generated hook scripts are mode 644 (not executable), so the gate would fail open. The test command `pytest` used the user-level Python without project dependencies, so Claude changed it to `uv run pytest` and added web globs; this needs `reconcile-config`. | User | resolved 2026-11-24: reconciled (`b3fcef3`); hooks made executable (`f9c7641`); hooks turned out to be live without a restart |
| B3 | ~~The gate daemon's `/workspaces` bind mount was stale (empty), so every gate check failed closed~~ | User (container restart) | resolved 2026-11-24 |

### Harness-os state
| Item | Value |
|---|---|
| Product spec | `PFOLD-PROD-001` v1 (active) |
| Phase 1 domain spec | `PFOLD-DOM-001` v1 (active): HP model core |
| Workflow | run 21, `new-feature`: spec ☑, risk ☑ (#28), RED observed ☑, implement ☑ (decision #298), review ☑ (round 1 WARNING → fixed → round 2 APPROVE, decision #304), finalize after commit |
| Risk assessment | #28, medium: gates `spec`, `tests`, `review:code`. Assessment #27 ("high") was a false positive: the keyword matched the phrase "no secrets" in the description, and the change was reassessed with the same scope. |

---

## Phase 0 — Repository setup ☑
| ID | Task | Files | Status |
|---|---|---|---|
| T0.1 | git init (`main`), repo-local identity, `.gitignore`, initial commit of docs | `.gitignore`, `docs/HANDOFF.md`, `docs/PRD.md` | ☑ commit `80dffa5` |
| T0.2 | Register product spec and Phase 1 domain spec in harness-os | — | ☑ `PFOLD-PROD-001`, `PFOLD-DOM-001` |
| T0.3 | Harness-os project init (gating hooks + config) | `.claude/harness.config.json`, `.claude/hooks/*`, `.claude/settings.json` | ☑ init `d6b01ee`, reconcile `b3fcef3`, hooks +x `f9c7641`; gate verified live (blocked a probe write) |
| T0.4 | uv environment + lockfile (Python 3.12.16, numpy 2.5.5, hypothesis 6.170) | `.venv/`, `uv.lock` | ☑ |

## Phase 1 — Core (PRD §10 row 1)
**Exit criteria:** E = −3 check passes; brute-force energy tests pass; benchmarks cited; ≥ 90% coverage on the HP model.

| ID | Task | Files | Tests (written first) | Status |
|---|---|---|---|---|
| T1.1 | Python project: uv env + lockfile, dependencies (numpy; dev: pytest, pytest-cov, hypothesis, scipy, pandas, matplotlib, bandit, pip-audit), Makefile targets `test`, `cov`, `lint`, `audit` | `pyproject.toml`, `uv.lock`, `Makefile`, `src/hpfold/__init__.py`, `src/hpfold_tools/__init__.py` | `make test`/`cov`/`lint`/`sec`/`audit`/`check` | ☑ |
| T1.2 | `parse_sequence` (FR-8): strip whitespace, uppercase, reject non-H/P and empty input | `src/hpfold/hp_model.py` | `tests/test_hp_model.py::test_parse_*` | ☑ |
| T1.3 | `decode`/`encode` with the (heading, up) frame, for 2D and 3D (FR-1, FR-2) | `src/hpfold/hp_model.py` | hand-checked coordinates; 3D U/D/L frame cases; round trip | ☑ |
| T1.4 | `is_valid`, `collisions`, `contacts`, `energy` with an O(n) hash map (FR-3, FR-4) | `src/hpfold/hp_model.py` | HPHHHHPHHP → −3 with contacts {(0,5),(3,8),(4,7)}; bonded pairs never count; brute-force O(n²) cross-check; exhaustive enumeration for small n | ☑ (exhaustive: 3^8 2D and 5^6 3D folds) |
| T1.5 | `evaluate(seq, moves, strategy, penalty, rng, dim)` → Evaluation (FR-5, §5.2) | `src/hpfold/fitness.py` | `tests/test_fitness.py`: penalty arithmetic; invalid flagged; repair strategy returns a valid fold | ☑ |
| T1.6 | `random_valid_fold` (backtracking SAW) and `repair` (FR-6, FR-7) | `src/hpfold/folds.py` | always valid for 2D n ≤ 100 and 3D n ≤ 100; deterministic for a given seed; repair output valid and same length | ☑ |
| T1.7 | Hypothesis properties: unit bonds; `encode(decode(m)) == m`; energy ≤ 0; energy == brute force; repair(m) valid | `tests/test_hp_properties.py` | (this is the test) + parse_sequence fuzzing | ☑ |
| T1.8 | Benchmarks from the primary literature: 2D set (8–11 sequences, n ≈ 20–100) and 3D subset (3–5), each with a citation; source disagreements recorded (FR-B1 to B5) | `data/benchmarks.json`, `docs/benchmarks_sources.md` | `tests/test_benchmarks.py`: alphabet, length, cited sources, independent notation expansion, contact bound, 21 golden pins | ☑ 21 instances (11 2D, 10 3D); reviewer re-verified against the source XML |
| T1.9 | Matplotlib fold plot, 2D and 3D (tooling package) | `src/hpfold_tools/visualize.py` | `tests/test_visualize.py`: contact endpoints, bead positions, H/P fills, no figure leaks, axes/dim checks, save paths | ☑ visually inspected 2D and 3D renders |
| T1.10 | Purity guard: the core package imports only stdlib + numpy | `tests/test_core_purity.py` | AST scan of `src/hpfold/**` imports | ☑ |

**Phase 1 exit check (2026-11-24):** E = −3 ☑ · brute-force + exhaustive energy tests ☑ · 21 benchmarks cited and re-verified ☑ · coverage 99.3% (≥ 90%) ☑ · ruff/Bandit/pip-audit clean ☑

## Phase 2 — Algorithms (PRD §10 row 2)
**Exit criteria:** all three reach a reasonable energy on a 20-mer in 2D and 3D; seeds reproduce; budget exact; cancel and progress frames work.

| ID | Task | Files | Tests (written first) | Status |
|---|---|---|---|---|
| T2.1 | Shared interface: `RunResult`, `Frame`, evaluation counter with an exact budget, `target`, `on_progress`, `should_stop`; NA result when no valid fold is found | `src/hpfold/algorithms/base.py` | `tests/test_algorithm_contract.py` (parametrized over every algorithm): budget never exceeded and exactly reached unless the target is hit or the run is cancelled; same seed gives an identical result; frames are monotone; cancel returns best-so-far; NA when no valid fold | ☑ + `RunResult.to_dict`, frames agree with history, seed/target validation |
| T2.2 | GA (§5.3): tournament, 1-point/2-point/uniform crossover, per-gene mutation, elitism, penalty or repair | `src/hpfold/algorithms/ga.py` | `tests/test_ga.py` + contract | ☑ review APPROVE + fixes |
| T2.3 | ACO (§5.3): τ[i][m], heuristic η from new H–H contacts, bounded backtracking + restart, evaporation, deposit rule, optional MMAS bounds; one evaluation per completed fold | `src/hpfold/algorithms/aco.py` | `tests/test_aco.py`: always valid folds; eval count == completed ants; τ stays within bounds; backtrack + restart paths; heuristic bias; colony diversity + contract | ☑ review APPROVE + fixes (`c73146a`) |
| T2.4 | PSO (§5.3): probability matrices, w/c₁/c₂ update, row renormalization, floor ε, sampling, penalty or repair | `src/hpfold/algorithms/pso.py` | `tests/test_pso.py`: rows sum to 1; min ≥ ε; no one-hot collapse; exact update; ring indices; r1/r2 independence; diversity source + contract | ☑ review APPROVE + mutant-killing tests (`b19f7f0`) |
| T2.5 | Frame `matrix` + `diversity` per algorithm (FR-T1) | the three algorithm files | frame schema test | ☑ (contract `test_progress_frames`, D17) |
| T2.6 | Pyodide timing spike: run GA/ACO/PSO in Pyodide (Node or browser) at n = 20, 48, 100; set effort multipliers and the length cap | throwaway scripts (session scratchpad), results → D21 | timings recorded; native-vs-Pyodide parity checked | ☑ |
| T2.7 | *(Could)* Pull-move local search + hybrid ACO-LS; LS evaluations count toward the budget | `src/hpfold/local_search.py`, `src/hpfold/algorithms/hybrid.py` | `tests/test_local_search.py`: moves keep validity; energy never worsens | ☐ |

**Phase 2 exit check (2026-11-24):** GA, ACO and PSO each reach ≤ −7 (2D) / ≤ −8 (3D) on the 20-mer and E* = −9 in the pilot ☑ · seeds reproduce ☑ · exact budget ☑ · cancel + frames ☑ · Pyodide parity in the spike ☑

## Phase 3 — Experiments (PRD §10 row 3)
**Exit criteria:** `results/runs.csv` complete; tuned defaults exported.

| ID | Task | Files | Tests | Status |
|---|---|---|---|---|
| T3.1 | Experiment runner: TOML configs, process pool, resume (skip finished rows), CSV schema FR-E3, budget B(n) = c·n², config + git hash recorded | `src/hpfold_tools/experiments.py`, `configs/*.toml` | `tests/test_experiments.py`: schema; resume skips/extends; truncated-line repair; config-change refusal; worker-count determinism; strict keys | ☑ review (2 HIGH) fixed (`3466993`) |
| T3.2 | Penalty vs. repair mini-experiment (§7.4) → choose a strategy | `configs/penalty_repair.toml`, `results/penalty_repair.csv` | 720 runs | ☑ repair wins for GA and PSO on all 3 sequences (D24) |
| T3.3 | Tuning with equal effort per algorithm (§7.3) | `configs/tuning.toml`, `results/tuning.csv` | 360 runs | ☑ winners in D27; applied in `configs/main_2d.toml`, `configs/ext_3d.toml` |
| T3.4 | Main 2D study: 30 seeds × 3 algorithms × all 2D benchmarks | `configs/main_2d.toml`, `results/runs.csv` | row-count check | ☐ |
| T3.5 | 3D extension: 30 seeds × 3 algorithms × 3D subset | `configs/ext_3d.toml` | row-count check | ☑ 450 runs = 3 algorithms × 5 sequences × 30 seeds |
| T3.6 | Web export: traces (median + best run, < 2 MB each), benchmarks JSON, `defaults.json` (FR-E6), summary JSON | `src/hpfold_tools/export_web.py`, `web/public/data/`, `make web-data` | `tests/test_export_web.py`: schemas, pre-flight + exact reproduction, thinning, stale cleanup, size cap, parallel = sequential | ☑ review (1 HIGH) fixed |

## Phase 4 — Analysis (PRD §10 row 4)
**Exit criteria:** every paper figure and table is regenerated by `make figures`.

| ID | Task | Files | Tests | Status |
|---|---|---|---|---|
| T4.1 | Statistics: summary (FR-A1), Wilcoxon rank-sum + Holm, A₁₂, optional Friedman | `src/hpfold_tools/analysis.py` | `tests/test_analysis.py`: hand-computed A₁₂, Holm (per sequence + global), exact-permutation p-values, success rate, evals-to-best on successes only, NA handling, Friedman | ☑ review (1 HIGH: asymptotic p under ties) fixed |
| T4.2 | Figures: convergence (median + IQR), box/violin, best folds 2D/3D | `src/hpfold_tools/figures.py`, `results/figures/`, `make figures` | `tests/test_figures.py`: step lookup, masked IQR bands, NA reporting, 3D folds, legend/tick layout | ☑ review fixed; visually inspected |
| T4.3 | LaTeX tables (booktabs) from the CSV | `src/hpfold_tools/tables.py`, `paper/tables/`, `make tables` | `tests/test_tables.py`: structure, column counts, bold-best, k/n success, NA, escaping, CLI | ☑ (`a024af3`) |
| T4.4 | Sensitivity study (§7.3) | `configs/sensitivity.toml`, `results/sensitivity.csv`, figure | — | ☐ |
| T4.5 | ~~Talk animation (GIF/MP4)~~ | — | — | ✗ dropped by the user (D30): the talk is given in person |

## Phase 5 — Webpage (PRD §10 row 5)
**Exit criteria:** §14 webpage criteria met; offline demo verified; all checks done with Playwright screenshots.

| ID | Task | Files | Tests / verification | Status |
|---|---|---|---|---|
| T5.1 | Scaffold: Vite + TS, ESLint (+ security plugin), vitest, Playwright, CSP meta compatible with WASM | `web/package.json`, `web/vite.config.ts`, `web/tsconfig.json`, `web/eslint.config.js` | `npm --prefix web test`, `npm --prefix web run lint`, `tsc --noEmit` | ☑ review (run 33) fixed (`5a6a9f3`) |
| T5.2 | Package hpfold as a wheel for Pyodide; self-host Pyodide | `scripts/build_web_wheel.py`, `web/public/py/`, `web/public/pyodide/` | worker loads offline | ☑ `make web-runtime`, review (run 34) fixed (`e696d85`) |
| T5.3 | Solver worker: run jobs, post frames, cancel, error reporting | `web/src/worker/solver.worker.ts`, `web/src/worker/protocol.ts` | vitest on the protocol; Playwright run | ☑ Python side (`e696d85`), worker/pool/protocol (`8e59729`, review run 35 fixed `241bf9d`); Chromium smoke: golden parity, cancel keeps best-so-far |
| T5.4 | Input + validation (FR-L1): normalize, H/P only, 3–100, presets, random | `web/src/solver/input.ts` | vitest + fast-check fuzzing of the parser (no throw, never accepts invalid input) | ☑ (`5a6a9f3`) |
| T5.5 | Options (FR-L2): lattice, algorithms, effort, seed, advanced params from `defaults.json` | `web/src/solver/options.ts` | vitest | ☑ options + job building from the real data files, review run 36 fixed |
| T5.6 | Live view (FR-L4): 2D fold SVG, convergence chart, internals heatmap, race layout, progress + elapsed time | `web/src/views/fold2d.ts`, `convergence.ts`, `heatmap.ts`, `race.ts` | Playwright screenshots while running | ☐ design via Claude Design from `docs/design-brief.md` (D32) |
| T5.7 | Results (FR-L5): 3D fold (three.js), metrics panel, comparison table, benchmark badge + citation | `web/src/views/fold3d.ts`, `metrics.ts`, `compare.ts` | Playwright screenshots | ☐ |
| T5.8 | Cancel, Pyodide loading state (FR-L7), honesty note (FR-L8), "no valid fold found" state | `web/src/solver/*` | Playwright | ☐ |
| T5.9 | Export + share URL codec (FR-L6), treating URL input as untrusted | `web/src/share/url.ts` | vitest + fuzzing of the decoder | ◐ share codec ☑ (`5a6a9f3`); PNG/SVG/JSON export pending |
| T5.10 | Paper results tab: replay + dashboard (FR-W1, W2) | `web/src/results/*` | Playwright | ☐ |
| T5.11 | About / explainers (FR-X1), including "search, not training" | `web/src/about/*` | Playwright | ☐ |
| T5.12 | Parity: same (seq, params, seed) gives the same result natively and in Pyodide, checked in a real browser | `web/tests/e2e/parity.spec.ts`, `tests/fixtures/parity.json` | Playwright | ☐ |
| T5.13 | Real-data-only audit: no mock, placeholder or hard-coded sample data in the shipped page | `web/tests/e2e/no_mock.spec.ts` + grep check | Playwright + CI grep | ☐ |
| T5.14 | Offline mode check (network blocked in Playwright) | `web/tests/e2e/offline.spec.ts` | Playwright | ☐ |
| T5.15 | Deploy as a public static site (**confirm with the user before publishing**) | `.github/workflows/pages.yml` or Vercel config | live URL check | ☐ |

## Phase 6 — Paper & talk (PRD §10 row 6)
| ID | Task | Files | Status |
|---|---|---|---|
| T6.1 | IEEE paper (§11), with figures and tables pulled from `results/` | `paper/main.tex`, `paper/refs.bib` | ☐ |
| T6.2 | 15-minute talk slides following the §11 timing table | `paper/talk/` | ☐ |
| T6.3 | Recorded live-demo fallback video | `paper/talk/demo.mp4` | ☐ |

## Cross-cutting — quality & security (run in every Code Review stage)
| ID | Check | Tool / file | Status |
|---|---|---|---|
| Q1 | Python SAST | `make sec` (Bandit) | ☑ Phase 1 clean |
| Q2 | Python SCA | `make audit` (pip-audit on the uv lock export) | ☑ Phase 1: no known vulnerabilities |
| Q3 | JS SAST + SCA | ESLint security rules, `npm audit` | ☐ (Phase 5) |
| Q4 | Semgrep (Python + TS rulesets) | `semgrep --config auto` | ☐ |
| Q5 | Fuzzing | Hypothesis (core), fast-check (web parser and URL decoder) | ◐ Hypothesis on the core (incl. parse_sequence fuzzing); web pending |
| Q6 | Dependabot | `.github/dependabot.yml`. Takes effect only once a GitHub remote exists. | ☐ |
| Q7 | DAST (optional) | OWASP ZAP baseline against the deployed site | ☐ |
| Q8 | Coverage ≥ 90% on `hp_model`, ≥ 80% overall | `make cov` (fails under 90% on `src/hpfold`) | ◐ 99.3% on `src/hpfold` (Phase 1) |

---

## Edge-case catalog (PRD §8) → required tests
Every row is a required task.

| # | Case | Expected | Test location | Status |
|---|---|---|---|---|
| E1 | Characters other than H/P; empty input | Reject with a clear message | `tests/test_hp_model.py::test_parse_rejects_*`; `web/src/solver/input.test.ts` | ◐ library ☑; web pending |
| E2 | Lowercase / whitespace | Normalized | `tests/test_hp_model.py::test_parse_normalizes`; web input test | ◐ library ☑; web pending |
| E3 | n < 3 | Library: trivial fold, E = 0; web: rejected | `tests/test_hp_model.py::test_short_sequences_are_trivial`, `tests/test_fitness.py::test_trivial_sequences`; web input test | ◐ library ☑; web pending |
| E4 | n > 100 on the web | Rejected with an explanation (CLI has no cap) | web input test; `tests/test_algorithm_contract.py::test_no_cap_in_library` | ☐ |
| E5 | No H / single H | E = 0; runs normally; the page notes it | `tests/test_algorithm_contract.py::test_no_contacts_possible_runs_normally`; Playwright | ◐ library ☑; web pending |
| E6 | All-H sequence | Valid stress test | `tests/test_algorithm_contract.py::test_all_h_stress` | ☑ |
| E7 | Never finds a valid fold (penalty) | best_energy = NA; counted as a failure; page says "no valid fold found" | `tests/test_algorithm_contract.py::test_zero_budget_gives_na_not_zero`; `tests/test_experiments.py::test_na_rows_are_written_as_empty_not_zero`; `tests/test_analysis.py::test_summary_counts_na_as_failure_and_excludes_it_from_energy_stats`, `test_na_ranks_below_an_energy_of_zero`; Playwright | ◐ library + analysis ☑; web pending |
| E8 | ACO ant dead end | Bounded backtracking → restart; restarts logged | `tests/test_aco.py::test_dead_ends_trigger_counted_restarts_and_stay_valid`, `test_backtracking_path_without_restarts`, `test_fallback_after_max_restarts_is_valid_and_flagged` | ☑ |
| E9 | PSO one-hot collapse | Prevented by ε; diversity logged | `tests/test_pso.py::test_floor_prevents_collapse` | ☑ |
| E10 | Fitness ties | Deterministic (stable sort) | `tests/test_algorithm_contract.py::test_same_seed_same_result_different_seed_differs` | ☑ |
| E11 | Disputed best-known / 3D lower bound | Both stored and cited; success uses the better value; flagged | `tests/test_benchmarks.py`; `tests/test_analysis.py::test_success_uses_best_known_inclusive` | ◐ data + analysis ☑ (no E* disputes exist in the two sources; success = best_energy ≤ best_known, where a disputed value would be stored as the better one); web flag pending |
| E12 | Target reached early | Stop; evals_to_best recorded; page shows "reached best-known" | `tests/test_algorithm_contract.py::test_target_early_stop`; Playwright | ☐ |
| E13 | Cancel / tab closed mid-run | Workers terminated; partial best shown | `tests/test_algorithm_contract.py::test_cancel`; Playwright | ☐ |
| E14 | Pyodide fails to load | Clear error; paper-results tab still works | `web/tests/e2e/pyodide_fail.spec.ts` | ☐ |
| E15 | Long sequences miss the best-known energy | Reported honestly (paper + page) | review checklist item in T6.1 / T5.7 | ☐ |

---

## Acceptance criteria (PRD §14) — project exit
- [ ] All unit tests pass, including E = −3, brute-force cross-checks, and the native-vs-Pyodide parity test.
- [ ] Benchmark file cites every sequence and energy.
- [ ] 30 runs × 3 algorithms × every 2D benchmark completed; `runs.csv` committed.
- [ ] 3D extension results for at least 3 sequences.
- [ ] Stats table with Holm-corrected Wilcoxon p-values and A₁₂.
- [ ] Every paper figure and table regenerated by `make figures` from the CSV.
- [ ] Webpage: any valid sequence (3–100), GA/ACO/PSO in 2D and/or 3D, live progress, cancel, 2D/3D folds with all FR-L5 metrics.
- [ ] Webpage: paper-results tab replays real traces and shows the dashboard.
- [ ] Webpage deployed publicly and works offline from localhost.
- [ ] Paper complete in IEEE format; 15-minute talk delivered with the live demo.

---

## Decisions log
Design decisions, especially where the PRD left details open.

| # | Date | Decision | Rationale |
|---|---|---|---|
| D1 | 2026-11-24 | Git: branch `main`, repo-local identity from the user's account email | User choice; no global git identity configured |
| D2 | 2026-11-24 | Core package `src/hpfold` (stdlib + numpy only) and tooling package `src/hpfold_tools` (scipy, pandas, matplotlib, multiprocessing, I/O) | PRD §5.3/§9: the core must run unmodified in Pyodide; enforced by T1.10 |
| D3 | 2026-11-24 | Benchmarks stored as `data/benchmarks.json`, with source notes in `docs/benchmarks_sources.md` | JSON is readable by both Python and the webpage without conversion |
| D6 | 2026-11-24 | `evaluate` lives in `src/hpfold/fitness.py` (not hp_model) and takes an explicit `dim` (inferred from U/D letters only if omitted); `DEFAULT_PENALTY = 2.0` provisional until T3.2 | Avoids a circular import hp_model ↔ folds; algorithms know their lattice, so repair must not guess it |
| D7 | 2026-11-24 | `repair` = the same DFS as `random_valid_fold`, but trying the fold's own move first at each position | Keeps the valid prefix unchanged and alters the fold minimally; always returns a valid fold |
| D8 | 2026-11-24 | Fold growth uses random restarts: per-attempt backtrack budget `BACKTRACKS_PER_BEAD = 50` × n, up to `MAX_ATTEMPTS = 100`; repair prefers the original moves for the first half of the attempts | Review H1: plain DFS hit a 200k-backtrack cap in ~0.07% of n=100 folds; with restarts, 0 failures in 3000 (n=100), 300 (n=200), 100 (n=400) runs; ~0.6 ms per n=100 fold |
| D9 | 2026-11-24 | `evaluate(sequence, moves, dim, ...)`: `dim` is required; moves are checked against it in both strategies; the sequence must already be uppercase H/P (`parse_sequence` output); the penalty must be finite and ≥ 0 | Review M2/L2/L3: inferring dim from U/D letters would silently confine 3D repair to a plane |
| D10 | 2026-11-24 | Benchmarks: Thachuk et al. 2007 Table 1 is the primary transcription source, cross-checked against Shmygelska & Hoos 2005; all `energy_kind = best_known` | Both papers agree on all 21 E*; the 2005 S1-6 and S2-1 notations fail their own length checks (51/47) |
| D11 | 2026-11-24 | Quality targets: `make lint` = ruff, `make sec` = Bandit, `make cov` fails under 90% on `src/hpfold`, `make audit` = pip-audit of `uv export`; the dev group includes the tooling deps | Review M1/M6: fresh-clone `make test` must work; lint and security scans are separate concerns |
| D12 | 2026-11-24 | Fold plots: `plot_fold` requires an already-normalized sequence (no silent lowercase handling); 3D uses an equal box aspect; `save_fold` requires a file suffix and creates parent directories | Review M3/L3/L4 |
| D13 | 2026-11-24 | Algorithm contract: `run(sequence, dim, params, seed, budget, target, on_progress, should_stop)`; budget enforced by `Evaluator`; cancel checked before every iteration (incl. the first); history starts at (0, None) and is recorded once per iteration; frames show the fittest VALID fold of the population; `RunResult.to_dict()` is JSON-ready | Review of unit PFOLD-DOM-003 |
| D14 | 2026-11-24 | GA defaults: pop 100, two-point crossover, pc 0.9, pm 2/L, tournament k 3, elite 2, penalty λ 2 (provisional until T3.2/T3.3) | Common textbook values; tuned later with equal effort per algorithm |
| D15 | 2026-11-24 | Open performance note: GA throughput (native CPython) ~24k evals/s at n=20 and ~8k at n=100 with penalty, but only ~430 evals/s at n=100 2D with repair (repair DFS on every colliding child) | Decide the strategy in T3.2; calibrate web effort presets in T2.6; consider optimizing `contacts`/repair if repair wins |
| D16 | 2026-11-24 | ACO provisional defaults: 30 ants, α 1, β 2, γ 1, ρ 0.05, τ ∈ [0.1, 2.0], deposit = iteration-best + global-best, backtrack limit 50, 20 restarts | The first defaults (ρ 0.2, τ ∈ [0.05, 5]) stagnated at E = −6 on S1-1; pilot sweep picked these (see D19) |
| D17 | 2026-11-24 | Frame diversity is always computed from realized move frequencies (GA population, ACO colony, PSO swarm positions); the heatmap shows each algorithm's internals (GA frequencies, ACO pheromone τ^α, PSO probabilities of the swarm-best particle) | Review: clamped pheromone entropy is not comparable with GA diversity |
| D18 | 2026-11-24 | Native throughput at n = 100 (2D, penalty): GA ~8k, PSO ~7.6k, ACO ~1.5k evals/s; repair is 4–20× slower | Equal-evaluation comparison; report wall-clock alongside evaluations in the paper |
| D19 | 2026-11-24 | PSO provisional defaults: swarm 30, w 0.9, c1 = c2 = 0.5, ε 0.03, global topology; GA provisional tournament k = 5. All three algorithms received the same pilot (7 configs × 4 seeds × 20k evals on S1-1): GA mean −8.75, ACO −9, PSO −8.5 | Textbook PSO defaults stagnated at −6; the reviewer flagged unequal tuning, so the GA got the identical pilot. S1-1 is the pilot instance: flag it in the paper; T3.3 tunes on held-out sequences with equal effort |
| D20 | 2026-11-24 | Accepted test gap: which particle's matrix is shown in a PSO frame (the swarm-best owner) is not pinned by a test | Would require exposing swarm internals; behaviour is verified by review |
| D21 | 2026-11-24 | Pyodide spike (Node 24, Pyodide 0.28.3 = Python 3.13.2 + numpy 2.2.5): 1.6–2.9× slower than native; load ≈ 1.4 s; ACO n = 100 ≈ 1.2k evals/s (slowest), GA/PSO ≈ 3.6–4.1k. Results bit-identical to native (12/12 cases, same seed). Web effort presets: budget = max(5000, c·n²) with c = 2 (Quick), 5 (Standard), 12 (Thorough); length cap 100 kept | Meets PRD targets (n = 20 Standard < 5 s; n = 100 Thorough ≤ ~100 s for ACO, the slowest); parity looks achievable without special handling |
| D22 | 2026-11-24 | Experiment runner: TOML configs with [[variants]]; B(n) = max(min, ceil(c·n²)); shared seeds; process pool, rows in completion order; safe resume (trim cut-off last line, error on corrupt rows, history re-sync, config-hash refusal; seeds/instances may grow); strict keys | Review PFOLD-DATA-002 (H1, H2, M1–M4) |
| D23 | 2026-11-24 | Statistics: rank-sum p-values by permutation (exact if ≤ 20 000 arrangements, else seeded Monte Carlo); report `p_holm_global` (Holm across all sequence×pair tests) in the paper, `p_holm` per sequence as secondary; A₁₂ for minimization with NA ranked worst; Friedman on per-sequence medians for the overall ranking | Review PFOLD-DOM-006: asymptotic p was 0.067 vs exact 0.167 under heavy ties |
| D24 | 2026-11-24 | Constraint handling for the main study: **repair** for GA and PSO (ACO needs none) | Penalty-vs-repair (720 runs; S1-4/5/7; 30 seeds; c = 10): repair beats every penalty λ for both algorithms on every sequence (A₁₂ of penalty vs repair 0.00–0.09, global-Holm p ≈ 0.008 = permutation floor). Mean gap to E*: GA-repair 3.9, PSO-repair 5.6 vs 7.2–14.2 for penalties; λ = 5 > 2 > 1. Repair costs ~3× wall-clock per evaluation (reported) |
| D25 | 2026-11-24 | Main-study budget: B(n) = max(10 000, 20·n²) (c = 20) | No variant reached E* at c = 10 for n = 36–60. On S1-7 (n = 60) all three algorithms plateau: 36k → 200k evaluations improves the best by only 1–2 (GA/ACO −32 → −33, PSO −29 → −30 vs E* −36). More budget alone will not reach E* on long chains — a limitation to report; local search (hybrid, T2.7) is the remedy |
| D26 | 2026-11-24 | History JSONL stores only change points (+ first/last); consumers MUST treat it as a right-continuous step function (look up with bisect_right − 1, never interpolate) | Lossless (verified for every evaluation count), 50× smaller (5.0 MB → 100 KB for 720 runs) |
| D27 | 2026-11-24 | Tuned parameters (equal effort; 6 configs × 2 held-out sequences × 10 seeds per algorithm; selection = lowest mean relative gap to E*, tie-break success rate): GA pop 100, k 3, repair (gap 0.065, success 35%); ACO ρ 0.1, β 3 (0.071, 40%); PSO w 0.7, c1 = c2 = 1.0, ε 0.05, repair (0.148, 15%). Library defaults stay provisional; configs carry the tuned values; web defaults exported from them (FR-E6) | ACO's best ρ is at the grid edge (0.1 = largest tried): note in the paper. PSO top-2 tied on gap and success; the faster won |
| D28 | 2026-11-24 | Figures: convergence median/IQR drawn only where every run has a valid fold (no survivorship bias); NA counts shown as text; legend below the panels; ticks in thousands | Review PFOLD-DOM-007 |
| D29 | 2026-11-25 | Web traces: best and lower-median run per (variant, sequence) re-run deterministically; pre-flight (experiment, variants, params hash, budget) + exact reproduction of energy, fold, evaluations, evals-to-best, target; frames thinned (always first, last improvement, last; ≤ 300; < 2 MB); runtime dropped so exports are byte-reproducible; experiments.json index | Review PFOLD-DATA-003 |
| D30 | 2026-11-24 | T4.5 / FR-A6 talk animation (GIF/MP4) dropped | User: "I don't think I need animation for the talk, I will do it in person, you don't need to create mp4 or gif for that". The live webpage already animates the search; the separate demo fallback video (T6.3) is a different item and is unaffected unless the user drops it too |
| D31 | 2026-11-24 | Browser runs: one JSON entry point `hpfold.webjob.run_job_json` (validation in Python, the code under test); cancel = terminate the worker, keeping the last emitted frame (every improvement is emitted, one frame per iteration); population/loop params capped at 1000, budget at 200k; self-hosted Pyodide with numpy sha256 pinned twice (lock + code); golden fixture `tests/fixtures/web_golden.json` checked natively and in Pyodide | No SharedArrayBuffer on static hosting, so Python cannot be interrupted; review run 34 |
| D32 | 2026-11-24 | Webpage visual style: clean academic (light background, restrained palette with H in one strong colour and P grey, serif headings, generous whitespace) | User's choice among clean academic / dark dashboard / theme toggle; projects well in lecture halls |
| D5 | 2026-11-24 | Harness config: gatedGlobs `**/*.py` + `web/src/**/*.ts`; test command `uv run pytest` (Python) and `npm --prefix web test` (web) | The generated bare `pytest` resolved to the user-level Python without project dependencies, so tests could never go GREEN; web globs added now to avoid a second reconcile in Phase 5 |
| D4 | 2026-11-24 | Harness-os spec IDs use the prefix `PFOLD` (`PFOLD-PROD-001`, `PFOLD-DOM-001`, …) | Schema requires `^[A-Z][A-Z0-9]{1,9}-<TYPE>-NNN$` |

Open details to decide during implementation (to be logged here): penalty λ default; ACO γ, deposit rule, τ bounds; PSO ε, w, c₁, c₂; budget constant c in B(n) = c·n²; trace frame interval k; web effort multipliers and length cap (after T2.6); choice between GitHub Pages and Vercel.

---

## Changelog
- 2026-11-25 — T3.6 web export reviewed + fixed (D29).
- 2026-11-24 — T4.3 LaTeX tables (`a024af3`): review APPROVE (2 MEDIUM fixed: variant selection, exact k/n success).
- 2026-11-24 — T3.3 tuning done (D27); T4.2 figures module reviewed + fixed (D28); main 2D and 3D configs written.
- 2026-11-24 — T3.2 penalty vs repair done → repair (D24); budget c = 20 (D25); history thinning (D26). Tuning config written.
- 2026-11-24 — T3.1 experiment runner (`3466993`) and T4.1 statistics (permutation p-values, global Holm, Friedman) reviewed and fixed. Penalty-vs-repair experiment (T3.2) finished.
- 2026-11-24 — T2.6 Pyodide spike: ~2× slowdown, bit-identical results to native; effort presets set (D21). **Phase 2 complete.**
- 2026-11-24 — Phase 2 unit 3 (PSO, `b19f7f0`): tests → RED → implemented → pilot defaults → review APPROVE → mutant-killing tests; GA got the same pilot sweep (k = 5). 371 tests. **All three algorithms done.**
- 2026-11-24 — Phase 2 unit 2 (ACO, `c73146a`): defaults fixed after stagnation (pilot sweep) → review APPROVE with 3 MEDIUM → fixed test-first. (Progress entry added late: the original update was skipped when `make lint` failed in the same script.)
- 2026-11-24 — Phase 2 unit 1 (interface + GA): contract + GA tests → RED → implemented → review APPROVE with 4 MEDIUM → fixed test-first → 232 tests, 98% coverage.
- 2026-11-24 — T1.9 fold plot + T1.1 Makefile: review round 1 (6 MEDIUM) → fixed test-first → round 2 APPROVE. 159 tests. **Phase 1 complete.**
- 2026-11-24 — T1.8 benchmarks: 21 instances generated by script, cited and cross-checked; harness run 22 RED re-demonstrated with user approval (sequencing slip); review APPROVE; golden pins added. 142 tests.
- 2026-11-24 — Review round 1 (WARNING: H1, M1–M3, L1–L5) → fixes test-first (new tests observed RED → GREEN) → round 2 APPROVE. 114 tests, 99% coverage, Bandit clean. Mutants (D-frame, U-frame, R=L) now caught.
- 2026-11-24 — Phase 1 unit 1: 5 test files written → observed RED (harness) → Gate 1 approved → hp_model/folds/fitness implemented → 76 tests pass, coverage 98% (core), Bandit clean → code review requested.
- 2026-11-24 — `harness init` done by the user (`d6b01ee`). Config fixed (D5); uv env created; waiting on B2.
- 2026-11-24 — Plan created. Phase 0 done (commit `80dffa5`). Harness specs registered. Blocked on B1 (`harness init`).
