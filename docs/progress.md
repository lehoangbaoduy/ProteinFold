# ProteinFold — Implementation Plan & Progress

Living document: always reflects current status. Spec: [PRD.md](PRD.md) · Brief: [HANDOFF.md](HANDOFF.md)

**Legend:** ☐ todo · ◐ in progress · ☑ done (verified) · ⊘ blocked · — skipped (with reason)

**Process per task (master prompt):** harness-os pipeline: Constitution → Specification → Test Suite (observed RED) → Generate Code (observed GREEN) → Code Review (correctness, performance, security). Then update this file.

---

## Current status

| | |
|---|---|
| **Phase** | 1 — Core |
| **Last update** | 2026-11-24 |
| **Next action** | ⊘ The user must run `harness init` (interactive and human-only by design). See Blockers. |

### Blockers
| ID | Blocker | Needed from | Since |
|---|---|---|---|
| B1 | Harness-os gating is not initialized for this project: no `.claude/harness.config.json` or `enforce-gate.sh` hooks. The `new-feature` workflow (run 21) is waiting for a host-observed RED test run, which only those hooks can record. `harness init` is TTY-gated, so Claude cannot run it. | User | 2026-11-24 |

### Harness-os state
| Item | Value |
|---|---|
| Product spec | `PFOLD-PROD-001` v1 (active) |
| Phase 1 domain spec | `PFOLD-DOM-001` v1 (active): HP model core |
| Workflow | run 21, `new-feature`, stage `tests` (waiting for RED) |
| Risk assessment | #28, medium: gates `spec`, `tests`, `review:code`. Assessment #27 ("high") was a false positive: the keyword matched the phrase "no secrets" in the description, and the change was reassessed with the same scope. |

---

## Phase 0 — Repository setup ☑
| ID | Task | Files | Status |
|---|---|---|---|
| T0.1 | git init (`main`), repo-local identity, `.gitignore`, initial commit of docs | `.gitignore`, `docs/HANDOFF.md`, `docs/PRD.md` | ☑ commit `80dffa5` |
| T0.2 | Register product spec and Phase 1 domain spec in harness-os | — | ☑ `PFOLD-PROD-001`, `PFOLD-DOM-001` |
| T0.3 | Harness-os project init (gating hooks + config) | `.claude/harness.config.json`, `.claude/hooks/*`, `.claude/settings.json` | ⊘ B1 (user) |

## Phase 1 — Core (PRD §10 row 1)
**Exit criteria:** E = −3 check passes; brute-force energy tests pass; benchmarks cited; ≥ 90% coverage on the HP model.

| ID | Task | Files | Tests (written first) | Status |
|---|---|---|---|---|
| T1.1 | Python project: uv env + lockfile, dependencies (numpy; dev: pytest, pytest-cov, hypothesis, scipy, pandas, matplotlib, bandit, pip-audit), Makefile targets `test`, `cov`, `lint`, `audit` | `pyproject.toml`, `uv.lock`, `Makefile`, `src/hpfold/__init__.py`, `src/hpfold_tools/__init__.py` | `make test` runs | ☐ |
| T1.2 | `parse_sequence` (FR-8): strip whitespace, uppercase, reject non-H/P and empty input | `src/hpfold/hp_model.py` | `tests/test_hp_model.py::test_parse_*` | ☐ |
| T1.3 | `decode`/`encode` with the (heading, up) frame, for 2D and 3D (FR-1, FR-2) | `src/hpfold/hp_model.py` | hand-checked coordinates; 3D U/D/L frame cases; round trip | ☐ |
| T1.4 | `is_valid`, `collisions`, `contacts`, `energy` with an O(n) hash map (FR-3, FR-4) | `src/hpfold/hp_model.py` | HPHHHHPHHP → −3 with contacts {(0,5),(3,8),(4,7)}; bonded pairs never count; brute-force O(n²) cross-check; exhaustive enumeration for small n | ☐ |
| T1.5 | `evaluate(seq, moves, strategy, penalty)` → Evaluation (FR-5, §5.2) | `src/hpfold/hp_model.py` | penalty arithmetic; invalid flagged; repair strategy returns a valid fold | ☐ |
| T1.6 | `random_valid_fold` (backtracking SAW) and `repair` (FR-6, FR-7) | `src/hpfold/folds.py` | always valid for 2D n ≤ 100 and 3D n ≤ 100; deterministic for a given seed; repair output valid and same length | ☐ |
| T1.7 | Hypothesis properties: unit bonds; `encode(decode(m)) == m`; energy ≤ 0; energy == brute force; repair(m) valid | `tests/test_hp_properties.py` | (this is the test) | ☐ |
| T1.8 | Benchmarks from the primary literature: 2D set (8–11 sequences, n ≈ 20–100) and 3D subset (3–5), each with a citation; source disagreements recorded (FR-B1 to B5) | `data/benchmarks.json`, `docs/benchmarks_sources.md` | `tests/test_benchmarks.py`: alphabet, stated length == len(seq), every entry has a source, energies ≤ 0 | ☐ |
| T1.9 | Matplotlib fold plot, 2D and 3D (tooling package) | `src/hpfold_tools/visualize.py` | `tests/test_visualize.py` smoke test (writes PNG to tmp_path) | ☐ |
| T1.10 | Purity guard: the core package imports only stdlib + numpy | `tests/test_core_purity.py` | AST scan of `src/hpfold/**` imports | ☐ |

## Phase 2 — Algorithms (PRD §10 row 2)
**Exit criteria:** all three reach a reasonable energy on a 20-mer in 2D and 3D; seeds reproduce; budget exact; cancel and progress frames work.

| ID | Task | Files | Tests (written first) | Status |
|---|---|---|---|---|
| T2.1 | Shared interface: `RunResult`, `Frame`, evaluation counter with an exact budget, `target`, `on_progress`, `should_stop`; NA result when no valid fold is found | `src/hpfold/algorithms/base.py` | `tests/test_algorithm_contract.py` (parametrized over every algorithm): budget never exceeded and exactly reached unless the target is hit or the run is cancelled; same seed gives an identical result; frames are monotone; cancel returns best-so-far; NA when no valid fold | ☐ |
| T2.2 | GA (§5.3): tournament, 1-point/2-point/uniform crossover, per-gene mutation, elitism, penalty or repair | `src/hpfold/algorithms/ga.py` | `tests/test_ga.py` + contract | ☐ |
| T2.3 | ACO (§5.3): τ[i][m], heuristic η from new H–H contacts, bounded backtracking + restart, evaporation, deposit rule, optional MMAS bounds; one evaluation per completed fold | `src/hpfold/algorithms/aco.py` | `tests/test_aco.py`: always valid folds; eval count == completed ants; τ stays within bounds + contract | ☐ |
| T2.4 | PSO (§5.3): probability matrices, w/c₁/c₂ update, row renormalization, floor ε, sampling, penalty or repair | `src/hpfold/algorithms/pso.py` | `tests/test_pso.py`: rows sum to 1; min ≥ ε; no one-hot collapse + contract | ☐ |
| T2.5 | Frame `matrix` + `diversity` per algorithm (FR-T1) | the three algorithm files | frame schema test | ☐ |
| T2.6 | Pyodide timing spike: run GA/ACO/PSO in Pyodide (Node or browser) at n = 20, 48, 100; set effort multipliers and the length cap | `web/spike/` (throwaway), results → Decisions log | timings recorded | ☐ |
| T2.7 | *(Could)* Pull-move local search + hybrid ACO-LS; LS evaluations count toward the budget | `src/hpfold/local_search.py`, `src/hpfold/algorithms/hybrid.py` | `tests/test_local_search.py`: moves keep validity; energy never worsens | ☐ |

## Phase 3 — Experiments (PRD §10 row 3)
**Exit criteria:** `results/runs.csv` complete; tuned defaults exported.

| ID | Task | Files | Tests | Status |
|---|---|---|---|---|
| T3.1 | Experiment runner: TOML configs, process pool, resume (skip finished rows), CSV schema FR-E3, budget B(n) = c·n², config + git hash recorded | `src/hpfold_tools/experiments.py`, `configs/*.toml` | `tests/test_experiments.py`: schema; resume skips; same seed + config gives an identical row | ☐ |
| T3.2 | Penalty vs. repair mini-experiment (§7.4) → choose a strategy | `configs/penalty_repair.toml`, `results/penalty_repair.csv` | — | ☐ |
| T3.3 | Tuning with equal effort per algorithm (§7.3) → `results/tuned_params.json` | `configs/tuning.toml` | — | ☐ |
| T3.4 | Main 2D study: 30 seeds × 3 algorithms × all 2D benchmarks | `configs/main_2d.toml`, `results/runs.csv` | row-count check | ☐ |
| T3.5 | 3D extension: 30 seeds × 3 algorithms × 3D subset | `configs/ext_3d.toml` | row-count check | ☐ |
| T3.6 | Web export: traces (median + best run, < 2 MB each), benchmarks JSON, `defaults.json` (FR-E6), summary JSON | `src/hpfold_tools/export_web.py`, `web/public/data/` | `tests/test_export_web.py`: size cap, schema | ☐ |

## Phase 4 — Analysis (PRD §10 row 4)
**Exit criteria:** every paper figure and table is regenerated by `make figures`.

| ID | Task | Files | Tests | Status |
|---|---|---|---|---|
| T4.1 | Statistics: summary (FR-A1), Wilcoxon rank-sum + Holm, A₁₂, optional Friedman | `src/hpfold_tools/analysis.py` | `tests/test_analysis.py`: hand-computed A₁₂, Holm ordering, success rate, evals-to-best on successes only, NA handling | ☐ |
| T4.2 | Figures: convergence (median + IQR), box/violin, best folds 2D/3D | `src/hpfold_tools/figures.py`, `results/figures/` | smoke tests | ☐ |
| T4.3 | LaTeX tables (booktabs) from the CSV | `src/hpfold_tools/tables.py`, `paper/tables/` | snapshot test on a tiny CSV | ☐ |
| T4.4 | Sensitivity study (§7.3) | `configs/sensitivity.toml`, `results/sensitivity.csv`, figure | — | ☐ |
| T4.5 | Talk animation (GIF/MP4) | `src/hpfold_tools/animate.py`, `results/figures/anim_*.gif` | smoke test | ☐ |

## Phase 5 — Webpage (PRD §10 row 5)
**Exit criteria:** §14 webpage criteria met; offline demo verified; all checks done with Playwright screenshots.

| ID | Task | Files | Tests / verification | Status |
|---|---|---|---|---|
| T5.1 | Scaffold: Vite + TS, ESLint (+ security plugin), vitest, Playwright, CSP meta compatible with WASM | `web/package.json`, `web/vite.config.ts`, `web/index.html`, `web/eslint.config.js` | `npm test`, `npm run lint` | ☐ |
| T5.2 | Package hpfold as a wheel for Pyodide; self-host Pyodide | `scripts/build_web_wheel.py`, `web/public/py/`, `web/public/pyodide/` | worker loads offline | ☐ |
| T5.3 | Solver worker: run jobs, post frames, cancel, error reporting | `web/src/worker/solver.worker.ts`, `web/src/worker/protocol.ts` | vitest on the protocol; Playwright run | ☐ |
| T5.4 | Input + validation (FR-L1): normalize, H/P only, 3–100, presets, random | `web/src/solver/input.ts` | vitest + fast-check fuzzing of the parser (no throw, never accepts invalid input) | ☐ |
| T5.5 | Options (FR-L2): lattice, algorithms, effort, seed, advanced params from `defaults.json` | `web/src/solver/options.ts` | vitest | ☐ |
| T5.6 | Live view (FR-L4): 2D fold SVG, convergence chart, internals heatmap, race layout, progress + elapsed time | `web/src/views/fold2d.ts`, `convergence.ts`, `heatmap.ts`, `race.ts` | Playwright screenshots while running | ☐ |
| T5.7 | Results (FR-L5): 3D fold (three.js), metrics panel, comparison table, benchmark badge + citation | `web/src/views/fold3d.ts`, `metrics.ts`, `compare.ts` | Playwright screenshots | ☐ |
| T5.8 | Cancel, Pyodide loading state (FR-L7), honesty note (FR-L8), "no valid fold found" state | `web/src/solver/*` | Playwright | ☐ |
| T5.9 | Export + share URL codec (FR-L6), treating URL input as untrusted | `web/src/share/url.ts` | vitest + fuzzing of the decoder | ☐ |
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
| Q1 | Python SAST | `bandit -r src` (`make lint`) | ☐ |
| Q2 | Python SCA | `pip-audit` (`make audit`) | ☐ |
| Q3 | JS SAST + SCA | ESLint security rules, `npm audit` | ☐ (Phase 5) |
| Q4 | Semgrep (Python + TS rulesets) | `semgrep --config auto` | ☐ |
| Q5 | Fuzzing | Hypothesis (core), fast-check (web parser and URL decoder) | ☐ |
| Q6 | Dependabot | `.github/dependabot.yml`. Takes effect only once a GitHub remote exists. | ☐ |
| Q7 | DAST (optional) | OWASP ZAP baseline against the deployed site | ☐ |
| Q8 | Coverage ≥ 90% on `hp_model`, ≥ 80% overall | `pytest --cov` + `validate_coverage` | ☐ |

---

## Edge-case catalog (PRD §8) → required tests
Every row is a required task.

| # | Case | Expected | Test location | Status |
|---|---|---|---|---|
| E1 | Characters other than H/P; empty input | Reject with a clear message | `tests/test_hp_model.py::test_parse_rejects_*`; `web/src/solver/input.test.ts` | ☐ |
| E2 | Lowercase / whitespace | Normalized | `tests/test_hp_model.py::test_parse_normalizes`; web input test | ☐ |
| E3 | n < 3 | Library: trivial fold, E = 0; web: rejected | `tests/test_hp_model.py::test_short_sequences`; web input test | ☐ |
| E4 | n > 100 on the web | Rejected with an explanation (CLI has no cap) | web input test; `tests/test_algorithm_contract.py::test_no_cap_in_library` | ☐ |
| E5 | No H / single H | E = 0; runs normally; the page notes it | `tests/test_algorithm_contract.py::test_no_contacts_possible`; Playwright | ☐ |
| E6 | All-H sequence | Valid stress test | `tests/test_algorithm_contract.py::test_all_h` | ☐ |
| E7 | Never finds a valid fold (penalty) | best_energy = NA; counted as a failure; page says "no valid fold found" | `tests/test_algorithm_contract.py::test_no_valid_fold_is_na`; `tests/test_analysis.py::test_na_is_failure`; Playwright | ☐ |
| E8 | ACO ant dead end | Bounded backtracking → restart; restarts logged | `tests/test_aco.py::test_dead_end_restart` | ☐ |
| E9 | PSO one-hot collapse | Prevented by ε; diversity logged | `tests/test_pso.py::test_floor_prevents_collapse` | ☐ |
| E10 | Fitness ties | Deterministic (stable sort) | `tests/test_algorithm_contract.py::test_determinism` | ☐ |
| E11 | Disputed best-known / 3D lower bound | Both stored and cited; success uses the better value; flagged | `tests/test_benchmarks.py`; `tests/test_analysis.py::test_success_uses_best_value` | ☐ |
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
| D4 | 2026-11-24 | Harness-os spec IDs use the prefix `PFOLD` (`PFOLD-PROD-001`, `PFOLD-DOM-001`, …) | Schema requires `^[A-Z][A-Z0-9]{1,9}-<TYPE>-NNN$` |

Open details to decide during implementation (to be logged here): penalty λ default; ACO γ, deposit rule, τ bounds; PSO ε, w, c₁, c₂; budget constant c in B(n) = c·n²; trace frame interval k; web effort multipliers and length cap (after T2.6); choice between GitHub Pages and Vercel.

---

## Changelog
- 2026-11-24 — Plan created. Phase 0 done (commit `80dffa5`). Harness specs registered. Blocked on B1 (`harness init`).
