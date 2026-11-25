# PRD: HP Lattice Protein Folding with GA, ACO and PSO

| | |
|---|---|
| **Course** | EECS 6180 – Biologically Inspired Computing (University of Toledo, Dr. Devinder Kaur) |
| **Deliverables** | Python software, IEEE-style technical paper (15% of grade), ~15-minute oral presentation, public interactive webpage |
| **Status** | Draft v2, 2026-11-24 |
| **Timeline** | No fixed dates. Work proceeds phase by phase (§10) until done. |
| **Source brief** | [HANDOFF.md](HANDOFF.md) |

---

## 1. Summary

Fold HP protein sequences on the 2D square lattice (main study) and the 3D cubic lattice (extension) using three nature-inspired metaheuristics: a **Genetic Algorithm (GA)**, **Ant Colony Optimization (ACO)** and a **probability-based discrete Particle Swarm Optimization (PSO)**. Compare them on standard literature benchmarks, under equal evaluation budgets, with sound statistics. Write it up as an IEEE-style paper.

The project also ships a **public webpage** where anyone can type an HP sequence and watch GA, ACO and PSO search for its lowest-energy fold live in the browser. The page shows the resulting 2D and 3D folds with their metrics. A second tab presents the paper's benchmark results.

> **Terminology: these are search algorithms, not trained models.** Nothing is trained in advance. For every sequence, each algorithm starts from scratch, generates candidate folds, scores them with the energy function, and improves them over iterations. What it learns during a run (the GA's population, ACO's pheromone, PSO's probabilities) is discarded afterwards. The only offline step is **parameter tuning**: choosing good settings on benchmark sequences, which the webpage then uses as defaults. Say "the algorithm searches for the lowest-energy fold", not "the model predicts the fold".

## 2. Goals and non-goals

### Goals
1. A correct, tested HP model (2D and 3D): move decoding, validity check, energy.
2. Working implementations of GA, ACO and PSO that share one interface and one evaluation counter.
3. A reproducible experiment pipeline: 30 seeded runs per algorithm per sequence, with results saved to CSV and traces saved to JSON.
4. A fair comparison: same fitness-evaluation budget, same benchmarks, same reporting.
5. Statistical analysis, convergence plots, a parameter-sensitivity study, and a penalty-vs-repair mini-experiment.
6. Figures for the paper and talk.
7. A public webpage with:
   - a **live solver**: the user inputs a sequence, and the real Python algorithms run in the browser and output 2D/3D folds plus metrics;
   - a **paper results** section: replays of saved benchmark runs and the statistics dashboard.

### Non-goals
- Predicting real protein structures. HP is a simplified model.
- Training a machine-learning model (neural network, reinforcement learning) to fold sequences.
- Beating state-of-the-art HP solvers. The aim is a sound comparative study, not a record.
- Re-implementing the algorithms in JavaScript. The webpage runs the same Python code as the experiments.
- A server backend. The site is static.
- Off-lattice, 20-letter, or other lattice types (e.g., triangular, FCC).

## 3. Background and problem definition

- **Input:** a string over {H, P} of length *n* (H = hydrophobic, P = polar).
- **Output:** a lattice coordinate for each bead such that the fold is
  1. **connected:** consecutive beads sit on adjacent lattice sites; and
  2. **self-avoiding:** no two beads share a site.
- **Objective:** minimize *E* = −(number of topological H–H contacts). A topological contact is a pair of H beads on adjacent sites that are *not* consecutive in the chain.
- **Complexity:** NP-hard in 2D and 3D (Crescenzi et al. 1998; Berger & Leighton 1998). This justifies using metaheuristics.

### Representation (shared by all algorithms)
- **Relative move encoding:** *n* − 2 free moves. The first bond is fixed along +x, which removes rotational symmetry.
  - 2D alphabet: {F, L, R}
  - 3D alphabet: {F, L, R, U, D}. The decoder tracks both the chain's heading and an "up" direction, so L/R/U/D are well defined. 2D is the special case of 3D that never uses U or D.
- **Symmetry note:** mirror images still encode distinct strings with equal energy. Optional: force the first non-F move to be L, halving the search space. Record in the paper whether this is used.

### Validation case (hand-checked)
`HPHHHHPHHP` with fold (0,0)→(1,0)→(2,0)→(2,1)→(1,1)→(0,1)→(0,2)→(1,2)→(2,2)→(3,2) must give **E = −3**.

## 4. Scope and priorities

| Priority | Item |
|---|---|
| **Must** | HP model 2D+3D with unit tests; GA, ACO, PSO; 2D benchmark study (30 runs each); stats; convergence plots; fold figures; paper; talk; **webpage live solver (2D + 3D)**; webpage paper-results tab (replay + summary) |
| **Should** | 3D extension on a subset of sequences; penalty-vs-repair mini-experiment; parameter sensitivity; webpage race mode; shareable result URLs |
| **Could** | Hybrid ACO (or GA) + pull-move local search as a fourth method; mirror-symmetry breaking; GPU-batched evaluation for experiments |
| **Won't (this term)** | Trained ML folding model; server backend; JavaScript port; other lattices; real-protein validation |

## 5. Functional requirements

### 5.1 HP model core (`hp_model`)
- **FR-1** `decode(moves, dim)` → list of integer coordinates (length *n*).
- **FR-2** `encode(coords)` → moves (inverse of decode, for tests and imported folds).
- **FR-3** `is_valid(coords)`, `collisions(coords)` (number of beads on already-occupied sites).
- **FR-4** `contacts(seq, coords)` → list of (i, j) H–H topological contacts; `energy = −len(contacts)`. Must run in O(n) using a site→bead hash map.
- **FR-5** `evaluate(seq, moves, strategy)` → energy, collisions, fitness. Each call counts as **one fitness evaluation**.
- **FR-6** `random_valid_fold(n, rng, dim)`: random self-avoiding walk with backtracking (for initialization and repair).
- **FR-7** `repair(moves)`: turn an invalid fold into a valid one (e.g., at each collision, re-pick the move from the remaining valid options, backtracking if stuck). Deterministic given the RNG.
- **FR-8** `parse_sequence(s)`: strip whitespace, uppercase, reject anything other than H/P with a clear message. Used by both the CLI and the webpage.

### 5.2 Constraint handling (penalty vs repair)
- **Penalty:** fitness = E + λ·collisions, with λ > 0 (default to be tuned; λ must be large enough that a valid fold beats any invalid one with similar contacts).
- **Repair:** invalid offspring are repaired before evaluation. The repaired genome is written back into the population (Lamarckian).
- Main runs (and the webpage defaults) use whichever strategy wins the mini-experiment (§7.4), applied the same way to GA and PSO. ACO builds valid folds by construction and needs neither.
- **Only valid folds count as results.** The best-so-far energy tracks valid folds only.

### 5.3 Algorithms (shared interface)
```
run(sequence, dim, params, seed, budget, target=None,
    on_progress=None, should_stop=None) -> RunResult
RunResult: algorithm, sequence, dim, seed, params, best_moves, best_energy,
           evaluations, evals_to_best, runtime_s, history[(evals, best_energy)]
```
- Every algorithm stops at the same **fitness-evaluation budget**. It may also stop early when `target` (the best-known energy) is reached.
- All randomness comes from a single `numpy.random.Generator(seed)`.
- `on_progress(frame)` is called once per iteration (or at a configurable interval) with a frame dict (§5.6). The experiments use it for traces and the webpage uses it for live updates.
- `should_stop()` is checked once per iteration so the webpage's **Cancel** button can end a run cleanly. A cancelled run returns its best so far.
- **Browser-compatible:** the core package (`hp_model`, `algorithms`) uses only the Python standard library and numpy, with no multiprocessing, file I/O, or matplotlib, so it runs unmodified in Pyodide. Parallelism and file output live in `experiments.py`, outside the core.

#### GA (baseline)
- Chromosome: move string. Initialization: random valid folds.
- Selection: tournament (k = 3, tunable). Crossover: one-point (two-point/uniform as options), rate *p_c*. Mutation: per-gene re-draw to a different move, rate *p_m* (default ≈ 2/L). Elitism: top *e* copied unchanged.
- Parameters: population size, *p_c*, *p_m*, *k*, *e*.

#### ACO (after Shmygelska & Hoos 2005)
- Pheromone τ[i][m]: one value per position *i* and relative move *m*.
- **Construction:** each ant grows a fold bead by bead. At each step it picks a move from the moves that lead to free sites, with probability ∝ τ^α · η^β. The heuristic η increases with the number of *new* H–H contacts the move would create (e.g., η = exp(new_contacts / γ)).
- **Dead ends:** backtrack a limited number of steps; if it still fails, restart the ant. Construction work does **not** count as evaluations. Each completed fold counts as **one** evaluation.
- **Update:** evaporation (1 − ρ), then deposits from the iteration-best and/or global-best ants, scaled by relative solution quality. Optional bounds τ_min/τ_max (MMAS style) to avoid stagnation.
- Parameters: number of ants, α, β, ρ, deposit rule, τ bounds.
- *Note:* the original method grows folds in both directions from a random start bead and uses local search. Pure ACO here is the core method; ACO + local search is the "Could" hybrid. The paper must state this difference from the reference.

#### PSO (probability-based discrete PSO)
- Each particle *k* holds a probability matrix P_k[i][m] (one distribution over moves for each position), plus a personal-best fold and a shared global-best fold.
- **Update:** P_k ← w·P_k + c₁r₁·onehot(pbest_k) + c₂r₂·onehot(gbest), then renormalize each row. Apply a probability floor ε so no move ever reaches zero, which preserves exploration.
- **Position:** sample a fold from P_k. The penalty or repair strategy applies as for the GA.
- Parameters: swarm size, w, c₁, c₂, ε, topology (global; ring optional).
- The paper explains how this maps onto continuous PSO: position → probability vector, velocity → drift toward the bests.

#### Hybrid (Could)
- Pull-move local search (Lesh, Mitzenmacher & Whitesides 2003) applied to the best k folds each iteration. Each local-search evaluation counts toward the budget.

### 5.4 Benchmarks (`data/benchmarks.*`)
- **FR-B1** 2D: the standard benchmark set from the literature, about 8–11 sequences with lengths of roughly 20–100. Each entry: id, sequence, length, best-known energy, and source citation.
- **FR-B2** 3D: a subset (e.g., 3–5 sequences) with published 3D best-known energies or lower bounds, each cited.
- **FR-B3** Every sequence and energy is **copied from and checked against the primary papers** (e.g., Unger & Moult 1993; Shmygelska & Hoos 2005, which compiles prior results). Nothing comes from memory. A unit test checks lengths and alphabet.
- **FR-B4** Where sources disagree, record both values and cite them. "Success" uses the lower (better) value.
- **FR-B5** The benchmark file is also exported to JSON for the webpage's presets and "best-known" badges.

### 5.5 Experiments (`experiments`)
- **FR-E1** Run grid: {GA, ACO, PSO} × benchmark sequences × 30 seeds (seeds 0–29, shared across algorithms).
- **FR-E2** Parallel execution: one process per run, using all CPU cores. Runs can be resumed: completed (algorithm, sequence, seed) rows are skipped.
- **FR-E3** Output `results/runs.csv`: algorithm, dim, seq_id, n, seed, budget, best_energy, best_moves, evaluations, evals_to_best, runtime_s, params_hash, plus a separate params JSON.
- **FR-E4** Budget per sequence scales with length (e.g., B(n) = c·n² evaluations, with *c* fixed in the config) and is identical across algorithms.
- **FR-E5** Configs live in version-controlled YAML/TOML. Each results file records the config and git commit hash.
- **FR-E6** The tuned parameters are exported to `web/public/data/defaults.json`, so the webpage uses the same settings as the paper.

### 5.6 Progress frames and trace export
- **FR-T1** A frame is a dict emitted by `on_progress`:
  ```json
  {"iter": 6, "evals": 1200, "budget": 40000, "best_energy": -5,
   "best_moves": "FLR...", "current_best_moves": "...",
   "diversity": 0.41, "matrix": [[0.33, 0.33, 0.34]]}
  ```
  `matrix` holds the ACO pheromone matrix or the PSO global-best particle's probability matrix, or for the GA the population's move-frequency matrix (normalized, rounded to 3 decimal places). Values are illustrative only.
- **FR-T2** For the paper-results tab, the experiments save the frames of selected runs (median and best run per algorithm per sequence) as JSON traces. Each trace records a frame on every improvement plus every *k*-th iteration, and stays under **2 MB**.

### 5.7 Analysis and figures
- **FR-A1** Per (algorithm, sequence): best, mean ± std, median, worst energy; success rate (share of runs reaching the best-known energy); mean evaluations-to-best on successful runs; mean runtime.
- **FR-A2** Statistics: pairwise two-sided Wilcoxon rank-sum test per sequence with Holm correction; effect size (Vargha–Delaney A₁₂). Optional: Friedman test plus a post-hoc test across sequences for an overall ranking.
- **FR-A3** Convergence plots: median best-so-far energy vs. evaluations with an interquartile band, one panel per sequence, all three algorithms overlaid.
- **FR-A4** Box/violin plots of final energies.
- **FR-A5** Fold figures: best fold per sequence per algorithm (2D grid; 3D projection) with H/P styling and contact lines.
- **FR-A6** ~~Animation (GIF/MP4) of a fold improving over a run, for the talk.~~ Dropped by the user (progress.md D30); the talk is given in person and the live page shows the search.
- **FR-A7** LaTeX-ready tables (booktabs) generated from the CSV, so no numbers are typed by hand.

### 5.8 Webpage

A public static site with two main sections.

#### 5.8.1 Live solver (main feature)
User flow: **enter sequence → choose options → Run → watch the search live → see results**.

- **FR-L1 Input**
  - Text box for an HP sequence. Whitespace is ignored and the input is case-insensitive.
  - Validated inline: only H and P allowed, length 3–100 (longer inputs are rejected with an explanation).
  - A preset dropdown loads benchmark sequences, showing each one's best-known energy and source.
  - A "Random" button generates a random sequence of a chosen length.
- **FR-L2 Options**
  - Lattice: 2D, 3D, or both.
  - Algorithms: any combination of GA, ACO and PSO (default: all three).
  - Effort: Quick / Standard / Thorough. Each sets an evaluation budget as a multiple of B(n), with an estimated time shown.
  - Seed: random by default, but shown and editable so any result can be reproduced.
  - Collapsed "Advanced" panel: per-algorithm parameters, pre-filled with the tuned defaults (FR-E6).
- **FR-L3 Execution**
  - The real Python package runs in the browser via **Pyodide**, inside **Web Workers**, so the page never freezes.
  - One worker per (algorithm, lattice) job, up to the number of CPU cores; any extra jobs wait in a queue.
  - Per job: a progress bar (evaluations / budget) and elapsed time.
  - **Cancel** stops all jobs and keeps the best result so far.
- **FR-L4 Live view while running**, updated from the progress frames:
  - best-so-far fold, redrawn as it improves
  - convergence curve growing in real time
  - internals heatmap: ACO pheromone, PSO probabilities, or GA population move frequencies
  - When several algorithms run, they're shown side by side ("race"), aligned by evaluation count.
- **FR-L5 Results**, for each (algorithm, lattice):
  - **2D fold** (SVG: H/P beads, chain, dashed H–H contacts, bead 0 marked) and **3D fold** (three.js: rotate, zoom, reset view)
  - **Metrics:** energy E, number of H–H contacts, fold validity, evaluations used, evaluations to best, runtime, seed, move string
  - **Comparison table** across algorithms, with the best highlighted
  - If the sequence matches a benchmark: best-known energy, a "reached / gap = x" badge, and the citation
- **FR-L6 Export and share:** download fold PNG/SVG and a results JSON; a shareable URL encodes the sequence, options and seed, so the same run can be reproduced.
- **FR-L7 Loading:** on first visit, show Pyodide's loading progress (roughly 10–20 MB download, then cached by the browser). The Run button is disabled until the runtime is ready.
- **FR-L8 Honesty note** on the page: a single run is one random sample; the paper's conclusions rest on 30 runs per algorithm.

#### 5.8.2 Paper results
- **FR-W1 Replay:** pick lattice, sequence, algorithm, and run (best/median); play/pause/scrub the saved trace, using the same fold, convergence and internals views as the live solver.
- **FR-W2 Dashboard:** summary table, box plots, success rates and statistical tests from `runs.csv` (exported to JSON).

#### 5.8.3 About
- **FR-X1 Explainers:** short illustrated sections on the HP model, the "search, not training" idea, and how each algorithm works. Written for a general audience.

#### 5.8.4 Webpage non-functional requirements
- **Responsiveness:** the UI stays at 60 fps while solving. Computation happens only in workers.
- **Performance targets:**
  - n = 20, Standard effort, one algorithm: under about 5 s on a mid-range laptop
  - n = 100, Thorough effort: under about 2 minutes, with live progress throughout
  - Measure real timings early (Phase 5 spike) and calibrate the effort presets from them.
- **Consistency:** a native-vs-Pyodide parity test confirms that the same (sequence, params, seed) gives the same result in both.
- **Hosting:** static deployment to GitHub Pages or Vercel. No backend and no user data stored.
- **Offline demo mode:** for the talk, Pyodide is self-hosted alongside the site rather than loaded from a CDN, so the demo also works from `localhost` with no internet.
- **Display:** usable at 1280×720 (projector) and on laptops; on phones it works but desktop is the primary target.
- **Accessibility:** H/P shown by fill and shape, not hue alone; keyboard-operable controls; the light theme works on a projector.
- **Tech:** Vite + TypeScript, Pyodide in Web Workers, SVG for 2D folds, three.js for 3D, a lightweight chart library (or hand-rolled SVG).

## 6. Non-functional requirements (project)
- **Language:** Python ≥ 3.11 (numpy, scipy, pandas, matplotlib, pytest). Confirmed acceptable by the instructor.
- **AI assistance:** permitted by the instructor. Code is still explained and documented so it can be defended in the talk.
- **Reproducibility:** fixed seeds; deterministic given (config, seed); pinned dependencies (`uv` lockfile); one command per stage (`make test`, `make experiments`, `make figures`, `make web-data`, `make web`).
- **Correctness:** at least 90% test coverage on `hp_model`; brute-force cross-checks of energy; exhaustive enumeration on small *n* to confirm true optima.
- **Performance:** decode + energy in O(n). Target: the full 2D experiment grid finishes overnight on a desktop CPU. The RTX 4070 Ti is **not required**. Runs are many small independent jobs, so CPU multiprocessing is the main lever. GPU batch evaluation is a "Could" item only if profiling shows evaluation dominates.
- **Fairness of runtime:** all algorithms share the same language, evaluator, and single-core-per-run setup. Runtimes are reported as wall-clock time per run on one stated machine.
- **Code clarity:** each algorithm file is readable alongside its paper pseudocode, with docstrings that state the mapping from the paper.

## 7. Experiments plan

### 7.1 Main study (2D)
All benchmark sequences × {GA, ACO, PSO} × 30 seeds × equal budget B(n).

### 7.2 3D extension
3–5 sequences × 3 algorithms × 30 seeds.

### 7.3 Parameter tuning and sensitivity
- **Tuning:** a small grid or random search per algorithm on 2 tuning sequences (a short and a medium one), with 10 seeds each. Every algorithm gets the same tuning effort. Tuned values become the experiment and webpage defaults.
- **Sensitivity:** one factor at a time around the tuned values:
  - GA: population size, *p_m*, *p_c*
  - ACO: ρ (evaporation), α/β, number of ants
  - PSO: swarm size, w, ε

  Plot final mean energy vs. each parameter, using 3 sequences and 15 seeds.

### 7.4 Penalty vs. repair mini-experiment
GA and PSO under {penalty λ ∈ {1, 2, 5}, repair} on 3 sequences × 30 seeds. Report energy, success rate, and the fraction of invalid individuals per generation.

### 7.5 (Could) Hybrid
ACO + pull-move local search vs. pure ACO on the 2D set.

## 8. Edge cases and validation rules
| Case | Expected behavior |
|---|---|
| Characters other than H/P; empty input | Reject with a clear message (webpage: inline, before running) |
| Lowercase or whitespace in input | Normalized (uppercased, whitespace stripped) |
| n < 3 | Rejected in the webpage. In the library: no free moves, trivial fold, E = 0 |
| n > 100 in the webpage | Rejected with an explanation (CLI has no cap) |
| No H or a single H | Optimum is E = 0. Runs normally; the page notes that no contacts are possible |
| All-H sequence | Valid input; used as a stress test |
| Algorithm never finds a valid fold (penalty mode) | best_energy = NA; counted as a failure, never as 0. Webpage shows "no valid fold found" |
| ACO ant dead end | Bounded backtracking, then restart; restart count logged |
| PSO probabilities collapse to one-hot | Prevented by the floor ε; diversity logged |
| Ties in fitness | Broken deterministically (stable sort) for reproducibility |
| Best-known energy disputed or only a lower bound (3D) | Store and cite both; success uses the better value; flagged in tables and on the page |
| Target reached early | Stop; evals_to_best recorded; webpage shows "reached best-known" |
| User cancels or closes the tab mid-run | Workers terminated; partial best shown on cancel |
| Pyodide fails to load (old browser, blocked CDN) | Clear error message; the paper-results tab still works because it is plain JSON |
| Long sequences (n ≈ 100) miss the best-known energy | Reported honestly, in the paper and on the page |

## 9. Repository structure
```
ProteinFold/
  docs/              HANDOFF.md, PRD.md
  src/hpfold/        pure Python + numpy; runs natively and in Pyodide
    hp_model.py      lattice, decode/encode, validity, energy, repair, parse_sequence
    algorithms/      base.py, ga.py, aco.py, pso.py, (hybrid.py)
  src/hpfold_tools/  native-only tooling
    experiments.py   batch runner, CLI, multiprocessing, resume
    analysis.py      stats, tables
    visualize.py     matplotlib fold plots, convergence
    export_web.py    traces, summaries, benchmarks and defaults -> web/public/data
  data/              benchmarks (with citations)
  configs/           main_2d.toml, ext_3d.toml, sensitivity.toml, penalty_repair.toml
  tests/             unit tests + native-vs-Pyodide parity test
  results/           runs.csv, traces/, figures/
  web/
    src/             Vite + TS app (solver UI, replay, dashboard, about)
    src/worker/      Pyodide Web Worker: loads hpfold, runs jobs, posts frames
    public/data/     exported JSON
    public/pyodide/  self-hosted Pyodide (offline demo)
  paper/             IEEE LaTeX source; figures come from results/
  Makefile, pyproject.toml, uv.lock
```

## 10. Phases (no fixed dates)
| Phase | Deliverables | Exit criteria |
|---|---|---|
| 1. Core | Repo setup; HP model 2D+3D + tests; verified benchmark file; fold plot | E = −3 check passes; brute-force energy tests pass; benchmarks cited |
| 2. Algorithms | GA, ACO, PSO with the shared interface, progress frames, cancel; penalty and repair | All three reach a reasonable energy on a 20-mer in 2D and 3D; seeds reproduce |
| 3. Experiments | Penalty-vs-repair; tuning; full 2D runs; 3D runs | `runs.csv` complete; tuned defaults exported |
| 4. Analysis | Stats, tables, figures, sensitivity | All paper figures and tables regenerated by `make figures` |
| 5. Webpage | Pyodide spike (timing + parity) → live solver MVP (input, run, live fold + convergence, 2D/3D results, metrics) → internals heatmap, race, comparison → paper-results tab → share/export → deploy | Acceptance criteria in §14 met; offline demo verified |
| 6. Paper & talk | IEEE paper; 15-min slides with live demo | Submitted; demo rehearsed offline |

Phases 4–6 can overlap. The Pyodide timing spike should run as soon as one algorithm works (Phase 2), since its result sets the webpage's effort presets and length cap.

## 11. Paper and talk

### Paper (IEEE conference format, no page limit; aim for a complete but focused ~8–10 pages)
Abstract · Introduction (HP model, NP-hardness, motivation) · Related work (GA: Unger & Moult; ACO: Shmygelska & Hoos; PSO variants; pull moves) · Methods (model, encoding, constraint handling, the three algorithms, fairness protocol) · Experimental setup (benchmarks, budgets, tuning, seeds, hardware) · Results (tables, convergence, stats, 3D, sensitivity, penalty vs repair) · Discussion (why ACO's constructive approach helps, PSO adaptation trade-offs, limitations) · Conclusion and future work · Software availability (link to webpage and code) · References.

### Talk (~15 minutes)
| Minutes | Content |
|---|---|
| 0–2 | Problem: HP model, why folding is hard (NP-hard) |
| 2–6 | Methods: encoding; GA, ACO and PSO in one slide each; fairness setup |
| 6–10 | Results: convergence, success rates, statistics, 3D |
| 10–13 | **Live demo**: type a sequence and race GA/ACO/PSO on the webpage (pre-recorded video as a fallback) |
| 13–15 | Limitations, conclusion, questions |

Prepare an answer for "what was it trained on?" (see the terminology note in §1).

## 12. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Benchmark values mis-transcribed | Copy from primary sources; cite each entry; cross-check against ≥ 2 papers |
| Pure ACO without local search underperforms published ACO | Say so explicitly; the hybrid is the "Could" follow-up |
| PSO adaptation performs poorly | A valid finding if reported honestly; the sensitivity study explains why |
| Experiment runtime too long | Profile early; length-scaled budgets; multiprocessing |
| Pyodide too slow for long sequences | Early timing spike; calibrate effort presets; cap length; optimize the hot path (evaluate) in numpy |
| Native and browser results diverge | Parity test in CI; core package stays pure Python + numpy |
| Live demo fails during the talk | Self-hosted Pyodide (offline mode); pre-warmed browser tab; recorded fallback video |
| Webpage work delays the paper | Webpage MVP first (FR-L1 to L5); polish items (share URLs, race, dashboard) later |
| Tuning bias favors one algorithm | Equal tuning effort per algorithm |

## 13. Decisions log
| Question | Decision |
|---|---|
| Language | Python (instructor approved) |
| AI assistance | Allowed |
| Due dates | None fixed; work until done |
| Talk length | ~15 minutes |
| Paper length | IEEE format, no page limit |
| Webpage hosting | Public (static) |
| Webpage type | Live in-browser solver (Pyodide) + paper-results replay; no backend |
| 3D | Extension on a subset; also available in the live solver |
| Invalid folds | Penalty vs. repair decided by the mini-experiment |
| Hybrid | Optional stretch goal |
| PSO | Probability-based discrete PSO |
| Compute | Local CPU (multiprocessing); GPU optional |

## 14. Acceptance criteria (definition of done)
- [ ] All unit tests pass, including the E = −3 sanity case, brute-force cross-checks, and the native-vs-Pyodide parity test.
- [ ] The benchmark file has a citation for every sequence and energy.
- [ ] 30 runs × 3 algorithms × every 2D benchmark completed, with `runs.csv` committed.
- [ ] 3D extension results for at least 3 sequences.
- [ ] Stats table with Holm-corrected Wilcoxon p-values and A₁₂.
- [ ] Every figure and table in the paper is regenerated by `make figures` from the CSV.
- [ ] Webpage: a user can enter any valid sequence (3–100), run GA/ACO/PSO in 2D and/or 3D, watch live progress, cancel, and see the 2D/3D folds with all FR-L5 metrics.
- [ ] Webpage: the paper-results tab replays real traces and shows the summary dashboard.
- [ ] Webpage is deployed publicly and also works offline from localhost.
- [ ] Paper complete in IEEE format; 15-minute talk delivered with the live demo.

## 15. References (to verify and complete)
- Dill, K. A. (1985). Theory for the folding and stability of globular proteins. *Biochemistry*.
- Unger, R., & Moult, J. (1993). Genetic algorithms for protein folding simulations. *J. Mol. Biol.*
- Shmygelska, A., & Hoos, H. H. (2005). An ant colony optimisation algorithm for the 2D and 3D hydrophobic polar protein folding problem. *BMC Bioinformatics*.
- Lesh, N., Mitzenmacher, M., & Whitesides, S. (2003). A complete and effective move set for simplified protein folding. *RECOMB*.
- Crescenzi, P., et al. (1998); Berger, B., & Leighton, T. (1998). NP-completeness of HP folding.
- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization; Dorigo, M., & Stützle, T. (2004). *Ant Colony Optimization*.
- Pyodide project (Python in the browser via WebAssembly), for the software-availability section.
