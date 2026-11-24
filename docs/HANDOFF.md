# Handoff: HP Protein Folding with Nature-Inspired Algorithms

## Context
I'm a graduate student in **EECS 6180 – Biologically Inspired Computing** (University of Toledo, Dr. Devinder Kaur). The course requires an **individual project** that applies nature-inspired algorithms to an engineering problem, is implemented in software, and is written up as a **technical research paper** (15% of the grade), plus an oral presentation.

- Course topics: GA (binary and real-coded), genetic programming, ANN and neuroevolution, PSO, ant colony optimization (ACO), Grey Wolf Optimizer (GWO), Cuckoo Search, DNA computing, artificial immune systems and clonal selection, evolutionary fuzzy systems, hybrid models.
- Course tooling: MATLAB (Global Optimization and Fuzzy Logic toolboxes). **Language not decided yet.** Python is preferred for visualization/animation if the instructor allows it. Ask me before starting.
- AI tools may be used for learning, not for graded assignments. I'll confirm with the instructor what's allowed for the project. Help me understand and build, and explain the code.

## Chosen project
**Solving the 2D HP lattice protein folding problem with nature-inspired algorithms, plus a comparative study.** 3D (cubic lattice) is an optional extension.

### Problem definition
- **Input:** a string over {H, P}, length n (H = hydrophobic, P = polar). Use the standard benchmark sequences, which are already in H/P form.
- **Output:** a fold, meaning a grid coordinate for each bead, such that:
  1. **Connected:** consecutive beads are on adjacent lattice sites.
  2. **Self-avoiding:** no two beads share a site.
- **Objective:** minimize E = −(number of topological H–H contacts). A topological contact is a pair of H beads on adjacent sites that are *not* consecutive in the chain.
- The problem is NP-hard, which is the justification for using metaheuristics.

### Representation
- **Relative move encoding:** n−1 moves (the first move is fixed), each in {F, L, R} for 2D, or {F, L, R, U, D} for 3D.
- Invalid (self-colliding) folds are handled by a **penalty** or by **repair**. Comparing the two is an optional mini-experiment.

### Algorithms to compare (pick 3; a hybrid is optional)
1. **GA:** baseline, with crossover and mutation on move strings.
2. **ACO:** ants build folds bead by bead; pheromones on moves, with a heuristic based on new H–H contacts. Strong in the literature (Shmygelska & Hoos, 2005).
3. **Discrete PSO or GWO:** shows a continuous algorithm adapted to a discrete space.
4. *(Optional)* **Hybrid:** GA or ACO plus local search (pull moves). Ties to the course's "Hybrid Models" week.

### Benchmarks and evaluation
- Standard 2D HP benchmark sequences (lengths roughly 20 to 100) with published best-known energies. **Look up and verify the sequences and best-known energies from the literature (e.g., Unger & Moult 1993; Shmygelska & Hoos 2005) before using them. Don't rely on memory.**
- Use 30 independent runs per algorithm per sequence. Report best, mean and std of energy, success rate (hitting the best-known energy), runtime, and evaluations to best.
- Include convergence plots, a statistical test (Wilcoxon rank-sum), and a parameter-sensitivity study (population size, mutation rate, pheromone evaporation).
- Visualize the best folds as static plots, plus an animation for the presentation.

### Honest limitations (for the paper)
The HP model is a simplification: 2 residue types instead of 20, on a lattice instead of continuous 3D. It does not predict real protein structures, but it's a standard benchmark in the research literature. Very long sequences may not reach the best-known energies; report that honestly.

## Suggested structure
```
hp-folding/
  src/
    hp_model.*        # lattice, move decoding, validity check, energy
    algorithms/       # ga, aco, pso_or_gwo, (hybrid)
    experiments.*     # batch runs, seeds, results to CSV
    visualize.*       # fold plots, convergence plots, animation
  data/benchmarks.*   # verified sequences + best-known energies + sources
  results/
  paper/
```

## Suggested plan (about 3 weeks)
1. **Week 1:** HP model core (decode, validity, energy) with unit tests (hand-checked small folds), fold visualization, and a basic GA working on short sequences.
2. **Week 2:** ACO and a third algorithm, then the full benchmark runs.
3. **Week 3:** statistics, figures, and the paper in IEEE-style format (abstract, intro, related work, method, experiments, results, discussion, conclusion).

## First steps for Claude Code
1. Ask me: MATLAB or Python? 2D only, or 3D too?
2. Set up the repo structure.
3. Implement and unit-test the HP model core. Sanity check: the 10-bead sequence `HPHHHHPHHP` with relative moves corresponding to the fold (0,0)→(1,0)→(2,0)→(2,1)→(1,1)→(0,1)→(0,2)→(1,2)→(2,2)→(3,2) should give **E = −3**.
4. Build a simple GA baseline and a fold plot.
