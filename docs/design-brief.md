# ProteinFold webpage: design brief

A brief for designing the ProteinFold webpage. The page will be implemented in an
existing codebase, so this brief also lists the technical limits a design has to
respect.

## 1. What the project is

ProteinFold is a graduate course project (EECS 6180, University of Toledo). It compares
three nature-inspired search algorithms on the **HP lattice protein-folding problem**:

- **GA**: genetic algorithm
- **ACO**: ant colony optimization
- **PSO**: particle swarm optimization (probability-based)

**The problem.**
- A protein is a string of **H** (hydrophobic) and **P** (polar) residues, for example
  `HPHPPHHPHPPHPHHPPHPH`.
- A *fold* places the chain on a grid: a square grid in 2D, a cube grid in 3D. Each bead
  sits next to the previous one, and no two beads share a cell.
- The **energy** is −1 for every pair of H beads that touch on the grid without being
  neighbours in the chain. Lower is better.
- Finding the best fold is NP-hard. The algorithms *search* for good folds.

**Important wording.** This is **search, not training**: nothing is learned in advance,
and no model is stored. Every run starts from scratch on the sequence you type. The
design should never suggest "AI model", "trained" or "prediction".

**Who sees it.**
- The page is public.
- It will be shown **live during a 15-minute talk** on a lecture-hall projector.
- Visitors include classmates, the instructor and anyone given the link.

## 2. Visual direction (decided)

The visual direction is **clean academic**:
- Light background, white cards with thin rules, and generous whitespace.
- Serif headings with a clean sans-serif body. Numbers set in tabular figures.
- A restrained palette:
  - **H residues** in one strong colour.
  - **P residues** in neutral grey.
  - **Three distinct series colours for GA / ACO / PSO**. These must be
    colour-blind-safe, distinguishable on a washed-out projector, and used consistently
    on every chart.
- It should feel like a polished companion to a research paper, not a startup landing page.
- There is no dark mode.

## 3. Pages and sections

The site has three tabs: **Live solver** (the main one), **Paper results** and **About**.

### 3.1 Live solver: enter a sequence, choose options, Run, watch it live, see results

**Input**
- **Sequence box.**
  - Accepts H/P, ignoring case and spaces, with a length of 3–100.
  - Shows an inline error for invalid input, for example "The sequence may contain only H
    and P." or "The sequence may have at most 100 residues in the browser (longer runs
    would take too long)."
- **Preset dropdown.** Lists benchmark sequences, each with its ID, length, best-known
  energy and source, for example "S1-1 · n = 20 · best known −9 · Thachuk et al. 2007".
- **Random button**, with a length field.

**Options**
- Lattice: 2D, 3D or both.
- Algorithms: any of GA, ACO and PSO, with all three selected by default.
- **Effort:** Quick, Standard or Thorough, with an estimated time shown.
- **Seed:** shown and editable, with a re-roll button.
- **Advanced** (collapsed): per-algorithm parameters, pre-filled with the tuned values
  from the paper.

**Actions**
- **Run**, which becomes **Cancel** while running.
- **Share link** and **Export** (PNG/SVG of a fold, JSON of the results).

**Loading state (first visit only).** The Python runtime downloads about 15 MB, then the
browser caches it. Show progress through these stages: "Loading the Python runtime",
"Loading numpy", "Loading the solver". Run stays disabled until loading finishes, which
takes about 5–10 seconds.

**Running: the "race".** There is one panel per algorithm × lattice, side by side; with 3
algorithms × 2 lattices that is up to 6 panels. Each panel shows:
- the **best fold so far**, redrawn whenever it improves
- a **convergence curve**: best energy so far against evaluations used, as a step line
  that grows live
- an **internals heatmap**:
  - It shows ACO's pheromone, PSO's probabilities or the GA population's move frequencies.
  - It's a grid of chain positions (rows, up to 98) by moves (columns): 3 columns in 2D
    (F/L/R) and 5 in 3D (F/L/R/U/D). Values run from 0 to 1.
- a **progress bar** (evaluations / budget), the elapsed time and the current best energy

Panels finish at different times. A finished panel should look finished while the others
keep going.

**Results (per panel).**
- **2D fold:**
  - Drawn on a faint square grid, with H and P beads and the chain as a line.
  - **Dashed lines** mark the H–H contacts.
  - Bead 0 is marked as the start.
- **3D fold:** a rotatable, zoomable ball-and-stick view (three.js) with a reset-view
  button and the same H/P colours.
- **Metrics:** energy E, number of H–H contacts, validity, evaluations used,
  evaluations to best, runtime, seed, and the move string.
- **Comparison table** across all panels, with the best energy highlighted.
- **Benchmark badge** when the sequence is a known benchmark: either "Reached best known
  −9" or "Gap 2 from best known −9", plus a citation.
- **Honesty note, always visible near the results:** "A single run is one random sample.
  The paper's conclusions rest on 30 runs per algorithm."

**States to design:**
- empty
- invalid input
- loading the runtime
- ready
- running
- cancelled (the best fold so far is kept and clearly labelled "cancelled")
- finished
- **no valid fold found** (shown as "NA", never as 0)
- a job error (plain text)
- the runtime failed to load

### 3.2 Paper results

- **Replay:** choose a sequence and an algorithm, then press play to watch a real recorded
  experiment run unfold: fold, convergence and heatmap, with play/pause and a scrubber.
  These are the best and median runs of 30.
- **Dashboard:**
  - per-sequence tables: best / mean ± sd / median / worst energy, success rate and evaluations to best
  - **box plots** of final energies per algorithm
  - **convergence plots**: median with an interquartile band
  - a statistics table with p-values (Holm-corrected) and effect sizes (A₁₂)
- Every number comes from the real experiment files; nothing is typed in by hand.

### 3.3 About

Short explainers, each about one screen, with small diagrams:
- the HP model
- how each algorithm searches
- the fairness rules: same evaluation budget, 30 seeds
- "search, not training"
- the differences from the original ACO paper
- links to the paper and the code

## 4. Real example values for mock-ups

Use these real values rather than invented numbers.

| Sequence | Lattice | Algorithm | Energy reached | Best known | Moves |
|---|---|---|---|---|---|
| S1-1 `HPHPPHHPHPPHPHHPPHPH` (n = 20) | 2D | GA | −9 | −9 | `LFLLRRLFLLRLRRLLFL` |
| S1-1 (same) | 3D | ACO | −11 | — | `RDRRFFRDDRRDFRDDFD` |
| S1-2 `HHPPHPPHPPHPPHPPHPPHPPHH` (n = 24) | 2D | PSO | −7 | −9 | `FLLFRFRLLRLLRLLRFFFLFF` |

**Other benchmark presets:**
- S1-3 `PPHPPHHPPPPHHPPPPHHPPPPHH`, n = 25, best known −8
- S2-1, a 48-residue 3D instance, best known −32

There are 11 2D and 10 3D instances in all, with sources from Thachuk, Shmygelska &
Hoos (2007) and Shmygelska & Hoos (2005).

**Effort presets** set the evaluation budget to max(5000, c·n²):
- Quick: c = 2
- Standard: c = 5
- Thorough: c = 12

**Timing:** a 20-residue Standard run takes a few seconds, and a 100-residue Thorough
ACO run takes up to about 2 minutes.

## 5. Technical limits the design must respect

- **Static site, no backend.** Python runs in the browser through Pyodide in Web
  Workers.
- **Works offline after the first visit.** Fonts and assets must be self-hostable, with
  no runtime CDN calls; open-license fonts are fine.
- **Built with plain TypeScript, SVG and three.js**, without a framework.
  - Prefer designs expressed as HTML and CSS with design tokens (colours, type scale,
    spacing, radii).
  - Anything that needs a heavy component library will be rebuilt by hand.
- **Responsive.**
  - The main target is a projector or laptop at 1280–1920 px wide.
  - It must still work on a 390 px phone, where the race panels stack.
- **Accessible.**
  - WCAG AA contrast.
  - Colour is never the only signal: H and P also differ in shape or label.
  - Keyboard-operable controls.
  - `prefers-reduced-motion` respected for the live animations.
- **Readable from the back of a lecture hall.**
  - Minimum 16 px body text.
  - Large energy numbers on the panels.
  - Thick enough lines in charts.

## 6. What to deliver back

- Design tokens: colours with hex values and their roles, typography, spacing and radii.
- Layouts for the Live solver (input, running race, results), Paper results and About,
  at about 1440 px and 390 px wide.
- Component designs:
  - sequence input with error
  - option controls
  - race panel
  - 2D fold
  - convergence chart
  - heatmap
  - metrics card
  - comparison table
  - benchmark badge
  - loading indicator
  - honesty note
- One mock-up for each state listed in 3.1.
