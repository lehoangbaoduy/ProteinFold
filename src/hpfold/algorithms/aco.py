"""Ant Colony Optimization for HP folding (PRD §5.3 ACO; after Shmygelska & Hoos 2005).

Construction
    Each ant grows a self-avoiding fold bead by bead, starting from bead 0.
    At move position i it chooses among the moves that lead to a FREE site,
    with probability proportional to  tau[i, m]^alpha * eta^beta,  where
    eta = exp(new H-H contacts made by the new bead / gamma) is the greedy
    heuristic. A dead end (no free site) makes the ant back up one step and
    exclude the move that led there; more than `backtrack_limit` backtracks
    restarts the ant; after `max_restarts` restarts it falls back to a random
    valid fold. Restarts and fallbacks are reported in RunResult.stats.

Pheromone (MAX-MIN Ant System style)
    tau <- (1 - rho) * tau, then the iteration-best and/or global-best ants
    deposit their relative quality on the moves they used; tau is clamped to
    [tau_min, tau_max] so no move ever becomes impossible (avoids stagnation).

Budget
    One fitness evaluation per completed ant; construction steps are free.
    Ants always build valid folds, so no penalty or repair is needed.

Differences from Shmygelska & Hoos (2005), to be stated in the paper: ants
grow in one direction only (from bead 0, not from a random bead in both
directions) and there is no local-search phase.
"""

from __future__ import annotations

import bisect
import itertools
import math
import numbers
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from hpfold.algorithms.base import ProgressFn, RunContext, RunResult, StopFn, frequency_matrix
from hpfold.folds import random_valid_fold
from hpfold.hp_model import (
    INITIAL_HEADING,
    INITIAL_UP,
    MOVES_2D,
    NEIGHBOR_OFFSETS,
    ORIGIN,
    Coord,
    add,
    step_frame,
)

DEPOSITS = ("iteration_best", "global_best", "both")
DEFAULT_BUDGET = 100_000
GLOBAL_BEST_QUALITY = 1.0
MAX_NEW_CONTACTS = 5  # a new bead has at most 5 non-bonded neighbours (cubic lattice)
MAX_LOG_BIAS = 50.0  # cap on beta * contacts / gamma, so eta^beta never overflows


@dataclass(frozen=True)
class Params:
    n_ants: int = 30
    alpha: float = 1.0  # weight of pheromone
    beta: float = 2.0  # weight of the contact heuristic
    rho: float = 0.05  # evaporation rate (low: slow convergence, avoids early stagnation)
    gamma: float = 1.0  # heuristic temperature: eta = exp(contacts / gamma)
    tau0: float = 1.0
    tau_min: float = 0.1
    tau_max: float = 2.0  # narrow MMAS bounds keep every move possible
    deposit: str = "both"
    backtrack_limit: int = 50  # backtracks allowed per construction attempt
    max_restarts: int = 20  # attempts before falling back to a random valid fold

    def __post_init__(self) -> None:
        for name in ("n_ants", "backtrack_limit", "max_restarts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ValueError(f"{name} must be an integer")
        for name in ("alpha", "beta", "rho", "gamma", "tau0", "tau_min", "tau_max"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.n_ants < 1:
            raise ValueError("n_ants must be >= 1")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be >= 0")
        if not 0.0 < self.rho <= 1.0:
            raise ValueError("rho must be in (0, 1]")
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0")
        if not 0.0 < self.tau_min <= self.tau_max:
            raise ValueError("need 0 < tau_min <= tau_max")
        if not self.tau_min <= self.tau0 <= self.tau_max:
            raise ValueError("tau0 must lie in [tau_min, tau_max]")
        if self.deposit not in DEPOSITS:
            raise ValueError(f"deposit must be one of {DEPOSITS}")
        if self.backtrack_limit < 0 or self.max_restarts < 0:
            raise ValueError("backtrack_limit and max_restarts must be >= 0")


@dataclass(frozen=True)
class Ant:
    moves: str
    restarts: int  # construction attempts abandoned before this fold (max_restarts on fallback)
    fallback: bool  # True if the fold is a random valid fold after all attempts failed
    backtracks: int = 0  # backtracks made in the successful attempt


def run(
    sequence: str,
    dim: int,
    params: Params | None = None,
    seed: int = 0,
    budget: int = DEFAULT_BUDGET,
    target: int | None = None,
    on_progress: ProgressFn | None = None,
    should_stop: StopFn | None = None,
) -> RunResult:
    p = params or Params()
    ctx = RunContext("ACO", sequence, dim, asdict(p), seed, budget, target, on_progress, should_stop)
    if ctx.trivial:
        return ctx.run_trivial()

    tau = np.full((ctx.n - 2, len(ctx.alphabet)), p.tau0)
    global_best: tuple[str, int] | None = None
    restarts = fallbacks = 0
    while not ctx.done and not ctx.stop_requested():
        colony: list[tuple[str, int]] = []
        bias = heuristic_table(p)
        for _ in range(p.n_ants):
            if ctx.done:
                break
            ant = construct_ant(sequence, tau, p, ctx.alphabet, ctx.rng, bias)
            restarts += ant.restarts
            fallbacks += int(ant.fallback)
            ev = ctx.evaluate(ant.moves)
            colony.append((ev.moves, ev.energy))
        iteration_best = min(colony, key=lambda a: a[1])
        if global_best is None or iteration_best[1] < global_best[1]:
            global_best = iteration_best
        tau = update_pheromone(tau, _deposits(p, iteration_best, global_best), p, ctx.alphabet)
        # Heatmap = pheromone (the 'internals' view); diversity = the colony's realized moves.
        ctx.record(iteration_best[0], choice_matrix(tau, p.alpha),
                   diversity_matrix=frequency_matrix([m for m, _ in colony], ctx.alphabet))
    ctx.stats.update(restarts=restarts, fallbacks=fallbacks)
    return ctx.result()


def _deposits(p: Params, iteration_best: tuple[str, int], global_best: tuple[str, int]):
    """With deposit='both', a fold that is both iteration- and global-best deposits twice
    (standard elitist behaviour: the best fold is reinforced most)."""
    deposits = []
    if p.deposit in ("iteration_best", "both"):
        deposits.append((iteration_best[0], quality(iteration_best[1], global_best[1])))
    if p.deposit in ("global_best", "both"):
        deposits.append((global_best[0], GLOBAL_BEST_QUALITY))
    return deposits


def quality(energy: int, best_energy: int) -> float:
    """Relative solution quality in (0, 1]: 1 for the global best, smaller for worse folds."""
    return (1 - energy) / (1 - best_energy)


def update_pheromone(tau: np.ndarray, deposits, p: Params, alphabet: str) -> np.ndarray:
    """Evaporate, deposit and clamp; returns a NEW array (the input is not modified)."""
    index = {m: k for k, m in enumerate(alphabet)}
    new = (1.0 - p.rho) * tau
    for moves, amount in deposits:
        new[np.arange(len(moves)), [index[m] for m in moves]] += amount
    return np.clip(new, p.tau_min, p.tau_max)


def choice_matrix(tau: np.ndarray, alpha: float) -> np.ndarray:
    """Row-normalized tau^alpha: the pheromone part of each position's move choice."""
    weights = tau ** alpha
    return weights / weights.sum(axis=1, keepdims=True)


def new_contacts(sequence: str, k: int, site: Coord, placed: Mapping[Coord, int]) -> int:
    """H-H contacts bead k would make at `site` with already-placed, non-bonded beads."""
    if sequence[k] != "H":
        return 0
    count = 0
    for d in NEIGHBOR_OFFSETS:
        j = placed.get(add(site, d))
        if j is not None and j < k - 1 and sequence[j] == "H":
            count += 1
    return count


def heuristic_table(p: Params) -> tuple[float, ...]:
    """eta^beta for 0..MAX_NEW_CONTACTS new contacts, with the exponent capped (no overflow)."""
    return tuple(math.exp(min(p.beta * c / p.gamma, MAX_LOG_BIAS)) for c in range(MAX_NEW_CONTACTS + 1))


def construct_ant(
    sequence: str, tau: np.ndarray, p: Params, alphabet: str, rng: np.random.Generator,
    bias: tuple[float, ...] | None = None,
) -> Ant:
    """Build one valid fold (see module docstring); never returns an invalid fold."""
    weights = tau ** p.alpha
    bias = bias or heuristic_table(p)
    for attempt in range(p.max_restarts + 1):
        moves, backtracks = _attempt(sequence, weights, bias, p.backtrack_limit, alphabet, rng)
        if moves is not None:
            return Ant(moves, restarts=attempt, fallback=False, backtracks=backtracks)
    dim = 2 if alphabet == MOVES_2D else 3
    return Ant(random_valid_fold(len(sequence), rng, dim), restarts=p.max_restarts, fallback=True)


def _candidates(sequence, weights_row, bias, alphabet, frame, last, placed, excluded, k):
    out = []
    for idx, move in enumerate(alphabet):
        if move in excluded:
            continue
        heading, up = step_frame(*frame, move)
        site = add(last, heading)
        if site in placed:
            continue
        out.append((move, (heading, up), site, weights_row[idx] * bias[new_contacts(sequence, k, site, placed)]))
    return out


def _attempt(sequence, weights, bias, backtrack_limit: int, alphabet: str, rng) -> tuple[str | None, int]:
    target = len(sequence) - 2
    coords: list[Coord] = [ORIGIN, INITIAL_HEADING]
    placed: dict[Coord, int] = {ORIGIN: 0, INITIAL_HEADING: 1}
    frames = [(INITIAL_HEADING, INITIAL_UP)]
    chosen: list[str] = []
    excluded: list[set[str]] = [set()]
    backtracks = 0
    while len(chosen) < target:
        depth = len(chosen)
        options = _candidates(sequence, weights[depth], bias, alphabet, frames[-1], coords[-1],
                              placed, excluded[-1], depth + 2)
        if not options:
            backtracks += 1
            if not chosen or backtracks > backtrack_limit:
                return None, backtracks
            excluded.pop()
            failed = chosen.pop()
            frames.pop()
            del placed[coords.pop()]
            excluded[-1].add(failed)
            continue
        cumulative = list(itertools.accumulate(o[3] for o in options))
        pick = options[bisect.bisect_right(cumulative, rng.random() * cumulative[-1])]
        chosen.append(pick[0])
        frames.append(pick[1])
        coords.append(pick[2])
        placed[pick[2]] = len(coords) - 1
        excluded.append(set())
    return "".join(chosen), backtracks
