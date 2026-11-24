"""Genetic algorithm baseline (PRD §5.3 GA).

Chromosome : relative-move string of length n-2.
Init       : random self-avoiding folds, so generation 0 is fully valid.
Selection  : tournament of size k (lowest fitness wins).
Crossover  : one-point, two-point or uniform, applied with probability pc.
Mutation   : each gene is re-drawn to a *different* move with probability pm
             (default 2 / L, i.e. two mutations per child on average).
Elitism    : the `elite` best individuals are copied unchanged.
Constraint : penalty (fitness = E + lambda * collisions) or Lamarckian repair.
Budget     : one fitness evaluation per individual evaluated (see base.Evaluator).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from hpfold.algorithms.base import (
    ProgressFn,
    RunContext,
    RunResult,
    StopFn,
    frequency_matrix,
)
from hpfold.fitness import DEFAULT_PENALTY, STRATEGIES
from hpfold.folds import random_valid_fold

CROSSOVERS = ("one_point", "two_point", "uniform")
DEFAULT_BUDGET = 100_000
MUTATIONS_PER_CHILD = 2.0  # default pm = this / chromosome length


@dataclass(frozen=True)
class Params:
    pop_size: int = 100
    crossover: str = "two_point"
    pc: float = 0.9
    pm: float | None = None
    tournament_k: int = 3
    elite: int = 2
    strategy: str = "penalty"
    penalty: float = DEFAULT_PENALTY

    def __post_init__(self) -> None:
        if self.pop_size < 2:
            raise ValueError("pop_size must be >= 2")
        if self.crossover not in CROSSOVERS:
            raise ValueError(f"crossover must be one of {CROSSOVERS}")
        if not 0.0 <= self.pc <= 1.0:
            raise ValueError("pc must be in [0, 1]")
        if self.pm is not None and not 0.0 < self.pm <= 1.0:
            raise ValueError("pm must be in (0, 1] or None")
        if not 1 <= self.tournament_k <= self.pop_size:
            raise ValueError("tournament_k must be in [1, pop_size]")
        if not 0 <= self.elite < self.pop_size:
            raise ValueError("elite must be in [0, pop_size)")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if self.penalty < 0:
            raise ValueError("penalty must be >= 0")


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
    ctx = RunContext("GA", sequence, dim, asdict(p), seed, budget, target,
                     on_progress, should_stop, p.strategy, p.penalty)
    if ctx.trivial:
        return ctx.run_trivial()
    pm = p.pm if p.pm is not None else min(1.0, MUTATIONS_PER_CHILD / (ctx.n - 2))
    ctx.stats["pm"] = pm
    if ctx.stop_requested():  # cancel is checked before every generation, including the first
        return ctx.result()

    pop: list[str] = []
    fit: list[float] = []
    valid: list[bool] = []
    while len(pop) < p.pop_size and not ctx.done:
        ev = ctx.evaluate(random_valid_fold(ctx.n, ctx.rng, dim))
        pop.append(ev.moves)
        fit.append(ev.fitness)
        valid.append(ev.valid)
    if pop:
        _record(ctx, pop, fit, valid)

    while not ctx.done and not ctx.stop_requested():
        pop, fit, valid = _next_generation(ctx, p, pop, np.asarray(fit), valid, pm)
        _record(ctx, pop, fit, valid)
    return ctx.result()


def _next_generation(ctx: RunContext, p: Params, pop: list[str], fit: np.ndarray, valid: list[bool], pm: float):
    order = np.argsort(fit, kind="stable")
    children = [pop[i] for i in order[: p.elite]]
    child_fit = [float(fit[i]) for i in order[: p.elite]]
    child_valid = [valid[i] for i in order[: p.elite]]
    while len(children) < p.pop_size and not ctx.done:
        a = pop[tournament(fit, p.tournament_k, ctx.rng)]
        b = pop[tournament(fit, p.tournament_k, ctx.rng)]
        pair = crossover(a, b, p.crossover, ctx.rng) if ctx.rng.random() < p.pc else (a, b)
        for child in pair:
            if len(children) >= p.pop_size or ctx.done:
                break
            ev = ctx.evaluate(mutate(child, pm, ctx.alphabet, ctx.rng))
            children.append(ev.moves)  # repaired moves are written back (Lamarckian)
            child_fit.append(ev.fitness)
            child_valid.append(ev.valid)
    return children, child_fit, child_valid


def _record(ctx: RunContext, pop: list[str], fit, valid: list[bool]) -> None:
    """Emit a frame; its current-best fold is the fittest VALID one when any is valid.

    (With the penalty strategy a colliding fold can have the lowest fitness, but
    frames are drawn by the web page, which expects self-avoiding folds.)
    """
    masked = np.where(np.asarray(valid), np.asarray(fit, dtype=float), np.inf)
    index = int(np.argmin(masked)) if any(valid) else int(np.argmin(fit))
    ctx.record(pop[index], frequency_matrix(pop, ctx.alphabet))


def tournament(fitness: np.ndarray, k: int, rng: np.random.Generator, replace: bool = True) -> int:
    """Index of the lowest-fitness individual among k drawn at random."""
    contestants = rng.choice(len(fitness), size=k, replace=replace)
    return int(contestants[np.argmin(fitness[contestants])])


def crossover(a: str, b: str, kind: str, rng: np.random.Generator) -> tuple[str, str]:
    """Recombine two parents; every child gene comes from one parent at the same position."""
    length = len(a)
    if length < 2:
        return a, b
    if kind == "one_point":
        cut = int(rng.integers(1, length))
        return a[:cut] + b[cut:], b[:cut] + a[cut:]
    if kind == "two_point":
        i, j = sorted(int(x) for x in rng.choice(np.arange(1, length + 1), size=2, replace=False))
        return a[:i] + b[i:j] + a[j:], b[:i] + a[i:j] + b[j:]
    if kind == "uniform":
        mask = rng.random(length) < 0.5
        c1 = "".join(x if m else y for x, y, m in zip(a, b, mask))
        c2 = "".join(y if m else x for x, y, m in zip(a, b, mask))
        return c1, c2
    raise ValueError(f"unknown crossover {kind!r}")


def mutate(moves: str, pm: float, alphabet: str, rng: np.random.Generator) -> str:
    """Re-draw each gene with probability pm, always to a different move."""
    mask = rng.random(len(moves)) < pm
    if not mask.any():
        return moves
    out = list(moves)
    for i in np.flatnonzero(mask):
        choices = [m for m in alphabet if m != out[i]]
        out[i] = choices[int(rng.integers(len(choices)))]
    return "".join(out)
