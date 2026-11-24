"""Probability-based discrete Particle Swarm Optimization (PRD §5.3 PSO).

Continuous PSO moves a particle's position x with a velocity v pulled toward its
personal best and the swarm's best. Folds are discrete (move strings), so here a
particle's "position" is a probability matrix P (one distribution over moves per
position) and the velocity is the drift of that probability toward the bests:

    P <- w * P + c1 * r1 * onehot(pbest) + c2 * r2 * onehot(neighbourhood best)
    P <- row-normalize(P)
    P <- epsilon + (1 - A * epsilon) * P      (floor: no move ever reaches 0)

Each iteration a particle samples a concrete fold from P (row by row), which is
evaluated (one fitness evaluation) under the penalty or Lamarckian repair
strategy, exactly as in the GA. w is the inertia, c1/c2 the cognitive/social
weights, r1/r2 ~ U(0, 1). Neighbourhood: the whole swarm ('global') or the two
ring neighbours ('ring'). Bests are compared by fitness.
"""

from __future__ import annotations

import math
import numbers
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

TOPOLOGIES = ("global", "ring")
DEFAULT_BUDGET = 100_000
MAX_EPSILON = 0.2  # floor must stay below 1/|alphabet| for the 3D alphabet (5 moves)


@dataclass(frozen=True)
class Params:
    swarm_size: int = 30
    w: float = 0.9  # inertia: how much of the old probabilities is kept
    c1: float = 0.5  # cognitive pull toward the particle's own best
    c2: float = 0.5  # social pull toward the neighbourhood best
    epsilon: float = 0.03  # probability floor per move (keeps exploring)
    topology: str = "global"
    strategy: str = "penalty"
    penalty: float = DEFAULT_PENALTY

    def __post_init__(self) -> None:
        if isinstance(self.swarm_size, bool) or not isinstance(self.swarm_size, numbers.Integral):
            raise ValueError("swarm_size must be an integer")  # noqa: TRY004 - contract: invalid params -> ValueError
        for name in ("w", "c1", "c2", "epsilon", "penalty"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.swarm_size < 1:
            raise ValueError("swarm_size must be >= 1")
        if min(self.w, self.c1, self.c2) < 0:
            raise ValueError("w, c1 and c2 must be >= 0")
        if not 0.0 < self.epsilon < MAX_EPSILON:
            raise ValueError(f"epsilon must be in (0, {MAX_EPSILON})")
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"topology must be one of {TOPOLOGIES}")
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
    ctx = RunContext("PSO", sequence, dim, asdict(p), seed, budget, target,
                     on_progress, should_stop, p.strategy, p.penalty)
    if ctx.trivial:
        return ctx.run_trivial()
    if ctx.stop_requested():
        return ctx.result()

    uniform = np.full((ctx.n - 2, len(ctx.alphabet)), 1.0 / len(ctx.alphabet))
    probs: list[np.ndarray] = []
    positions: list[str] = []
    fit: list[float] = []
    valid: list[bool] = []
    while len(positions) < p.swarm_size and not ctx.done:  # start from random valid folds
        ev = ctx.evaluate(random_valid_fold(ctx.n, ctx.rng, dim))
        probs.append(uniform)
        positions.append(ev.moves)
        fit.append(ev.fitness)
        valid.append(ev.valid)
    if not positions:
        return ctx.result()
    pbest, pbest_fit = list(positions), list(fit)
    _record(ctx, probs, positions, fit, valid, pbest_fit)

    while not ctx.done and not ctx.stop_requested():
        guides = [pbest[i] for i in neighbourhood_best(pbest_fit, p.topology)]
        for k in range(len(probs)):
            if ctx.done:
                break
            r1, r2 = ctx.rng.random(2)
            probs[k] = update_probabilities(probs[k], pbest[k], guides[k], ctx.alphabet,
                                            p.w, p.c1, p.c2, r1, r2, p.epsilon)
            ev = ctx.evaluate(sample(probs[k], ctx.alphabet, ctx.rng))
            positions[k], fit[k], valid[k] = ev.moves, ev.fitness, ev.valid  # repaired moves written back
            if ev.fitness < pbest_fit[k]:
                pbest[k], pbest_fit[k] = ev.moves, ev.fitness
        _record(ctx, probs, positions, fit, valid, pbest_fit)
    return ctx.result()


def neighbourhood_best(pbest_fit: list[float], topology: str) -> list[int]:
    """Index of the best personal best each particle is pulled toward."""
    size = len(pbest_fit)
    if topology == "global":
        best = int(np.argmin(pbest_fit))
        return [best] * size
    return [min(((k - 1) % size, k, (k + 1) % size), key=lambda i: pbest_fit[i]) for k in range(size)]


def update_probabilities(
    probs: np.ndarray, pbest: str, nbest: str, alphabet: str,
    w: float, c1: float, c2: float, r1: float, r2: float, epsilon: float,
) -> np.ndarray:
    """One velocity step on a probability matrix; returns a NEW matrix (input untouched)."""
    index = {m: k for k, m in enumerate(alphabet)}
    rows = np.arange(len(pbest))
    raw = w * probs
    raw[rows, [index[m] for m in pbest]] += c1 * r1
    raw[rows, [index[m] for m in nbest]] += c2 * r2
    totals = raw.sum(axis=1, keepdims=True)
    width = probs.shape[1]
    normalized = np.where(totals > 0, raw / np.where(totals > 0, totals, 1.0), 1.0 / width)
    return epsilon + (1.0 - width * epsilon) * normalized


def sample(probs: np.ndarray, alphabet: str, rng: np.random.Generator) -> str:
    """Draw one move per row from the row's distribution."""
    cumulative = np.cumsum(probs, axis=1)
    draws = rng.random((probs.shape[0], 1))
    picks = np.minimum((draws > cumulative).sum(axis=1), probs.shape[1] - 1)
    return "".join(alphabet[i] for i in picks)


def _record(ctx: RunContext, probs, positions, fit, valid, pbest_fit) -> None:
    """Frame: fittest valid position; heatmap = probabilities of the particle owning the
    swarm best; diversity = realized move frequencies of the swarm (comparable with GA/ACO)."""
    masked = np.where(np.asarray(valid), np.asarray(fit, dtype=float), np.inf)
    index = int(np.argmin(masked)) if any(valid) else int(np.argmin(fit))
    owner = int(np.argmin(pbest_fit))
    ctx.record(positions[index], probs[owner],
               diversity_matrix=frequency_matrix(positions, ctx.alphabet))
