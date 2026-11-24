"""Fitness evaluation with constraint handling (PRD FR-5, §5.2).

Every call to `evaluate` is one *fitness evaluation*: the unit in which all
algorithms' budgets are counted.

Strategies for self-colliding folds:

* ``"penalty"``: fitness = E + penalty * collisions. The fold is kept as is.
* ``"repair"``: the fold is first repaired into a valid one (`hpfold.folds.repair`);
  the repaired moves are returned so the caller can write them back into its
  population (Lamarckian repair). Fitness = E of the repaired fold.

`dim` is required: it fixes which move alphabet is legal (and which alphabet
repair may use), so a 3D genome that happens to contain no U/D moves is still
treated as 3D.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hpfold.folds import repair
from hpfold.hp_model import HP_ALPHABET, check_moves, collisions, energy, fold_coords

DEFAULT_PENALTY = 2.0  # provisional; tuned by the penalty-vs-repair experiment (T3.2)
STRATEGIES = ("penalty", "repair")


@dataclass(frozen=True)
class Evaluation:
    moves: str  # the evaluated fold (after repair, if the repair strategy was used)
    energy: int  # HP energy; meaningful as a result only when valid
    collisions: int
    fitness: float  # value to minimise

    @property
    def valid(self) -> bool:
        return self.collisions == 0


def evaluate(
    sequence: str,
    moves: str,
    dim: int,
    strategy: str = "penalty",
    penalty: float = DEFAULT_PENALTY,
    rng: np.random.Generator | None = None,
) -> Evaluation:
    """Score a fold of `sequence` (uppercase H/P, e.g. from `parse_sequence`) given by `moves`."""
    _check_arguments(sequence, strategy, penalty)
    check_moves(moves, dim)
    if strategy == "repair":
        if rng is None:
            raise ValueError("the repair strategy needs an rng (numpy Generator)")
        moves = repair(moves, rng, dim)

    coords = fold_coords(sequence, moves)
    e = energy(sequence, coords)
    k = collisions(coords)
    return Evaluation(moves=moves, energy=e, collisions=k, fitness=float(e + penalty * k))


def _check_arguments(sequence: str, strategy: str, penalty: float) -> None:
    if not sequence or not set(sequence) <= HP_ALPHABET:
        raise ValueError("sequence must be a non-empty uppercase H/P string (use parse_sequence)")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r} (expected one of {STRATEGIES})")
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError(f"penalty must be a finite number >= 0, got {penalty}")
