"""Shared interface for all search algorithms (PRD §5.3, FR-T1).

Every algorithm exposes the same entry point::

    run(sequence, dim, params=None, seed=0, budget=..., target=None,
        on_progress=None, should_stop=None) -> RunResult

Fairness rules enforced here, not in each algorithm:

* `Evaluator` counts every fitness evaluation and refuses to go past the
  budget, so all algorithms spend exactly the same number of evaluations
  (unless the target energy is reached or the run is cancelled).
* Only VALID folds can become the best result. A run that never evaluates a
  valid fold reports best_energy = None (NA), never 0.
* All randomness comes from one numpy Generator seeded with `seed`.

`RunContext` bundles that bookkeeping (evaluator, rng, cancel check, progress
frames, convergence history) so each algorithm only implements its search loop.
"""

from __future__ import annotations

import math
import numbers
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hpfold.fitness import DEFAULT_PENALTY, Evaluation, evaluate
from hpfold.hp_model import alphabet_for, parse_sequence

MATRIX_DECIMALS = 3  # rounding used only when frames are serialized (to_dict)
TRIVIAL_LENGTH = 3  # sequences shorter than this have exactly one fold ("")

ProgressFn = Callable[["Frame"], None]
StopFn = Callable[[], bool]


@dataclass(frozen=True)
class Frame:
    """Progress snapshot emitted once per iteration (PRD FR-T1)."""

    iter: int
    evals: int
    budget: int
    best_energy: int | None
    best_moves: str | None  # best valid fold so far (None = none yet)
    current_best_moves: str  # best fold of the current population (valid if any is)
    diversity: float  # 0 = population/probabilities collapsed, 1 = uniform
    matrix: tuple[tuple[float, ...], ...]  # positions x moves, rows sum to 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "iter": self.iter,
            "evals": self.evals,
            "budget": self.budget,
            "best_energy": self.best_energy,
            "best_moves": self.best_moves,
            "current_best_moves": self.current_best_moves,
            "diversity": round(self.diversity, MATRIX_DECIMALS),
            "matrix": [[round(v, MATRIX_DECIMALS) for v in row] for row in self.matrix],
        }


@dataclass(frozen=True)
class RunResult:
    algorithm: str
    sequence: str
    dim: int
    seed: int
    params: Mapping[str, Any]
    best_moves: str | None  # None = no valid fold found (NA)
    best_energy: int | None
    evaluations: int
    evals_to_best: int | None
    runtime_s: float
    history: tuple[tuple[int, int | None], ...]  # (evals so far, best energy so far)
    cancelled: bool
    target_reached: bool
    stats: Mapping[str, float] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.best_energy is not None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (plain ints/floats/lists; best_* are null when NA)."""
        return {
            "algorithm": self.algorithm,
            "sequence": self.sequence,
            "dim": self.dim,
            "seed": self.seed,
            "params": dict(self.params),
            "best_moves": self.best_moves,
            "best_energy": self.best_energy,
            "evaluations": self.evaluations,
            "evals_to_best": self.evals_to_best,
            "runtime_s": self.runtime_s,
            "history": [list(point) for point in self.history],
            "cancelled": self.cancelled,
            "target_reached": self.target_reached,
            "stats": dict(self.stats),
        }


class Evaluator:
    """Budget-enforcing fitness counter that remembers the best valid fold."""

    def __init__(
        self,
        sequence: str,
        dim: int,
        budget: int,
        strategy: str = "penalty",
        penalty: float = DEFAULT_PENALTY,
        rng: np.random.Generator | None = None,
        target: int | None = None,
    ) -> None:
        self.sequence = sequence
        self.dim = dim
        self.budget = budget
        self.strategy = strategy
        self.penalty = penalty
        self.rng = rng
        self.target = target
        self.evaluations = 0
        self.best_moves: str | None = None
        self.best_energy: int | None = None
        self.evals_to_best: int | None = None

    @property
    def target_reached(self) -> bool:
        return self.target is not None and self.best_energy is not None and self.best_energy <= self.target

    @property
    def done(self) -> bool:
        return self.evaluations >= self.budget or self.target_reached

    def evaluate(self, moves: str) -> Evaluation:
        if self.evaluations >= self.budget:
            raise RuntimeError("evaluation budget exhausted")
        ev = evaluate(self.sequence, moves, self.dim, self.strategy, self.penalty, self.rng)
        self.evaluations += 1
        if ev.valid and (self.best_energy is None or ev.energy < self.best_energy):
            self.best_moves, self.best_energy, self.evals_to_best = ev.moves, ev.energy, self.evaluations
        return ev


def _is_int(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def check_run_args(sequence: str, dim: int, budget: int, seed: int = 0, target: int | None = None) -> None:
    """Validate the shared run() arguments at the boundary (CONST-SEC-005)."""
    if parse_sequence(sequence) != sequence:
        raise ValueError("sequence must already be uppercase H/P without whitespace (use parse_sequence)")
    alphabet_for(dim)
    if not _is_int(budget) or budget < 0:
        raise ValueError(f"budget must be a non-negative integer, got {budget!r}")
    if not _is_int(seed) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    if target is not None and not _is_int(target):
        raise ValueError(f"target must be an integer energy or None, got {target!r}")


def normalized_entropy(matrix: np.ndarray) -> float:
    """Mean per-position entropy of a row-stochastic matrix, scaled to [0, 1]."""
    if matrix.size == 0 or matrix.shape[1] < 2:
        return 0.0
    p = np.clip(matrix, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
    return float(min(1.0, max(0.0, h.mean() / math.log(matrix.shape[1]))))


def frequency_matrix(population: Sequence[str], alphabet: str) -> np.ndarray:
    """Positions x moves matrix of how often each move appears in the population."""
    length = len(population[0]) if population else 0
    counts = np.zeros((length, len(alphabet)))
    index = {m: k for k, m in enumerate(alphabet)}
    for moves in population:
        for i, m in enumerate(moves):
            counts[i, index[m]] += 1
    return counts / max(len(population), 1)


class RunContext:
    """Bookkeeping shared by all algorithms for one run."""

    def __init__(
        self,
        algorithm: str,
        sequence: str,
        dim: int,
        params: Mapping[str, Any],
        seed: int,
        budget: int,
        target: int | None = None,
        on_progress: ProgressFn | None = None,
        should_stop: StopFn | None = None,
        strategy: str = "penalty",
        penalty: float = DEFAULT_PENALTY,
    ) -> None:
        check_run_args(sequence, dim, budget, seed, target)
        budget, seed = int(budget), int(seed)  # plain ints (numpy ints break JSON)
        target = None if target is None else int(target)
        self.algorithm = algorithm
        self.sequence = sequence
        self.dim = dim
        self.params = dict(params)
        self.seed = seed
        self.alphabet = alphabet_for(dim)
        self.rng = np.random.default_rng(seed)
        self.evaluator = Evaluator(sequence, dim, budget, strategy, penalty, self.rng, target)
        self._on_progress = on_progress
        self._should_stop = should_stop
        self._history: list[tuple[int, int | None]] = [(0, None)]
        self._iteration = 0
        self._start = time.perf_counter()
        self.cancelled = False
        self.stats: dict[str, float] = {}

    @property
    def n(self) -> int:
        return len(self.sequence)

    @property
    def trivial(self) -> bool:
        return self.n < TRIVIAL_LENGTH

    @property
    def done(self) -> bool:
        return self.evaluator.done

    def evaluate(self, moves: str) -> Evaluation:
        return self.evaluator.evaluate(moves)

    def stop_requested(self) -> bool:
        if self._should_stop is not None and self._should_stop():
            self.cancelled = True
        return self.cancelled

    def record(self, current_best_moves: str, matrix: np.ndarray,
               diversity_matrix: np.ndarray | None = None) -> None:
        """Close one iteration: append to the history and emit a progress frame.

        `matrix` is the algorithm's internals view (shown on the web page);
        diversity is computed from `diversity_matrix` when given (e.g. the realized
        move frequencies of a colony), so all algorithms report comparable diversity.
        """
        ev = self.evaluator
        self._history.append((ev.evaluations, ev.best_energy))
        self._iteration += 1
        if self._on_progress is None:
            return
        self._on_progress(Frame(
            iter=self._iteration,
            evals=ev.evaluations,
            budget=ev.budget,
            best_energy=ev.best_energy,
            best_moves=ev.best_moves,
            current_best_moves=current_best_moves,
            diversity=normalized_entropy(matrix if diversity_matrix is None else diversity_matrix),
            matrix=tuple(tuple(float(v) for v in row) for row in matrix),
        ))

    def run_trivial(self) -> RunResult:
        """n < 3: the only fold is the empty move string (one evaluation if budget allows)."""
        if not self.done:
            self.evaluate("")
            self.record("", np.zeros((0, len(self.alphabet))))
        return self.result()

    def result(self) -> RunResult:
        ev = self.evaluator
        if not self._history or self._history[-1][0] != ev.evaluations:
            self._history.append((ev.evaluations, ev.best_energy))
        return RunResult(
            algorithm=self.algorithm,
            sequence=self.sequence,
            dim=self.dim,
            seed=self.seed,
            params=dict(self.params),
            best_moves=ev.best_moves,
            best_energy=ev.best_energy,
            evaluations=ev.evaluations,
            evals_to_best=ev.evals_to_best,
            runtime_s=time.perf_counter() - self._start,
            history=tuple(self._history),
            cancelled=self.cancelled,
            target_reached=ev.target_reached,
            stats=dict(self.stats),
        )
