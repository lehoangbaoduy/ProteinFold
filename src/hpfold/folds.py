"""Growing valid (self-avoiding) folds: random initialization and repair (PRD FR-6, FR-7).

Both functions use one depth-first search that places beads one at a time and
backtracks out of dead ends (a bead with no free neighbouring site):

* `random_valid_fold` tries the moves at each position in a random order.
* `repair` tries the fold's *own* move first and only then the alternatives, so
  the fold is usually altered only from its first collision onward. (DFS may
  back up further when the region past the collision is a dead-end pocket.)

Plain DFS can occasionally wander deep into a trapped region and need a huge
number of backtracks. So each attempt gets a backtrack budget proportional to
n, and an exhausted attempt is abandoned and restarted with fresh random
choices (random restarts). Repair restarts keep preferring the original moves
for the first half of the attempts and then fall back to unbiased growth.

All randomness comes from the caller's numpy Generator, so results are
reproducible for a given seed.
"""

from __future__ import annotations

import numbers

import numpy as np

from hpfold.hp_model import (
    INITIAL_HEADING,
    INITIAL_UP,
    ORIGIN,
    Coord,
    add,
    alphabet_for,
    check_moves,
    decode,
    is_valid,
    step_frame,
)

BACKTRACKS_PER_BEAD = 50  # per-attempt backtrack budget = this * n
MAX_ATTEMPTS = 100  # random restarts before giving up (never reached in practice)


def random_valid_fold(n: int, rng: np.random.Generator, dim: int) -> str:
    """A random self-avoiding fold of n beads (n-2 moves) on the 2D or 3D lattice."""
    if isinstance(n, bool) or not isinstance(n, numbers.Integral):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    n = int(n)  # accept numpy integers (e.g. np.int64 from array shapes)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    alphabet_for(dim)  # validates dim
    return _grow(n, rng, dim, preferred=None)


def repair(moves: str, rng: np.random.Generator, dim: int) -> str:
    """Return a valid fold of the same length, changing `moves` only where needed.

    A fold that is already valid is returned unchanged without using the rng.
    """
    if not isinstance(moves, str):
        raise TypeError(f"moves must be a string, got {type(moves).__name__}")
    check_moves(moves, dim)  # validates dim and alphabet
    if is_valid(decode(moves)):
        return moves
    return _grow(len(moves) + 2, rng, dim, preferred=moves)


def _grow(n: int, rng: np.random.Generator, dim: int, preferred: str | None) -> str:
    if n <= 2:
        return ""
    budget = BACKTRACKS_PER_BEAD * n
    for attempt in range(MAX_ATTEMPTS):
        pref = preferred if attempt < MAX_ATTEMPTS // 2 else None
        result = _attempt(n, rng, alphabet_for(dim), pref, budget)
        if result is not None:
            return result
    raise RuntimeError(f"could not grow a valid fold of {n} beads in {dim}D after {MAX_ATTEMPTS} attempts")


def _candidate_order(alphabet: str, rng: np.random.Generator, preferred: str | None) -> list[str]:
    shuffled = [str(m) for m in rng.permutation(list(alphabet))]
    if preferred is None:
        return shuffled
    return [preferred] + [m for m in shuffled if m != preferred]


def _attempt(
    n: int, rng: np.random.Generator, alphabet: str, preferred: str | None, budget: int
) -> str | None:
    """One DFS attempt; returns the moves, or None if the backtrack budget runs out."""
    target = n - 2

    def options_at(depth: int) -> list[str]:
        pref = None if preferred is None else preferred[depth]
        return _candidate_order(alphabet, rng, pref)

    coords: list[Coord] = [ORIGIN, INITIAL_HEADING]
    occupied = set(coords)
    frames = [(INITIAL_HEADING, INITIAL_UP)]  # frame after each placed bond
    pending = [options_at(0)]  # untried moves at each depth
    chosen: list[str] = []
    backtracks = 0

    while len(chosen) < target:
        if not pending[-1]:
            backtracks += 1
            if not chosen or backtracks > budget:
                return None
            pending.pop()
            frames.pop()
            chosen.pop()
            occupied.discard(coords.pop())
            continue
        move = pending[-1].pop(0)
        heading, up = step_frame(*frames[-1], move)
        site = add(coords[-1], heading)
        if site in occupied:
            continue
        chosen.append(move)
        coords.append(site)
        occupied.add(site)
        frames.append((heading, up))
        if len(chosen) < target:
            pending.append(options_at(len(chosen)))

    return "".join(chosen)
