"""Unit tests for hpfold.folds: random_valid_fold and repair (PRD FR-6, FR-7)."""

import numpy as np
import pytest

from hpfold import folds
from hpfold.folds import random_valid_fold, repair
from hpfold.hp_model import MOVES_2D, MOVES_3D, decode, is_valid


@pytest.mark.parametrize("dim,n", [(2, 3), (2, 20), (2, 64), (2, 100), (3, 3), (3, 48), (3, 100)])
def test_random_valid_fold_is_valid(dim, n):
    rng = np.random.default_rng(0)
    alphabet = MOVES_2D if dim == 2 else MOVES_3D
    for _ in range(5):
        moves = random_valid_fold(n, rng, dim)
        assert len(moves) == n - 2
        assert set(moves) <= set(alphabet)
        assert is_valid(decode(moves))


@pytest.mark.parametrize("n", [0, 1, 2])
def test_random_valid_fold_trivial_lengths(n):
    assert random_valid_fold(n, np.random.default_rng(0), 2) == ""


def test_random_valid_fold_is_deterministic_for_a_seed():
    a = random_valid_fold(50, np.random.default_rng(42), 2)
    b = random_valid_fold(50, np.random.default_rng(42), 2)
    assert a == b


def test_random_valid_fold_varies_across_seeds():
    folds = {random_valid_fold(30, np.random.default_rng(s), 2) for s in range(10)}
    assert len(folds) > 1


def test_random_valid_fold_rejects_bad_dim():
    with pytest.raises(ValueError, match="dim"):
        random_valid_fold(10, np.random.default_rng(0), 4)


def test_repair_leaves_valid_fold_unchanged():
    moves = "FLLFRRFF"
    assert repair(moves, np.random.default_rng(0), 2) == moves


def test_repair_fixes_a_collision():
    moves = "LLLFF"  # bead 4 collides with bead 0
    assert not is_valid(decode(moves))
    fixed = repair(moves, np.random.default_rng(0), 2)
    assert len(fixed) == len(moves)
    assert is_valid(decode(fixed))


def test_repair_keeps_valid_prefix():
    # Only the move that causes the first collision (index 2) and later moves may change.
    moves = "LLLFF"
    fixed = repair(moves, np.random.default_rng(0), 2)
    assert fixed[:2] == moves[:2]


@pytest.mark.parametrize("dim", [2, 3])
def test_repair_always_returns_valid_folds(dim):
    rng = np.random.default_rng(7)
    alphabet = list(MOVES_2D if dim == 2 else MOVES_3D)
    for _ in range(200):
        n = int(rng.integers(3, 60))
        moves = "".join(rng.choice(alphabet, size=n - 2))
        fixed = repair(moves, rng, dim)
        assert len(fixed) == len(moves)
        assert set(fixed) <= set(alphabet)
        assert is_valid(decode(fixed))


def test_repair_is_deterministic_for_a_seed():
    moves = "LLLLLLLLLLRRRRRRRRRR"
    a = repair(moves, np.random.default_rng(3), 2)
    b = repair(moves, np.random.default_rng(3), 2)
    assert a == b


def test_tiny_backtrack_budget_still_returns_valid_folds(monkeypatch):
    # Force the per-attempt budget to run out often: restarts must still deliver a valid fold.
    monkeypatch.setattr(folds, "BACKTRACKS_PER_BEAD", 0)
    rng = np.random.default_rng(11)
    for _ in range(20):
        assert is_valid(decode(random_valid_fold(60, rng, 2)))
        assert is_valid(decode(repair("L" * 58, rng, 2)))


def test_raises_only_after_all_restarts_fail(monkeypatch):
    monkeypatch.setattr(folds, "BACKTRACKS_PER_BEAD", 0)
    monkeypatch.setattr(folds, "MAX_ATTEMPTS", 1)
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="valid fold"):
        for _ in range(200):  # some attempt will hit a dead end with zero budget
            random_valid_fold(100, rng, 2)


def test_long_2d_folds_never_raise():
    # Regression for the review finding H1 (rare cap hits at n = 100 with plain DFS).
    rng = np.random.default_rng(2024)
    for _ in range(300):
        assert is_valid(decode(random_valid_fold(100, rng, 2)))


def test_repair_valid_fold_does_not_consume_rng():
    rng = np.random.default_rng(5)
    state = rng.bit_generator.state
    assert repair("FLLFRRFF", rng, 2) == "FLLFRRFF"
    assert rng.bit_generator.state == state


def test_repair_fixes_3d_collision():
    moves = "UUUFF"  # U,U,U circles back: bead 4 collides
    assert not is_valid(decode(moves))
    fixed = repair(moves, np.random.default_rng(1), 3)
    assert len(fixed) == len(moves) and is_valid(decode(fixed))


def test_repair_usually_keeps_collision_free_prefix():
    # "Minimal change" is a heuristic, not a guarantee: DFS may back up into the prefix
    # when everything beyond it is a dead-end pocket. It must still hold in most cases.
    rng = np.random.default_rng(9)
    kept = 0
    trials = 100
    for _ in range(trials):
        moves = "".join(rng.choice(list("FLR"), size=40))
        coords = decode(moves)
        # index of the first bead that lands on an occupied site
        seen, first_bad = set(), None
        for k, c in enumerate(coords):
            if c in seen:
                first_bad = k
                break
            seen.add(c)
        fixed = repair(moves, rng, 2)
        keep = max(len(moves) if first_bad is None else first_bad - 3, 0)
        kept += fixed[:keep] == moves[:keep]
    assert kept >= 0.9 * trials


@pytest.mark.parametrize("moves", [""])
def test_repair_trivial(moves):
    assert repair(moves, np.random.default_rng(0), 2) == ""


@pytest.mark.parametrize("bad_n", [2.5, "10", None, True])
def test_random_valid_fold_rejects_non_integer_n(bad_n):
    with pytest.raises(TypeError):
        random_valid_fold(bad_n, np.random.default_rng(0), 2)


def test_random_valid_fold_accepts_numpy_integers():
    moves = random_valid_fold(np.int64(20), np.random.default_rng(0), 2)
    assert len(moves) == 18 and is_valid(decode(moves))


def test_random_valid_fold_rejects_negative_n():
    with pytest.raises(ValueError, match="n"):
        random_valid_fold(-1, np.random.default_rng(0), 2)


def test_repair_rejects_bad_dim_and_non_string():
    with pytest.raises(ValueError, match="dim"):
        repair("FL", np.random.default_rng(0), 4)
    with pytest.raises(TypeError):
        repair(None, np.random.default_rng(0), 2)  # type: ignore[arg-type]


def test_repair_rejects_moves_outside_alphabet():
    with pytest.raises(ValueError, match="move"):
        repair("FLU", np.random.default_rng(0), 2)
