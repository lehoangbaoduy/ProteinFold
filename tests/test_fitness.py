"""Unit tests for hpfold.fitness.evaluate (PRD FR-5, §5.2 penalty vs repair)."""

import math

import numpy as np
import pytest

from hpfold.fitness import DEFAULT_PENALTY, Evaluation, evaluate
from hpfold.hp_model import decode, energy, is_valid


def test_valid_fold_penalty_strategy():
    ev = evaluate("HPHHHHPHHP", "FLLFRRFF", dim=2)
    assert ev == Evaluation(moves="FLLFRRFF", energy=-3, collisions=0, fitness=-3.0)
    assert ev.valid


def test_dim_is_required():
    with pytest.raises(TypeError):
        evaluate("HPHHHHPHHP", "FLLFRRFF")  # type: ignore[call-arg]


def test_penalty_strategy_adds_lambda_per_collision():
    ev = evaluate("PPPPPPP", "LLLFF", dim=2, strategy="penalty", penalty=5.0)
    assert ev.collisions == 1
    assert not ev.valid
    assert ev.fitness == ev.energy + 5.0


def test_zero_penalty_means_raw_energy():
    ev = evaluate("HHHHHHH", "LLLFF", dim=2, penalty=0.0)
    assert ev.fitness == ev.energy


def test_default_penalty_is_positive():
    assert DEFAULT_PENALTY > 0


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_non_finite_or_negative_penalty_rejected(bad):
    with pytest.raises(ValueError, match="penalty"):
        evaluate("HHHH", "FF", dim=2, penalty=bad)


def test_moves_checked_against_dim_for_both_strategies():
    with pytest.raises(ValueError, match="not allowed in 2D"):
        evaluate("HHHH", "UU", dim=2)
    with pytest.raises(ValueError, match="not allowed in 2D"):
        evaluate("HHHH", "UU", dim=2, strategy="repair", rng=np.random.default_rng(0))


def test_repair_strategy_returns_valid_repaired_fold():
    seq = "HHHHHHH"
    ev = evaluate(seq, "LLLFF", dim=2, strategy="repair", rng=np.random.default_rng(0))
    assert ev.valid
    assert ev.moves != "LLLFF"
    coords = decode(ev.moves)
    assert is_valid(coords)
    assert ev.energy == energy(seq, coords)
    assert ev.fitness == ev.energy


def test_repair_strategy_leaves_valid_fold_unchanged():
    ev = evaluate("HPHHHHPHHP", "FLLFRRFF", dim=2, strategy="repair", rng=np.random.default_rng(0))
    assert ev.moves == "FLLFRRFF" and ev.energy == -3


def test_repair_in_3d_may_use_3d_moves_even_if_input_has_none():
    # 3D genome without U/D that collides: repair must work in 3D (review finding M2).
    rng = np.random.default_rng(4)
    ev = evaluate("HHHHHHHHHHHH", "LLLLLLLLLL", dim=3, strategy="repair", rng=rng)
    assert ev.valid and set(ev.moves) <= set("FLRUD")


def test_repair_strategy_requires_rng():
    with pytest.raises(ValueError, match="rng"):
        evaluate("HHHHH", "LLL", dim=2, strategy="repair")


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="strategy"):
        evaluate("HHHH", "FF", dim=2, strategy="magic")


@pytest.mark.parametrize("bad_seq", ["hhhh", "HHXH", ""])
def test_invalid_sequence_rejected(bad_seq):
    with pytest.raises(ValueError, match="sequence"):
        evaluate(bad_seq, "FF", dim=2)


def test_wrong_move_count_rejected():
    with pytest.raises(ValueError, match="n-2"):
        evaluate("HHHHH", "FF", dim=2)


@pytest.mark.parametrize("seq", ["H", "HP"])
def test_trivial_sequences(seq):
    ev = evaluate(seq, "", dim=2, strategy="repair", rng=np.random.default_rng(0))
    assert ev == Evaluation(moves="", energy=0, collisions=0, fitness=0.0)


def test_3d_moves_are_evaluated():
    ev = evaluate("HHHH", "UU", dim=3)
    assert ev.valid and ev.energy == -1  # (0,0,0) and (0,0,1) touch


def test_evaluation_is_immutable():
    ev = evaluate("HHHH", "LL", dim=2)
    with pytest.raises(AttributeError):
        ev.energy = 5  # type: ignore[misc]
