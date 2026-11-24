"""PSO-specific tests (PRD §5.3 PSO; plan T2.4; edge case E9). Shared rules: test_algorithm_contract.py."""

import numpy as np
import pytest

from hpfold.algorithms import pso
from hpfold.hp_model import MOVES_2D, decode, is_valid

S1_1 = "HPHPPHHPHPPHPHHPPHPH"  # E* = -9 (2D)


@pytest.mark.parametrize(
    "bad",
    [
        {"swarm_size": 0}, {"swarm_size": 2.5}, {"w": -0.1}, {"c1": -1.0}, {"c2": -1.0},
        {"epsilon": 0.0}, {"epsilon": 0.2}, {"topology": "star"}, {"strategy": "magic"},
        {"penalty": -1.0}, {"w": float("nan")}, {"c1": float("inf")},
    ],
)
def test_params_validation(bad):
    with pytest.raises(ValueError):
        pso.Params(**bad)


def _uniform(rows, cols):
    return np.full((rows, cols), 1.0 / cols)


def test_update_rows_sum_to_one_with_floor_and_no_mutation():
    probs = _uniform(4, 3)
    before = probs.copy()
    new = pso.update_probabilities(probs, "FLRF", "RRRR", MOVES_2D, w=0.5, c1=1.0, c2=1.0,
                                   r1=0.7, r2=0.4, epsilon=0.02)
    assert np.array_equal(probs, before), "input must not be mutated"
    assert np.allclose(new.sum(axis=1), 1.0)
    assert new.min() >= 0.02 - 1e-12


def test_update_moves_probability_toward_the_bests():
    probs = _uniform(2, 3)
    new = pso.update_probabilities(probs, "LL", "LL", MOVES_2D, w=0.5, c1=1.0, c2=1.0,
                                   r1=1.0, r2=1.0, epsilon=0.01)
    assert new[0, 1] > probs[0, 1] and new[0, 0] < probs[0, 0]


def test_floor_prevents_collapse():
    # Edge case E9: pulling toward the same fold forever must never zero out a move.
    probs = _uniform(5, 3)
    for _ in range(500):
        probs = pso.update_probabilities(probs, "FFFFF", "FFFFF", MOVES_2D, w=0.9, c1=2.0, c2=2.0,
                                         r1=1.0, r2=1.0, epsilon=0.01)
    assert probs.min() >= 0.01 - 1e-12 and probs[:, 0].max() < 1.0


def test_sampling_follows_the_probabilities():
    rng = np.random.default_rng(0)
    probs = np.array([[0.7, 0.2, 0.1]] * 2000)
    moves = pso.sample(probs, MOVES_2D, rng)
    assert len(moves) == 2000 and set(moves) <= set(MOVES_2D)
    freq = [moves.count(m) / 2000 for m in MOVES_2D]
    assert np.allclose(freq, [0.7, 0.2, 0.1], atol=0.03)


def test_diversity_is_logged_and_decreases():
    frames = []
    pso.run(S1_1, 2, seed=0, budget=6000, on_progress=frames.append)
    assert all(0.0 <= f.diversity <= 1.0 for f in frames)
    assert frames[-1].diversity < frames[0].diversity


def test_repair_strategy_keeps_swarm_positions_valid():
    frames = []
    pso.run(S1_1, 2, params=pso.Params(strategy="repair"), seed=1, budget=2000, on_progress=frames.append)
    assert all(is_valid(decode(f.current_best_moves)) for f in frames)


def test_ring_topology_runs_and_differs_from_global():
    a = pso.run(S1_1, 2, params=pso.Params(topology="global"), seed=2, budget=2000)
    b = pso.run(S1_1, 2, params=pso.Params(topology="ring"), seed=2, budget=2000)
    assert b.best_energy is not None and (a.history != b.history or a.best_moves != b.best_moves)


def test_frame_matrix_is_a_particle_probability_matrix():
    frames = []
    pso.run(S1_1, 2, params=pso.Params(epsilon=0.03), seed=3, budget=1000, on_progress=frames.append)
    for f in frames:
        m = np.asarray(f.matrix)
        assert np.allclose(m.sum(axis=1), 1.0) and m.min() >= 0.03 - 1e-9


@pytest.mark.parametrize("dim,threshold", [(2, -7), (3, -8)])
def test_reaches_reasonable_energy_on_a_20mer(dim, threshold):
    r = pso.run(S1_1, dim, seed=0, budget=20_000)
    assert r.best_energy <= threshold


# ---------------------------------------------------------------- review round 1 ----

def test_neighbourhood_best_global_and_ring():
    fit = [3.0, 1.0, 2.0, 0.0]
    assert pso.neighbourhood_best(fit, "global") == [3, 3, 3, 3]
    # ring neighbours of k are (k-1, k, k+1) mod 4
    assert pso.neighbourhood_best(fit, "ring") == [3, 1, 3, 3]
    assert pso.neighbourhood_best([5.0], "ring") == [0]
    assert pso.neighbourhood_best([5.0, 1.0], "ring") == [1, 1]


def test_update_exact_with_asymmetric_pulls():
    probs = _uniform(1, 3)
    new = pso.update_probabilities(probs, "F", "L", MOVES_2D, w=0.5, c1=1.0, c2=2.0,
                                   r1=0.5, r2=0.5, epsilon=0.01)
    raw = np.array([1 / 6 + 0.5, 1 / 6 + 1.0, 1 / 6])  # F gets c1*r1, L gets c2*r2
    expected = 0.01 + (1 - 3 * 0.01) * raw / raw.sum()
    assert np.allclose(new[0], expected)


def test_inertia_only_keeps_probabilities():
    probs = np.array([[0.6, 0.3, 0.1]])
    new = pso.update_probabilities(probs, "F", "L", MOVES_2D, w=1.0, c1=0.0, c2=0.0,
                                   r1=1.0, r2=1.0, epsilon=0.02)
    assert np.allclose(new, 0.02 + (1 - 3 * 0.02) * probs)


def test_zero_total_falls_back_to_uniform():
    new = pso.update_probabilities(_uniform(2, 3), "FF", "LL", MOVES_2D, w=0.0, c1=0.0, c2=0.0,
                                   r1=0.0, r2=0.0, epsilon=0.01)
    assert np.allclose(new, 1 / 3)


def test_r1_and_r2_are_drawn_independently(monkeypatch):
    seen = []
    original = pso.update_probabilities

    def spy(*args, **kwargs):
        seen.append((args[7], args[8]))
        return original(*args, **kwargs)

    monkeypatch.setattr(pso, "update_probabilities", spy)
    pso.run(S1_1, 2, seed=0, budget=300)
    assert seen and any(r1 != r2 for r1, r2 in seen)


def test_diversity_comes_from_realized_positions():
    # One particle: its realized move frequencies are one-hot, so diversity is exactly 0,
    # although its probability matrix (floored) is never one-hot.
    frames = []
    pso.run(S1_1, 2, params=pso.Params(swarm_size=1), seed=0, budget=200, on_progress=frames.append)
    assert frames and all(f.diversity == 0.0 for f in frames)
    assert all(np.asarray(f.matrix).min() > 0 for f in frames)
