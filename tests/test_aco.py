"""ACO-specific tests (PRD §5.3 ACO; plan T2.3; edge case E8). Shared rules: test_algorithm_contract.py."""

import math

import numpy as np
import pytest

from hpfold.algorithms import aco
from hpfold.hp_model import MOVES_2D, MOVES_3D, decode, energy, is_valid

S1_1 = "HPHPPHHPHPPHPHHPPHPH"  # E* = -9 (2D)


@pytest.mark.parametrize(
    "bad",
    [
        {"n_ants": 0}, {"alpha": -1.0}, {"beta": -0.5}, {"rho": 0.0}, {"rho": 1.5},
        {"gamma": 0.0}, {"tau_min": 0.0}, {"tau_min": 5.0, "tau_max": 1.0},
        {"tau0": 100.0}, {"deposit": "everyone"}, {"backtrack_limit": -1}, {"max_restarts": -1},
        {"n_ants": 2.5}, {"backtrack_limit": 1.5}, {"alpha": float("nan")}, {"beta": float("inf")},
        {"gamma": float("nan")}, {"rho": float("nan")},
    ],
)
def test_params_validation(bad):
    with pytest.raises(ValueError):
        aco.Params(**bad)


def test_new_contacts_counts_only_non_bonded_h_neighbours():
    seq = "HPPHH"
    placed = {(0, 0, 0): 0, (1, 0, 0): 1, (1, 1, 0): 2}
    # bead 3 (H) at (0, 1, 0): touches bead 0 (H, non-bonded) and bead 2 (P, bonded) -> 1
    assert aco.new_contacts(seq, 3, (0, 1, 0), placed) == 1
    # a P bead never makes contacts
    assert aco.new_contacts("HPPPH", 3, (0, 1, 0), placed) == 0


@pytest.mark.parametrize("dim,n", [(2, 20), (2, 64), (2, 100), (3, 48), (3, 100)])
def test_ants_always_build_valid_folds(dim, n):
    alphabet = MOVES_2D if dim == 2 else MOVES_3D
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("HP"), size=n))
    tau = np.ones((n - 2, len(alphabet)))
    for _ in range(10):
        ant = aco.construct_ant(seq, tau, aco.Params(), alphabet, rng)
        assert len(ant.moves) == n - 2 and set(ant.moves) <= set(alphabet)
        assert is_valid(decode(ant.moves))


def test_dead_ends_trigger_counted_restarts_and_stay_valid():
    # Zero backtracks allowed: every dead end forces a restart (edge case E8).
    params = aco.Params(backtrack_limit=0, max_restarts=1000)
    rng = np.random.default_rng(1)
    tau = np.ones((98, 3))
    restarts = 0
    for _ in range(30):
        ant = aco.construct_ant("H" * 100, tau, params, MOVES_2D, rng)
        assert is_valid(decode(ant.moves))
        restarts += ant.restarts
    assert restarts > 0


def test_fallback_after_max_restarts_is_valid_and_flagged():
    params = aco.Params(backtrack_limit=0, max_restarts=0)
    rng = np.random.default_rng(2)
    tau = np.ones((98, 3))
    ants = [aco.construct_ant("H" * 100, tau, params, MOVES_2D, rng) for _ in range(300)]
    assert all(is_valid(decode(a.moves)) for a in ants)
    assert any(a.fallback for a in ants)


def test_run_reports_restart_stats():
    r = aco.run("H" * 60, 2, params=aco.Params(backtrack_limit=0), seed=0, budget=200)
    assert r.stats["restarts"] >= 0 and "fallbacks" in r.stats


def test_pheromone_update_evaporates_deposits_and_respects_bounds():
    p = aco.Params(rho=0.5, tau_min=0.1, tau_max=2.0)
    tau = np.full((3, 3), 1.0)
    before = tau.copy()
    new = aco.update_pheromone(tau, [("FLR", 1.0)], p, MOVES_2D)
    assert np.array_equal(tau, before), "input must not be mutated"
    assert new[0, 0] == pytest.approx(1.5) and new[1, 1] == pytest.approx(1.5) and new[2, 2] == pytest.approx(1.5)
    assert new[0, 1] == pytest.approx(0.5)
    many = new
    for _ in range(50):
        many = aco.update_pheromone(many, [("FLR", 1.0)], p, MOVES_2D)
    assert many.max() <= 2.0 and many.min() >= 0.1


def test_relative_quality():
    assert aco.quality(-9, -9) == pytest.approx(1.0)
    assert 0 < aco.quality(-3, -9) < aco.quality(-6, -9) < 1
    assert aco.quality(0, 0) == pytest.approx(1.0)


def test_pheromone_concentrates_over_time():
    frames = []
    aco.run(S1_1, 2, seed=0, budget=4000, on_progress=frames.append)
    assert frames[-1].diversity < frames[0].diversity


def test_every_evaluated_fold_is_valid():
    # ACO builds valid folds by construction, so the population best is always valid.
    frames = []
    aco.run(S1_1, 3, seed=3, budget=1500, on_progress=frames.append)
    assert all(is_valid(decode(f.current_best_moves)) for f in frames)


@pytest.mark.parametrize("dim,threshold", [(2, -7), (3, -8)])
def test_reaches_reasonable_energy_on_a_20mer(dim, threshold):
    r = aco.run(S1_1, dim, seed=0, budget=20_000)
    assert r.best_energy <= threshold


# ---------------------------------------------------------------- review round 1 ----

def test_bonded_h_predecessor_is_not_a_contact():
    placed = {(0, 0, 0): 0, (1, 0, 0): 1, (1, 1, 0): 2}
    # bead 3 at (2, 1, 0): its only placed neighbour is bead 2, which is bonded to it
    assert aco.new_contacts("HHHH", 3, (2, 1, 0), placed) == 0
    # at (0, 1, 0) it also touches bead 0: exactly one contact, the bonded bead 2 excluded
    assert aco.new_contacts("HHHH", 3, (0, 1, 0), placed) == 1


def test_backtracking_path_without_restarts():
    params = aco.Params(backtrack_limit=100_000, max_restarts=0)
    rng = np.random.default_rng(4)
    tau = np.ones((98, 3))
    ants = [aco.construct_ant("H" * 100, tau, params, MOVES_2D, rng) for _ in range(60)]
    assert all(is_valid(decode(a.moves)) and not a.fallback and a.restarts == 0 for a in ants)
    assert sum(a.backtracks for a in ants) > 0


def test_choice_matrix_rows_sum_to_one():
    tau = np.array([[0.1, 2.0, 1.0], [1.0, 1.0, 1.0]])
    m = aco.choice_matrix(tau, 1.5)
    assert np.allclose(m.sum(axis=1), 1.0) and m[0, 1] > m[0, 2] > m[0, 0]


def test_deposit_modes_differ():
    a = aco.run(S1_1, 2, params=aco.Params(deposit="global_best"), seed=3, budget=1500)
    b = aco.run(S1_1, 2, params=aco.Params(deposit="iteration_best"), seed=3, budget=1500)
    assert a.history != b.history or a.best_moves != b.best_moves


def test_contact_heuristic_biases_toward_contacts():
    rng = np.random.default_rng(5)
    tau = np.ones((len(S1_1) - 2, 3))

    def mean_energy(beta):
        p = aco.Params(beta=beta)
        return np.mean([energy(S1_1, decode(aco.construct_ant(S1_1, tau, p, MOVES_2D, rng).moves))
                        for _ in range(200)])

    assert mean_energy(4.0) < mean_energy(0.0) - 1.0


@pytest.mark.parametrize("kw", [{"gamma": 1e-4}, {"beta": 1e4}, {"gamma": 1e-6, "beta": 50.0}])
def test_extreme_heuristic_settings_do_not_crash(kw):
    r = aco.run(S1_1, 2, params=aco.Params(**kw), seed=0, budget=300)
    assert r.best_energy is not None and is_valid(decode(r.best_moves))
    assert all(math.isfinite(e) for _, e in r.history if e is not None)


def test_diversity_measures_the_colony_and_can_collapse():
    # Diversity is computed from the ants' realized moves (comparable with the GA),
    # not from the clamped pheromone, so a converged colony approaches 0.
    frames = []
    wide = aco.Params(rho=0.5, tau_min=0.001, tau_max=10.0)  # wide bounds allow full convergence
    aco.run(S1_1, 2, params=wide, seed=0, budget=6000, on_progress=frames.append)
    assert frames[0].diversity > 0.8 and frames[-1].diversity < 0.1
