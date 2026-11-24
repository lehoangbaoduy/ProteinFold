"""GA-specific tests (PRD §5.3 GA; plan T2.2). The shared contract is in test_algorithm_contract.py."""

import itertools

import numpy as np
import pytest

from hpfold.algorithms import ga
from hpfold.hp_model import energy, fold_coords, is_valid

S1_1 = "HPHPPHHPHPPHPHHPPHPH"  # E* = -9 (2D)


@pytest.mark.parametrize(
    "bad",
    [
        {"pop_size": 1}, {"pc": -0.1}, {"pc": 1.5}, {"pm": 0.0}, {"pm": 1.2},
        {"tournament_k": 0}, {"tournament_k": 500}, {"elite": -1}, {"elite": 200},
        {"crossover": "three_point"}, {"strategy": "magic"}, {"penalty": -1.0},
    ],
)
def test_params_validation(bad):
    with pytest.raises(ValueError):
        ga.Params(**bad)


@pytest.mark.parametrize("kind", ["one_point", "two_point", "uniform"])
def test_crossover_children_inherit_each_gene_from_a_parent(kind):
    rng = np.random.default_rng(0)
    a, b = "FFFFFFFFFFFF", "LLLLLLLLLLLL"
    for _ in range(50):
        c1, c2 = ga.crossover(a, b, kind, rng)
        assert len(c1) == len(c2) == len(a)
        for i in range(len(a)):
            assert {c1[i], c2[i]} == {a[i], b[i]}  # genes are swapped, never invented


def test_one_point_crossover_is_a_single_cut():
    c1, _ = ga.crossover("FFFFFFFF", "LLLLLLLL", "one_point", np.random.default_rng(3))
    assert c1.count("FL") + c1.count("LF") <= 1


def test_mutation_rate_one_changes_every_gene():
    out = ga.mutate("FLRFLRFLR", 1.0, "FLR", np.random.default_rng(0))
    assert all(x != y for x, y in zip(out, "FLRFLRFLR"))


def test_mutation_stays_in_alphabet():
    out = ga.mutate("FLRUDFLRUD", 1.0, "FLRUD", np.random.default_rng(1))
    assert set(out) <= set("FLRUD")


def test_tournament_with_whole_population_picks_the_best():
    fitness = np.array([3.0, -2.0, 0.5, -1.0])
    assert ga.tournament(fitness, k=4, rng=np.random.default_rng(0), replace=False) == 1


def test_elitism_never_loses_the_population_best():
    frames = []
    params = ga.Params(pop_size=30, elite=1, strategy="repair")
    ga.run(S1_1, 2, params=params, seed=0, budget=3000, on_progress=frames.append)
    energies = [energy(S1_1, fold_coords(S1_1, f.current_best_moves)) for f in frames]
    assert all(b <= a for a, b in itertools.pairwise(energies))


def test_repair_strategy_keeps_population_valid():
    frames = []
    ga.run(S1_1, 2, params=ga.Params(strategy="repair"), seed=1, budget=2000, on_progress=frames.append)
    assert all(is_valid(fold_coords(S1_1, f.current_best_moves)) for f in frames)


@pytest.mark.parametrize("dim,threshold", [(2, -7), (3, -8)])
def test_reaches_reasonable_energy_on_a_20mer(dim, threshold):
    r = ga.run(S1_1, dim, seed=0, budget=20_000)
    assert r.best_energy <= threshold


def test_params_recorded_in_result():
    p = ga.Params(pop_size=40)
    r = ga.run(S1_1, 2, params=p, seed=0, budget=200)
    assert r.params["pop_size"] == 40 and r.params["crossover"] == p.crossover


# ---------------------------------------------------------------- review round 1 ----

def _boundaries(child, a, b):
    source = ["a" if c == x else "b" for c, x, y in zip(child, a, b)]
    return sum(1 for s, t in itertools.pairwise(source) if s != t)


@pytest.mark.parametrize("kind,max_cuts", [("one_point", 1), ("two_point", 2)])
def test_point_crossovers_really_cut(kind, max_cuts):
    rng = np.random.default_rng(2)
    a, b = "F" * 16, "L" * 16
    for _ in range(100):
        c1, c2 = ga.crossover(a, b, kind, rng)
        assert 1 <= _boundaries(c1, a, b) <= max_cuts
        assert c1 not in (a, b) and c2 not in (a, b)


def test_two_point_sometimes_makes_two_cuts():
    rng = np.random.default_rng(5)
    cuts = {_boundaries(ga.crossover("F" * 16, "L" * 16, "two_point", rng)[0], "F" * 16, "L" * 16)
            for _ in range(100)}
    assert 2 in cuts


def test_uniform_crossover_mixes():
    rng = np.random.default_rng(6)
    children = [ga.crossover("F" * 30, "L" * 30, "uniform", rng)[0] for _ in range(20)]
    assert all(0 < c.count("F") < 30 for c in children)


def test_mutation_rate_is_respected_statistically():
    rng = np.random.default_rng(7)
    moves = "F" * 1000
    changed = sum(sum(x != y for x, y in zip(ga.mutate(moves, 0.1, "FLR", rng), moves)) for _ in range(20))
    assert 0.08 < changed / 20_000 < 0.12


def test_tiny_mutation_rate_usually_returns_input():
    rng = np.random.default_rng(8)
    assert sum(ga.mutate("FLRFLR", 1e-6, "FLR", rng) == "FLRFLR" for _ in range(100)) == 100


def test_tournament_with_replacement_favours_the_best():
    rng = np.random.default_rng(9)
    fitness = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    wins = np.bincount([ga.tournament(fitness, 3, rng) for _ in range(3000)], minlength=6)
    assert wins[5] > wins[4] > wins[0]


def test_default_mutation_rate_is_two_per_child():
    r = ga.run(S1_1, 2, seed=0, budget=200)
    assert r.stats["pm"] == pytest.approx(2 / (len(S1_1) - 2))


@pytest.mark.parametrize("pc", [0.0, 1.0])
def test_crossover_probability_extremes_run(pc):
    r = ga.run(S1_1, 2, params=ga.Params(pc=pc), seed=0, budget=1000)
    assert r.evaluations == 1000 and r.best_energy is not None


def test_provisional_defaults_from_pilot_sweep():
    # Decision D19: same 7-config x 4-seed pilot on S1-1 as ACO/PSO (D16) picked k = 5.
    assert ga.Params().tournament_k == 5
