"""Contract every search algorithm must satisfy (PRD §5.3, FR-T1; §8 E4-E7, E10, E12, E13).

Parametrized over the registry, so each algorithm added to
`hpfold.algorithms.ALGORITHMS` is held to the same rules automatically.
"""

import itertools
import json
import math

import numpy as np
import pytest

from hpfold.algorithms import ALGORITHMS, Evaluator, Frame, RunResult
from hpfold.hp_model import alphabet_for, energy, fold_coords, is_valid

SANITY_SEQ = "HPHHHHPHHP"  # a fold with E = -3 exists
S1_1 = "HPHPPHHPHPPHPHHPPHPH"  # 2D benchmark, E* = -9
NAMES = sorted(ALGORITHMS)


def run(name, sequence, dim=2, budget=2000, **kw):
    return ALGORITHMS[name].run(sequence, dim, budget=budget, **kw)


# ---------------------------------------------------------------- registry ----

def test_registry_contains_required_algorithms():
    assert {"GA"} <= set(ALGORITHMS)


@pytest.mark.parametrize("name", NAMES)
def test_module_exposes_run_and_params(name):
    module = ALGORITHMS[name]
    assert callable(module.run)
    module.Params()  # defaults must be valid


# ---------------------------------------------------------------- results ----

@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dim", [2, 3])
def test_result_is_a_valid_fold_with_consistent_energy(name, dim):
    r = run(name, S1_1, dim, seed=1)
    assert isinstance(r, RunResult)
    assert r.algorithm == name and r.dim == dim and r.seed == 1
    assert set(r.best_moves) <= set(alphabet_for(dim))
    coords = fold_coords(S1_1, r.best_moves)
    assert is_valid(coords)
    assert r.best_energy == energy(S1_1, coords)


@pytest.mark.parametrize("name", NAMES)
def test_budget_is_used_exactly(name):
    r = run(name, S1_1, budget=1234, seed=3)
    assert r.evaluations == 1234
    assert not r.cancelled and not r.target_reached
    assert 1 <= r.evals_to_best <= r.evaluations


@pytest.mark.parametrize("name", NAMES)
def test_same_seed_same_result_different_seed_differs(name):
    a = run(name, S1_1, budget=1500, seed=7)
    b = run(name, S1_1, budget=1500, seed=7)
    assert (a.best_moves, a.best_energy, a.history, a.evals_to_best) == (
        b.best_moves, b.best_energy, b.history, b.evals_to_best)
    others = {run(name, S1_1, budget=1500, seed=s).best_moves for s in range(8, 12)}
    assert len(others | {a.best_moves}) > 1


@pytest.mark.parametrize("name", NAMES)
def test_history_is_monotone(name):
    r = run(name, S1_1, budget=3000, seed=2)
    evals = [e for e, _ in r.history]
    energies = [x for _, x in r.history if x is not None]
    assert evals == sorted(evals) and evals[-1] == r.evaluations
    assert all(b <= a for a, b in itertools.pairwise(energies))
    assert energies[-1] == r.best_energy


@pytest.mark.parametrize("name", NAMES)
def test_target_reached_stops_early(name):
    r = run(name, SANITY_SEQ, budget=50_000, target=-3, seed=0)
    assert r.target_reached and r.best_energy <= -3
    assert r.evaluations < 50_000
    assert r.evals_to_best == r.evaluations


@pytest.mark.parametrize("name", NAMES)
def test_cancel_returns_best_so_far(name):
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 3

    r = run(name, S1_1, budget=100_000, seed=0, should_stop=stop)
    assert r.cancelled and r.evaluations < 100_000
    assert r.best_moves is not None and is_valid(fold_coords(S1_1, r.best_moves))


@pytest.mark.parametrize("name", NAMES)
def test_zero_budget_gives_na_not_zero(name):
    r = run(name, S1_1, budget=0)
    assert r.evaluations == 0
    assert r.best_energy is None and r.best_moves is None and r.evals_to_best is None


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("seq", ["H", "HP", "PP"])
def test_trivial_sequences(name, seq):
    r = run(name, seq)
    assert r.best_moves == "" and r.best_energy == 0 and r.evaluations == 1


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("seq", ["PPPPPPPPPPPP", "PPPPPHPPPPPP"])
def test_no_contacts_possible_runs_normally(name, seq):
    r = run(name, seq, budget=500)
    assert r.best_energy == 0 and r.evaluations == 500


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dim", [2, 3])
def test_all_h_stress(name, dim):
    r = run(name, "H" * 30, dim, budget=1500, seed=4)
    assert r.best_energy < 0 and is_valid(fold_coords("H" * 30, r.best_moves))


@pytest.mark.parametrize("name", NAMES)
def test_library_has_no_length_cap(name):
    seq = "HP" * 75  # 150 residues: above the web cap of 100
    r = run(name, seq, budget=300, seed=0)
    assert len(r.best_moves) == 148


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bad", [{"sequence": "HPXH"}, {"sequence": "hphp"}, {"dim": 4}, {"budget": -1}])
def test_invalid_arguments_rejected(name, bad):
    args = {"sequence": "HPHPHPHP", "dim": 2, "budget": 100} | bad
    with pytest.raises(ValueError):
        ALGORITHMS[name].run(args.pop("sequence"), args.pop("dim"), **args)


# ---------------------------------------------------------------- frames ----

@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("dim", [2, 3])
def test_progress_frames(name, dim):
    frames = []
    r = run(name, S1_1, dim, budget=2000, seed=5, on_progress=frames.append)
    assert frames and all(isinstance(f, Frame) for f in frames)
    alphabet = alphabet_for(dim)
    assert [f.evals for f in frames] == sorted(f.evals for f in frames)
    assert frames[-1].evals == r.evaluations and all(f.budget == 2000 for f in frames)
    best = [f.best_energy for f in frames if f.best_energy is not None]
    assert all(b <= a for a, b in itertools.pairwise(best))
    for f in frames:
        assert 0.0 <= f.diversity <= 1.0
        matrix = np.asarray(f.matrix)
        assert matrix.shape == (len(S1_1) - 2, len(alphabet))
        assert np.allclose(matrix.sum(axis=1), 1.0) and (matrix >= 0).all()
        assert set(f.best_moves or "") <= set(alphabet)
        assert is_valid(fold_coords(S1_1, f.current_best_moves)), "frames must show a valid fold"
    d = frames[-1].to_dict()
    assert {"iter", "evals", "budget", "best_energy", "best_moves", "current_best_moves",
            "diversity", "matrix"} <= set(d)
    assert all(not isinstance(v, float) or math.isfinite(v) for v in d.values())


# ---------------------------------------------------------------- evaluator ----

def test_evaluator_counts_and_enforces_budget():
    ev = Evaluator("HPHHHHPHHP", 2, budget=2, rng=np.random.default_rng(0))
    ev.evaluate("FLLFRRFF")
    assert ev.evaluations == 1 and ev.best_energy == -3 and ev.evals_to_best == 1
    ev.evaluate("FFFFFFFF")
    assert ev.done and ev.best_energy == -3
    with pytest.raises(RuntimeError, match="budget"):
        ev.evaluate("FFFFFFFF")


def test_evaluator_ignores_invalid_folds_for_best():
    ev = Evaluator("HHHHHHH", 2, budget=5, strategy="penalty", rng=np.random.default_rng(0))
    ev.evaluate("LLLFF")  # self-colliding
    assert ev.best_energy is None and ev.best_moves is None


def test_evaluator_target():
    ev = Evaluator("HPHHHHPHHP", 2, budget=10, target=-3, rng=np.random.default_rng(0))
    assert not ev.done
    ev.evaluate("FLLFRRFF")
    assert ev.done and ev.target_reached


# ---------------------------------------------------------------- review round 1 ----

@pytest.mark.parametrize("name", NAMES)
def test_cancel_before_start_spends_nothing(name):
    r = run(name, S1_1, budget=5000, should_stop=lambda: True)
    assert r.cancelled and r.evaluations == 0 and r.best_energy is None


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("budget", [1, 7, 99])
def test_budget_smaller_than_one_iteration(name, budget):
    r = run(name, S1_1, budget=budget, seed=0)
    assert r.evaluations == budget and r.best_energy is not None


@pytest.mark.parametrize("name", NAMES)
def test_progress_callback_does_not_change_the_result(name):
    a = run(name, S1_1, budget=1500, seed=9)
    b = run(name, S1_1, budget=1500, seed=9, on_progress=lambda f: None)
    assert (a.best_moves, a.history, a.evals_to_best) == (b.best_moves, b.history, b.evals_to_best)


@pytest.mark.parametrize("name", NAMES)
def test_frames_agree_with_history(name):
    frames = []
    r = run(name, S1_1, budget=2500, seed=1, on_progress=frames.append)
    assert r.history[0] == (0, None)
    history = dict(r.history)
    for f in frames:
        assert history[f.evals] == f.best_energy


@pytest.mark.parametrize("name", NAMES)
def test_result_to_dict_is_json_serializable(name):
    r = ALGORITHMS[name].run(S1_1, 2, budget=np.int64(300), seed=np.int64(4))
    d = r.to_dict()
    text = json.dumps(d)
    assert json.loads(text)["evaluations"] == 300 and d["seed"] == 4
    assert d["best_moves"] == r.best_moves and d["best_energy"] == r.best_energy
    assert d["history"][0] == [0, None]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bad", [{"seed": -1}, {"seed": 1.5}, {"target": 1.5}, {"target": "x"}])
def test_invalid_seed_or_target_rejected(name, bad):
    with pytest.raises(ValueError):
        ALGORITHMS[name].run(S1_1, 2, budget=100, **bad)


def test_result_params_are_a_copy():
    r = ALGORITHMS["GA"].run(S1_1, 2, budget=200)
    r.params["pop_size"] = -5
    assert ALGORITHMS["GA"].run(S1_1, 2, budget=200).params["pop_size"] != -5
