"""Statistics tests with hand-computed values (PRD FR-A1, FR-A2; plan T4.1; edge cases E7, E11)."""

import csv
import math

import pytest

from hpfold_tools import analysis as an
from hpfold_tools.experiments import COLUMNS


def _row(variant, seed, energy, evals_to_best=None, seq_id="S1-1", best_known=-9, runtime=1.0):
    return {
        "experiment": "t", "variant": variant, "algorithm": variant.split("-")[0], "dim": 2,
        "seq_id": seq_id, "n": 20, "seed": seed, "budget": 1000, "best_known": best_known,
        "best_energy": "" if energy is None else energy, "best_moves": "" if energy is None else "F" * 18,
        "evaluations": 1000, "evals_to_best": "" if evals_to_best is None else evals_to_best,
        "target_reached": energy is not None and energy <= best_known, "runtime_s": runtime,
        "params_hash": "x",
    }


def _csv(tmp_path, rows):
    path = tmp_path / "runs.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------- A12 + Holm ----

def test_a12_lower_is_better():
    assert an.a12([1, 2], [3, 4]) == 1.0  # A always lower (better)
    assert an.a12([3, 4], [1, 2]) == 0.0
    assert an.a12([5, 5], [5, 5]) == 0.5
    # pairs: (1,2) win, (1,2) win, (3,2) loss, (3,2) loss -> 2/4
    assert an.a12([1, 3], [2, 2]) == 0.5
    # pairs: (-9,-8) win, (-9,-9) tie, (-8,-8) tie, (-8,-9) loss -> (1 + 0.5*2) / 4
    assert an.a12([-9, -8], [-8, -9]) == pytest.approx(0.5)
    assert an.a12([-9, -9, -8], [-8, -8, -8]) == pytest.approx((6 + 0.5 * 3) / 9)


def test_a12_treats_na_as_worst():
    assert an.a12([-5, None], [-3, -3]) == pytest.approx(0.5)  # -5 wins twice, NA loses twice
    assert an.a12([None], [None]) == 0.5


def test_holm_adjustment_hand_computed():
    # sorted: 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> monotone: 0.03, 0.06, 0.06
    assert an.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert an.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])  # capped at 1
    assert an.holm([]) == []


# ---------------------------------------------------------------- summary ----

def test_summary_counts_na_as_failure_and_excludes_it_from_energy_stats(tmp_path):
    rows = [_row("GA", 0, -9, 100), _row("GA", 1, -7, 500), _row("GA", 2, None),
            _row("GA", 3, -9, 300)]
    table = an.summarize(an.load_runs(_csv(tmp_path, rows)))
    r = table.loc[("GA", "S1-1")]
    assert r["runs"] == 4 and r["found"] == 3
    assert r["best"] == -9 and r["worst"] == -7
    assert r["mean"] == pytest.approx((-9 - 7 - 9) / 3)
    assert r["median"] == -9
    assert r["std"] == pytest.approx(math.sqrt(((-9 + 25 / 3) ** 2 * 2 + (-7 + 25 / 3) ** 2) / 2))
    assert r["success_rate"] == pytest.approx(2 / 4)  # NA is a failure, not a success or a 0
    assert r["evals_to_best_success"] == pytest.approx((100 + 300) / 2)  # successes only


def test_success_uses_best_known_inclusive(tmp_path):
    rows = [_row("PSO", 0, -10, 50), _row("PSO", 1, -9, 60), _row("PSO", 2, -8, 70)]
    table = an.summarize(an.load_runs(_csv(tmp_path, rows)))
    assert table.loc[("PSO", "S1-1")]["success_rate"] == pytest.approx(2 / 3)


def test_summary_all_na_has_no_energy_stats(tmp_path):
    rows = [_row("ACO", 0, None), _row("ACO", 1, None)]
    r = an.summarize(an.load_runs(_csv(tmp_path, rows))).loc[("ACO", "S1-1")]
    assert r["found"] == 0 and r["success_rate"] == 0.0
    assert math.isnan(r["mean"]) and math.isnan(r["evals_to_best_success"])


def test_load_runs_types(tmp_path):
    df = an.load_runs(_csv(tmp_path, [_row("GA", 0, -9, 10), _row("GA", 1, None)]))
    assert df["best_energy"].isna().sum() == 1
    assert df["target_reached"].dtype == bool and df["seed"].dtype.kind == "i"


# ---------------------------------------------------------------- pairwise ----

def test_pairwise_detects_a_clear_difference_and_applies_holm(tmp_path):
    rows = ([_row("GA", s, -9, 10) for s in range(15)]
            + [_row("PSO", s, -5, 10) for s in range(15)]
            + [_row("ACO", s, -9 if s % 2 else -8, 10) for s in range(15)])
    tests = an.pairwise(an.load_runs(_csv(tmp_path, rows)))
    ga_pso = tests.loc[("S1-1", "GA", "PSO")]
    assert ga_pso["p_raw"] < 1e-4 and ga_pso["a12"] == 1.0
    assert ga_pso["p_holm"] >= ga_pso["p_raw"]
    assert len(tests) == 3  # 3 pairs for 3 variants
    assert set(tests.index.get_level_values(1)) | set(tests.index.get_level_values(2)) == {"GA", "PSO", "ACO"}


def test_pairwise_identical_samples_are_not_significant(tmp_path):
    rows = [_row("GA", s, -9 + s % 3) for s in range(10)] + [_row("PSO", s, -9 + s % 3) for s in range(10)]
    t = an.pairwise(an.load_runs(_csv(tmp_path, rows))).loc[("S1-1", "GA", "PSO")]
    assert t["p_raw"] > 0.5 and t["a12"] == pytest.approx(0.5)


def test_pairwise_is_per_sequence(tmp_path):
    rows = ([_row("GA", s, -9, seq_id="S1-1") for s in range(5)]
            + [_row("PSO", s, -9, seq_id="S1-1") for s in range(5)]
            + [_row("GA", s, -20, seq_id="S1-4", best_known=-14) for s in range(5)]
            + [_row("PSO", s, -10, seq_id="S1-4", best_known=-14) for s in range(5)])
    tests = an.pairwise(an.load_runs(_csv(tmp_path, rows)))
    assert {k[0] for k in tests.index} == {"S1-1", "S1-4"}
    assert tests.loc[("S1-4", "GA", "PSO")]["a12"] == 1.0


# ---------------------------------------------------------------- review round 1 ----

def _exact_two_sided_p(a, b):
    """Independent exact permutation p-value of the rank-sum statistic (two-sided, 2*min tail)."""
    import itertools

    from scipy.stats import rankdata

    pooled = list(a) + list(b)
    ranks = rankdata(pooled)
    n = len(a)
    observed = ranks[:n].sum()
    sums = [ranks[list(idx)].sum() for idx in itertools.combinations(range(len(pooled)), n)]
    less = sum(s <= observed + 1e-9 for s in sums) / len(sums)
    greater = sum(s >= observed - 1e-9 for s in sums) / len(sums)
    return min(1.0, 2 * min(less, greater))


@pytest.mark.parametrize(
    "a,b",
    [
        ([-9] * 5, [-9, -9, -8, -8, -8]),  # heavy ties: asymptotic p would be too small
        ([-9, -9, -9, -8, -7], [-8, -8, -7, -7, -6]),
        ([-9, -8, -7], [-6, -5, -4, -3]),  # unequal sizes
    ],
)
def test_p_values_match_exact_permutation(tmp_path, a, b):
    rows = [_row("GA", s, e) for s, e in enumerate(a)] + [_row("PSO", s, e) for s, e in enumerate(b)]
    t = an.pairwise(an.load_runs(_csv(tmp_path, rows))).loc[("S1-1", "GA", "PSO")]
    assert t["p_raw"] == pytest.approx(_exact_two_sided_p(a, b), abs=1e-9)
    assert t["n_a"] == len(a) and t["n_b"] == len(b)


def test_p_holm_is_holm_of_p_raw_per_sequence_and_global(tmp_path):
    rows = []
    for seq_id, shift in (("S1-1", 0), ("S1-4", 1)):
        rows += [_row("GA", s, -9 + (s % 2), seq_id=seq_id) for s in range(8)]
        rows += [_row("PSO", s, -8 + shift + (s % 3 == 0), seq_id=seq_id) for s in range(8)]
        rows += [_row("ACO", s, -9 + shift, seq_id=seq_id) for s in range(8)]
    tests = an.pairwise(an.load_runs(_csv(tmp_path, rows)))
    for seq_id in ("S1-1", "S1-4"):
        part = tests.loc[seq_id]
        assert part["p_holm"].tolist() == pytest.approx(an.holm(part["p_raw"].tolist()))
    assert tests["p_holm_global"].tolist() == pytest.approx(an.holm(tests["p_raw"].tolist()))


def test_na_ranks_below_an_energy_of_zero():
    assert an.a12([0, 0], [None, None]) == 1.0
    assert an.a12([None], [0]) == 0.0


def test_runtime_single_run_and_unequal_counts(tmp_path):
    rows = [_row("GA", 0, -9, 5, runtime=2.0), _row("GA", 1, -8, runtime=4.0), _row("PSO", 0, -7, runtime=1.5)]
    table = an.summarize(an.load_runs(_csv(tmp_path, rows)))
    assert table.loc[("GA", "S1-1")]["runtime_mean"] == pytest.approx(3.0)
    assert math.isnan(table.loc[("PSO", "S1-1")]["std"])  # single run: ddof=1 std undefined
    assert table.loc[("PSO", "S1-1")]["runs"] == 1


def test_single_variant_gives_an_empty_but_well_formed_table(tmp_path):
    tests = an.pairwise(an.load_runs(_csv(tmp_path, [_row("GA", s, -9) for s in range(3)])))
    assert tests.empty
    assert {"p_raw", "p_holm", "p_holm_global", "a12", "n_a", "n_b"} <= set(tests.columns)


def test_target_reached_parsed_explicitly(tmp_path):
    df = an.load_runs(_csv(tmp_path, [_row("GA", 0, -9, 3), _row("GA", 1, -5)]))
    assert df["target_reached"].tolist() == [True, False]


def test_friedman_overall_ranking(tmp_path):
    rows = []
    for k, seq_id in enumerate(["S1-1", "S1-2", "S1-3", "S1-4", "S1-5"]):
        rows += [_row("GA", s, -20 - k, seq_id=seq_id) for s in range(5)]  # always best
        rows += [_row("ACO", s, -15 - k, seq_id=seq_id) for s in range(5)]
        rows += [_row("PSO", s, -10 - k, seq_id=seq_id) for s in range(5)]  # always worst
    result = an.friedman(an.load_runs(_csv(tmp_path, rows)))
    assert result.mean_ranks == {"GA": 1.0, "ACO": 2.0, "PSO": 3.0}
    assert result.p_value < 0.05 and result.n_blocks == 5
