"""Experiment runner tests (PRD FR-E1..E5; plan T3.1; spec PFOLD-DATA-002)."""

import csv
import itertools
import json

import pytest

from hpfold_tools import experiments as ex

COLUMNS = ["experiment", "variant", "algorithm", "dim", "seq_id", "n", "seed", "budget", "best_known",
           "best_energy", "best_moves", "evaluations", "evals_to_best", "target_reached", "runtime_s",
           "params_hash"]

BASE = """
name = "tiny"
dim = 2
instances = ["S1-1"]
seeds = 2
budget_c = 0
budget_min = 300
stop_at_best_known = true

[[variants]]
label = "GA-small"
algorithm = "GA"
params = { pop_size = 20 }

[[variants]]
label = "PSO"
algorithm = "PSO"
"""


def write(tmp_path, text=BASE, name="cfg.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_load_config(tmp_path):
    cfg = ex.load_config(write(tmp_path))
    assert cfg.name == "tiny" and cfg.dim == 2 and cfg.instances == ("S1-1",)
    assert [v.label for v in cfg.variants] == ["GA-small", "PSO"]
    assert cfg.variants[0].params == {"pop_size": 20}


@pytest.mark.parametrize(
    "change",
    [
        ('algorithm = "PSO"', 'algorithm = "SA"'),
        ('instances = ["S1-1"]', 'instances = ["S9-9"]'),
        ('instances = ["S1-1"]', 'instances = ["S2-1"]'),  # a 3D instance in a 2D experiment
        ('label = "PSO"', 'label = "GA-small"'),  # duplicate label
        ("params = { pop_size = 20 }", "params = { pop_size = 1 }"),  # rejected by GA Params
        ("params = { pop_size = 20 }", "params = { colour = 3 }"),  # unknown parameter
        ("seeds = 2", "seeds = 0"),
        ("budget_c = 0", "budget_c = -1"),
        ('name = "tiny"', 'name = "../escape"'),  # name is used in output file names
    ],
)
def test_invalid_configs_rejected(tmp_path, change):
    old, new = change
    assert old in BASE
    with pytest.raises(ValueError):
        ex.load_config(write(tmp_path, BASE.replace(old, new, 1)))


def test_budget_formula():
    assert ex.budget_for(20, budget_c=10, budget_min=10_000) == 10_000
    assert ex.budget_for(100, budget_c=10, budget_min=10_000) == 100_000


def test_run_writes_csv_history_and_meta(tmp_path):
    out = tmp_path / "results"
    summary = ex.run_experiment(ex.load_config(write(tmp_path)), out, workers=1)
    assert summary.new == 4 and summary.skipped == 0
    table = rows(out / "tiny.csv")
    assert list(table[0]) == COLUMNS and len(table) == 4
    for r in table:
        assert r["experiment"] == "tiny" and r["n"] == "20" and r["budget"] == "300"
        assert r["best_known"] == "-9" and int(r["best_energy"]) <= 0
        assert int(r["evaluations"]) <= 300 and r["target_reached"] in ("True", "False")
    history = [json.loads(line) for line in (out / "tiny.history.jsonl").read_text().splitlines()]
    assert len(history) == 4 and history[0]["history"][0] == [0, None]
    meta = json.loads((out / "tiny.meta.json").read_text())
    assert meta["git_commit"] and meta["config"]["name"] == "tiny"
    assert meta["variants"]["GA-small"]["params"]["pop_size"] == 20
    hashes = {r["variant"]: r["params_hash"] for r in table}
    assert hashes["GA-small"] == meta["variants"]["GA-small"]["params_hash"] != hashes["PSO"]


def test_resume_skips_completed_rows(tmp_path):
    out = tmp_path / "results"
    cfg = ex.load_config(write(tmp_path))
    ex.run_experiment(cfg, out, workers=1)
    again = ex.run_experiment(cfg, out, workers=1)
    assert again.new == 0 and again.skipped == 4
    assert len(rows(out / "tiny.csv")) == 4


def _strip_runtime(table):
    return sorted(({k: v for k, v in r.items() if k != "runtime_s"} for r in table),
                  key=lambda r: (r["variant"], r["seed"]))


def test_rerun_is_deterministic_and_worker_count_does_not_matter(tmp_path):
    cfg = ex.load_config(write(tmp_path))
    ex.run_experiment(cfg, tmp_path / "a", workers=1)
    ex.run_experiment(cfg, tmp_path / "b", workers=2)
    assert _strip_runtime(rows(tmp_path / "a" / "tiny.csv")) == _strip_runtime(rows(tmp_path / "b" / "tiny.csv"))


def test_na_rows_are_written_as_empty_not_zero(tmp_path):
    text = BASE.replace("budget_min = 300", "budget_min = 0")
    out = tmp_path / "results"
    ex.run_experiment(ex.load_config(write(tmp_path, text)), out, workers=1)
    for r in rows(out / "tiny.csv"):
        assert r["best_energy"] == "" and r["best_moves"] == "" and r["evals_to_best"] == ""
        assert r["evaluations"] == "0"


def test_seed_offset_and_instance_order(tmp_path):
    text = BASE.replace("seeds = 2", "seeds = 2\nseed_offset = 10").replace(
        'instances = ["S1-1"]', 'instances = ["S1-2", "S1-1"]')
    out = tmp_path / "results"
    ex.run_experiment(ex.load_config(write(tmp_path, text)), out, workers=1)
    table = rows(out / "tiny.csv")
    assert {r["seed"] for r in table} == {"10", "11"}
    assert {r["seq_id"] for r in table} == {"S1-1", "S1-2"} and len(table) == 8


def test_cli(tmp_path):
    out = tmp_path / "results"
    assert ex.main([str(write(tmp_path)), "--out", str(out), "--workers", "1"]) == 0
    assert len(rows(out / "tiny.csv")) == 4


# ---------------------------------------------------------------- review round 1 ----

def _run(tmp_path, text=BASE, out="results"):
    return ex.run_experiment(ex.load_config(write(tmp_path, text)), tmp_path / out, workers=1)


def _valid(table):
    return all(None not in r.values() and len(r) == len(COLUMNS) for r in table)


def test_truncated_last_line_is_repaired_and_rerun(tmp_path):
    _run(tmp_path)
    path = tmp_path / "results" / "tiny.csv"
    text = path.read_text(encoding="utf-8")
    path.write_text(text[: len(text) - 25], encoding="utf-8")  # crash mid-way through the last row
    again = _run(tmp_path)
    assert again.new == 1
    table = rows(path)
    assert len(table) == 4 and _valid(table)
    history = [json.loads(line) for line in (tmp_path / "results" / "tiny.history.jsonl").read_text().splitlines()]
    assert sorted((h["variant"], h["seq_id"], h["seed"]) for h in history) == sorted(
        (r["variant"], r["seq_id"], int(r["seed"])) for r in table)


def test_corrupt_row_in_the_middle_is_an_error(tmp_path):
    _run(tmp_path)
    path = tmp_path / "results" / "tiny.csv"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[2] = "tiny,GA-small,GA\n"
    path.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        _run(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        ("params = { pop_size = 20 }", "params = { pop_size = 30 }"),
        ("budget_min = 300", "budget_min = 400"),
        ("stop_at_best_known = true", "stop_at_best_known = false"),
    ],
)
def test_resume_with_changed_settings_is_refused(tmp_path, change):
    _run(tmp_path)
    with pytest.raises(ValueError, match="config changed"):
        _run(tmp_path, BASE.replace(*change))


def test_resume_can_extend_seeds(tmp_path):
    _run(tmp_path)
    more = _run(tmp_path, BASE.replace("seeds = 2", "seeds = 3"))
    assert more.new == 2 and more.skipped == 4
    path = tmp_path / "results" / "tiny.csv"
    assert path.read_text(encoding="utf-8").count("experiment,variant") == 1
    assert len(rows(path)) == 6 and _valid(rows(path))


def test_meta_keeps_first_start_and_records_sessions(tmp_path):
    _run(tmp_path)
    meta_path = tmp_path / "results" / "tiny.meta.json"
    first = json.loads(meta_path.read_text())
    _run(tmp_path, BASE.replace("seeds = 2", "seeds = 3"))
    meta = json.loads(meta_path.read_text())
    assert meta["started_at"] == first["started_at"]
    assert len(meta["sessions"]) == 2 and meta["config_hash"] == first["config_hash"]
    assert all(s["git_commit"] for s in meta["sessions"])


@pytest.mark.parametrize(
    "change",
    [
        ("seeds = 2", "seeds = 2\nseed_offet = 10"),  # typo in a top-level key
        ('label = "PSO"\nalgorithm = "PSO"', 'label = "PSO"\nalgorithm = "PSO"\nparam = { w = 0.5 }'),
        ('algorithm = "PSO"\n', ""),  # algorithm missing
        ("budget_c = 0\n", ""),  # budget keys are required, never silently 0
        ('name = "tiny"', 'name = "tiny\\n"'),  # newline in the name
        ("dim = 2", "dim = 4"),
        ("stop_at_best_known = true", 'stop_at_best_known = "yes"'),
        ("budget_c = 0", "budget_c = nan"),
    ],
)
def test_more_invalid_configs_rejected(tmp_path, change):
    old, new = change
    assert old in BASE
    with pytest.raises(ValueError):
        ex.load_config(write(tmp_path, BASE.replace(old, new, 1)))


def test_budget_rounds_up():
    assert ex.budget_for(3, budget_c=0.5, budget_min=0) == 5  # 4.5 -> 5, never down


def test_cli_rejects_non_positive_workers(tmp_path):
    with pytest.raises(SystemExit):
        ex.main([str(write(tmp_path)), "--out", str(tmp_path / "r"), "--workers", "0"])


@pytest.mark.parametrize("config", ["smoke.toml", "penalty_repair.toml"])
def test_repository_configs_are_valid(config):
    cfg = ex.load_config(ex.ROOT / "configs" / config)
    assert cfg.variants and cfg.instances


# ---------------------------------------------------------------- history thinning ----

def test_thin_history_keeps_change_points_first_and_last():
    full = [[0, None], [100, -5], [200, -5], [300, -6], [400, -6], [500, -6]]
    assert ex.thin_history(full) == [[0, None], [100, -5], [300, -6], [500, -6]]
    assert ex.thin_history([[0, None]]) == [[0, None]]
    assert ex.thin_history([[0, None], [7, None]]) == [[0, None], [7, None]]
    assert ex.thin_history([]) == []


def test_history_file_stores_only_change_points(tmp_path):
    out = tmp_path / "results"
    ex.run_experiment(ex.load_config(write(tmp_path)), out, workers=1)
    for line in (out / "tiny.history.jsonl").read_text().splitlines():
        points = json.loads(line)["history"]
        interior = [e for _, e in points[1:-1]]
        assert all(a != b for a, b in itertools.pairwise(interior))
        assert points == ex.thin_history(points)


def _step_value(points, evals):
    """Best-so-far at `evals` from a step-function history (value holds until the next point)."""
    import bisect

    index = bisect.bisect_right([p[0] for p in points], evals) - 1
    return points[index][1]


@pytest.mark.parametrize("algorithm", ["GA", "ACO", "PSO"])
@pytest.mark.parametrize("dim", [2, 3])
def test_thinning_is_lossless_for_step_lookups(algorithm, dim):
    from hpfold.algorithms import ALGORITHMS

    full = [list(p) for p in ALGORITHMS[algorithm].run("HPHPPHHPHPPHPHHPPHPH", dim, seed=3, budget=2500).history]
    thin = ex.thin_history(full)
    assert len(thin) <= len(full)
    for evals in range(full[-1][0] + 50):
        assert _step_value(full, evals) == _step_value(thin, evals)
