"""Web data export tests (PRD FR-B5, FR-E6, FR-T1, FR-T2; plan T3.6; spec PFOLD-DATA-003)."""

import json

import pytest

from hpfold_tools import experiments as ex
from hpfold_tools import export_web as xw

CONFIG = """
name = "web"
dim = 2
instances = ["S1-1"]
seeds = 3
budget_c = 0
budget_min = 600
stop_at_best_known = true

[[variants]]
label = "GA"
algorithm = "GA"
params = { strategy = "repair", pop_size = 20 }

[[variants]]
label = "PSO"
algorithm = "PSO"
params = { strategy = "repair", swarm_size = 10 }
"""


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("web")
    cfg = root / "web.toml"
    cfg.write_text(CONFIG, encoding="utf-8")
    ex.run_experiment(ex.load_config(cfg), root / "results", workers=1)
    return cfg, root / "results" / "web.csv"


def test_defaults_carry_tuned_params_effort_presets_and_limits(experiment, tmp_path):
    cfg, _ = experiment
    data = xw.write_defaults(cfg, tmp_path)
    saved = json.loads((tmp_path / "defaults.json").read_text())
    assert saved == data
    assert data["algorithms"]["GA"]["pop_size"] == 20 and data["algorithms"]["GA"]["strategy"] == "repair"
    assert data["algorithms"]["PSO"]["swarm_size"] == 10
    assert [e["c"] for e in data["effort"]] == [2, 5, 12] and all(e["min"] == 5000 for e in data["effort"])
    assert data["limits"] == {"min_length": 3, "max_length": 100}


def test_benchmarks_export(tmp_path):
    data = xw.write_benchmarks(tmp_path)
    ids = [b["id"] for b in data["instances"]]
    assert "S1-1" in ids and "S2-10" in ids
    first = data["instances"][0]
    assert {"id", "dim", "length", "sequence", "best_known_energy", "sources"} <= set(first)
    assert all(s["citation"] and s["url"].startswith("https://") for s in first["sources"])


def test_summary_export_is_json_with_nulls(experiment, tmp_path):
    _, csv_path = experiment
    data = xw.write_summary([csv_path], tmp_path)
    text = (tmp_path / "summary.json").read_text()
    assert "NaN" not in text
    web = data["experiments"]["web"]
    assert {r["variant"] for r in web["summary"]} == {"GA", "PSO"}
    assert all({"seq_id", "a", "b", "a12", "p_holm_global"} <= set(r) for r in web["pairwise"])


def test_select_best_and_median_runs(experiment):
    _, csv_path = experiment
    from hpfold_tools.analysis import load_runs

    df = load_runs(csv_path)
    picks = xw.select_runs(df)
    kinds = {(p["variant"], p["kind"]) for p in picks}
    assert kinds == {("GA", "best"), ("GA", "median"), ("PSO", "best"), ("PSO", "median")}
    for p in picks:
        runs = df[(df.variant == p["variant"]) & (df.seq_id == p["seq_id"])]
        if p["kind"] == "best":
            assert p["best_energy"] == runs.best_energy.min()


def test_thin_frames_keeps_improvements_and_last():
    frames = [{"evals": i * 10, "best_energy": e} for i, e in enumerate([-1, -1, -2, -2, -2, -3, -3, -3, -3, -3])]
    thin = xw.thin_frames(frames, every=4, max_frames=100)
    energies = [f["best_energy"] for f in thin]
    assert thin[0] == frames[0] and thin[-1] == frames[-1]
    assert -2 in energies and -3 in energies
    assert len(thin) < len(frames)
    capped = xw.thin_frames([{"evals": i, "best_energy": -i} for i in range(1000)], every=1, max_frames=50)
    assert len(capped) <= 50 and capped[-1]["evals"] == 999


def test_traces_reproduce_csv_and_stay_small(experiment, tmp_path):
    cfg, csv_path = experiment
    index = xw.export_traces(csv_path, cfg, tmp_path)
    assert len(index["traces"]) == 4
    for entry in index["traces"]:
        path = tmp_path / entry["file"]
        assert path.stat().st_size < xw.MAX_TRACE_BYTES
        trace = json.loads(path.read_text())
        assert trace["meta"]["kind"] in ("best", "median")
        assert trace["frames"] and trace["frames"][-1]["best_energy"] == trace["result"]["best_energy"]
        assert trace["result"]["best_energy"] == entry["best_energy"]
    saved = json.loads((tmp_path / "traces" / "web" / "index.json").read_text())
    assert saved == index


def test_trace_reproduction_mismatch_is_an_error(experiment, tmp_path):
    cfg, csv_path = experiment
    lines = csv_path.read_text().splitlines()
    col = lines[0].split(",").index("best_moves")
    tampered_lines = [lines[0]]
    for line in lines[1:]:
        fields = line.split(",")
        fields[col] = "F" * len(fields[col])  # not what the algorithm produced
        tampered_lines.append(",".join(fields))
    tampered = tmp_path / "web.csv"
    tampered.write_text("\n".join(tampered_lines) + "\n")
    with pytest.raises(ValueError, match="reproduce"):
        xw.export_traces(tampered, cfg, tmp_path / "out")


def test_cli(experiment, tmp_path):
    cfg, csv_path = experiment
    out = tmp_path / "data"
    assert xw.main(["--config", str(cfg), "--csv", str(csv_path), "--out", str(out)]) == 0
    for name in ("defaults.json", "benchmarks.json", "summary.json"):
        assert (out / name).is_file()
    assert (out / "traces" / "web" / "index.json").is_file()


# ---------------------------------------------------------------- review round 1 ----

def _config(tmp_path, text, name="cfg.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_preflight_rejects_changed_params(experiment, tmp_path):
    _, csv_path = experiment
    changed = _config(tmp_path, CONFIG.replace("pop_size = 20", "pop_size = 30"))
    with pytest.raises(ValueError, match="params_hash"):
        xw.export_traces(csv_path, changed, tmp_path / "out")


def test_rerun_must_match_evaluations_not_just_energy(experiment, tmp_path):
    cfg, csv_path = experiment
    # Same energy and fold, but the CSV claims a different evaluation count: a trace
    # built from such a run would contradict the dashboard, so it must be refused.
    lines = csv_path.read_text().splitlines()
    col = lines[0].split(",").index("evaluations")
    tampered = [lines[0]]
    for line in lines[1:]:
        fields = line.split(",")
        fields[col] = str(int(fields[col]) - 1)
        tampered.append(",".join(fields))
    path = tmp_path / "web.csv"
    path.write_text("\n".join(tampered) + "\n")
    with pytest.raises(ValueError, match="reproduce"):
        xw.export_traces(path, cfg, tmp_path / "out")


def test_unknown_variant_in_csv_is_a_clear_error(experiment, tmp_path):
    _, csv_path = experiment
    only_ga = _config(tmp_path, CONFIG.split("[[variants]]\nlabel = \"PSO\"")[0])
    with pytest.raises(ValueError, match="PSO"):
        xw.export_traces(csv_path, only_ga, tmp_path / "out")


def test_thinning_keeps_the_last_improvement_on_a_long_plateau():
    frames = [{"evals": i, "best_energy": -min(i, 500)} for i in range(1000)]
    thin = xw.thin_frames(frames, every=1, max_frames=50)
    assert len(thin) <= 50
    assert any(f["evals"] == 500 for f in thin)  # the frame where -500 was first reached
    assert thin[-1]["evals"] == 999


def test_stale_trace_files_are_removed(experiment, tmp_path):
    cfg, csv_path = experiment
    stale = tmp_path / "traces" / "web" / "S1-1_GA_best_s999.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    xw.export_traces(csv_path, cfg, tmp_path)
    assert not stale.exists()


def test_trace_meta_describes_the_pick(experiment, tmp_path):
    cfg, csv_path = experiment
    index = xw.export_traces(csv_path, cfg, tmp_path)
    for entry in index["traces"]:
        meta = json.loads((tmp_path / entry["file"]).read_text())["meta"]
        assert meta["runs"] == 3 and meta["rank"] == (1 if meta["kind"] == "best" else 2)
        assert meta["median_rule"] == "lower" and "best_energy" in meta and meta["n"] == 20


def test_oversized_trace_is_an_error(experiment, tmp_path, monkeypatch):
    cfg, csv_path = experiment
    monkeypatch.setattr(xw, "MAX_TRACE_BYTES", 10)
    with pytest.raises(ValueError, match="size"):
        xw.export_traces(csv_path, cfg, tmp_path)


def test_parallel_export_is_identical(experiment, tmp_path):
    cfg, csv_path = experiment
    one = xw.export_traces(csv_path, cfg, tmp_path / "a", workers=1)
    two = xw.export_traces(csv_path, cfg, tmp_path / "b", workers=2)
    assert one == two
    for entry in one["traces"]:
        assert (tmp_path / "a" / entry["file"]).read_text() == (tmp_path / "b" / entry["file"]).read_text()


def test_benchmark_notes_and_experiments_index(experiment, tmp_path):
    cfg, _ = experiment
    notes = {b["id"]: b["notes"] for b in xw.write_benchmarks(tmp_path)["instances"]}
    assert "51" in notes["S1-6"]  # recorded source discrepancy is shown on the page
    index = xw.write_experiments([cfg], tmp_path)
    entry = index["experiments"][0]
    assert entry["name"] == "web" and entry["dim"] == 2 and entry["trace_index"] == "traces/web/index.json"
    assert entry["variants"]["GA"]["params"]["pop_size"] == 20 and entry["variants"]["GA"]["params_hash"]
    assert json.loads((tmp_path / "experiments.json").read_text()) == index
