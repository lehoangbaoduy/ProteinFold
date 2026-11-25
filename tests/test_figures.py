"""Paper figure tests (PRD FR-A3..A5; plan T4.2; spec PFOLD-DOM-007)."""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from hpfold_tools import experiments as ex
from hpfold_tools import figures as fg

CONFIG = """
name = "fig"
dim = 2
instances = ["S1-1", "S1-2"]
seeds = 3
budget_c = 0
budget_min = 400
stop_at_best_known = false

[[variants]]
label = "GA"
algorithm = "GA"
params = { pop_size = 20 }

[[variants]]
label = "PSO"
algorithm = "PSO"
"""


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("fig")
    cfg_path = root / "fig.toml"
    cfg_path.write_text(CONFIG, encoding="utf-8")
    ex.run_experiment(ex.load_config(cfg_path), root / "results", workers=1)
    return root / "results" / "fig.csv"


def test_step_values_hold_until_next_point():
    points = [[0, None], [10, -3], [25, -5]]
    assert fg.step_values(points, [0, 5, 10, 24, 25, 100]) == [None, None, -3, -3, -5, -5]


def test_convergence_bands_hand_computed():
    histories = [
        [[0, None], [10, -2], [20, -4]],
        [[0, None], [5, -1], [30, -6]],
        [[0, None], [15, -3]],
    ]
    bands = fg.convergence_bands(histories, [0, 10, 20, 30])
    assert bands["count"] == [0, 2, 3, 3]
    assert math.isnan(bands["median"][0])
    assert bands["median"][1:] == pytest.approx([-1.5, -3.0, -4.0])
    assert bands["q25"][1] == pytest.approx(-1.75) and bands["q75"][1] == pytest.approx(-1.25)
    assert bands["q25"][2] == pytest.approx(-3.5) and bands["q75"][2] == pytest.approx(-2.0)


def test_convergence_figure_structure(experiment, tmp_path):
    fig = fg.convergence_figure(experiment, experiment.parent / "fig.history.jsonl")
    panels = [ax for ax in fig.axes if ax.get_title()]
    assert sorted(ax.get_title().split()[0] for ax in panels) == ["S1-1", "S1-2"]
    for ax in panels:
        labels = {ln.get_label() for ln in ax.get_lines()}
        assert {"GA", "PSO"} <= labels
        assert any(ln.get_gid() == "best_known" for ln in ax.get_lines())
    out = tmp_path / "conv.png"
    fig.savefig(out)
    assert out.stat().st_size > 1000


def test_box_figure_reports_na_instead_of_plotting_zero(experiment, tmp_path):
    import csv

    rows = list(csv.DictReader(experiment.open(encoding="utf-8")))
    rows[0]["best_energy"] = rows[0]["best_moves"] = rows[0]["evals_to_best"] = ""
    na_csv = tmp_path / "na.csv"
    with na_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig = fg.box_figure(na_csv)
    panels = [ax for ax in fig.axes if ax.get_title()]
    assert len(panels) == 2
    texts = " ".join(t.get_text() for ax in panels for t in ax.texts)
    assert "NA" in texts  # the failed run is reported, not drawn as energy 0
    for ax in panels:
        assert all(y < 0 for line in ax.get_lines() if line.get_gid() == "best_known" for y in line.get_ydata())


def test_best_folds_figure_has_one_panel_per_variant_and_sequence(experiment):
    fig = fg.best_folds_figure(experiment)
    assert len([ax for ax in fig.axes if ax.get_title()]) == 4


def test_cli_writes_png_and_pdf(experiment, tmp_path):
    out = tmp_path / "figs"
    assert fg.main([str(experiment), "--out", str(out)]) == 0
    for name in ("convergence", "boxes", "best_folds"):
        for suffix in (".png", ".pdf"):
            assert (out / f"{name}{suffix}").stat().st_size > 1000


def test_convergence_x_ticks_are_compact(experiment):
    fig = fg.convergence_figure(experiment, experiment.parent / "fig.history.jsonl")
    fig.canvas.draw()
    for ax in (a for a in fig.axes if a.get_title()):
        labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        assert 2 <= len(labels) <= 6
        assert all(label == "0" or label.endswith("k") for label in labels)


# ---------------------------------------------------------------- review round 1 ----

def _rewrite(experiment, tmp_path, mutate, name="mut.csv"):
    import csv

    rows = list(csv.DictReader(experiment.open(encoding="utf-8")))
    rows = mutate(rows)
    out = tmp_path / name
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return out


def _na(row):
    row["best_energy"] = row["best_moves"] = row["evals_to_best"] = ""
    return row


def test_box_figure_survives_an_all_na_sequence_and_counts_exactly(experiment, tmp_path):
    path = _rewrite(experiment, tmp_path, lambda rows: [_na(r) if r["seq_id"] == "S1-2" else r for r in rows])
    fig = fg.box_figure(path)
    panel = next(ax for ax in fig.axes if ax.get_title().startswith("S1-2"))
    labels = sorted(t.get_text() for t in panel.texts)
    assert labels == ["NA: 3", "NA: 3"]


def test_step_value_is_held_after_an_early_stop():
    stopped = [[0, None], [40, -9]]  # run reached E* at 40 evaluations and stopped
    assert fg.step_values(stopped, [39, 40, 400, 10_000]) == [None, -9, -9, -9]


def test_median_is_drawn_only_where_every_run_has_a_valid_fold(tmp_path):
    histories = [[[0, None], [10, -2]], [[0, None], [50, -3]]]
    grid = [0, 10, 20, 50, 60]
    masked = fg.masked_bands(histories, grid)
    assert [math.isnan(v) for v in masked["median"]] == [True, True, True, False, False]
    assert masked["median"][3] == pytest.approx(-2.5)


def test_best_folds_in_3d_and_no_valid_fold_panel(tmp_path):
    cfg = tmp_path / "d3.toml"
    cfg.write_text(CONFIG.replace('name = "fig"', 'name = "d3"').replace("dim = 2", "dim = 3")
                   .replace('instances = ["S1-1", "S1-2"]', 'instances = ["S2-1"]').replace("seeds = 3", "seeds = 1"),
                   encoding="utf-8")
    ex.run_experiment(ex.load_config(cfg), tmp_path / "r", workers=1)
    path = _rewrite(tmp_path / "r" / "d3.csv", tmp_path, lambda rows: [_na(r) if r["variant"] == "PSO" else r for r in rows])
    fig = fg.best_folds_figure(path)
    titled = [ax for ax in fig.axes if ax.get_title()]
    assert any(ax.name == "3d" for ax in titled)
    assert any("no valid fold" in ax.get_title() for ax in titled)


def test_mixed_dimensions_are_rejected(experiment, tmp_path):
    def mix(rows):
        rows[0]["dim"] = "3"
        return rows

    with pytest.raises(ValueError, match="dim"):
        fg.best_folds_figure(_rewrite(experiment, tmp_path, mix))


def test_convergence_legend_does_not_cover_axis_labels(experiment):
    fig = fg.convergence_figure(experiment, experiment.parent / "fig.history.jsonl")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = fig.legends[0].get_window_extent(renderer)
    for ax in (a for a in fig.axes if a.get_title()):
        assert not legend_box.overlaps(ax.xaxis.label.get_window_extent(renderer))


def test_legend_clear_of_labels_with_many_variants():
    csv_path = ex.ROOT / "results" / "penalty_repair.csv"  # committed: 8 variants -> 2 legend rows
    fig = fg.convergence_figure(csv_path)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = fig.legends[0].get_window_extent(renderer)
    for ax in (a for a in fig.axes if a.get_title()):
        assert not legend_box.overlaps(ax.xaxis.label.get_window_extent(renderer))
