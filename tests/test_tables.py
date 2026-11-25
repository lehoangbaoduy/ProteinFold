"""LaTeX table tests (PRD FR-A7; plan T4.3; spec PFOLD-DOM-008)."""

import csv

import pytest

from hpfold_tools import analysis as an
from hpfold_tools import tables as tb
from hpfold_tools.experiments import COLUMNS


def _row(variant, seed, energy, seq_id="S1-1", n=20, best_known=-9, evals_to_best=100):
    return {
        "experiment": "t", "variant": variant, "algorithm": variant, "dim": 2, "seq_id": seq_id, "n": n,
        "seed": seed, "budget": 1000, "best_known": best_known,
        "best_energy": "" if energy is None else energy, "best_moves": "" if energy is None else "F" * (n - 2),
        "evaluations": 1000, "evals_to_best": "" if energy is None else evals_to_best,
        "target_reached": energy is not None and energy <= best_known, "runtime_s": 1.0, "params_hash": "x",
    }


@pytest.fixture
def runs(tmp_path):
    rows = ([_row("GA", s, e) for s, e in enumerate([-9, -8])]
            + [_row("ACO", s, e) for s, e in enumerate([-9, -9])]
            + [_row("PSO", s, e) for s, e in enumerate([-7, None])]
            + [_row("GA", s, -10, seq_id="S1-4", n=36, best_known=-14) for s in range(2)]
            + [_row("ACO", s, -12, seq_id="S1-4", n=36, best_known=-14) for s in range(2)]
            + [_row("PSO", s, -11, seq_id="S1-4", n=36, best_known=-14) for s in range(2)])
    path = tmp_path / "runs.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return an.load_runs(path), path


def test_summary_table_structure_and_order(runs):
    df, _ = runs
    tex = tb.summary_table(df)
    assert "\\toprule" in tex and "\\midrule" in tex and "\\bottomrule" in tex
    assert tex.index("GA") < tex.index("ACO") < tex.index("PSO")  # variant order of first appearance
    assert tex.index("S1-1") < tex.index("S1-4")  # ordered by n
    s11 = next(line for line in tex.splitlines() if line.startswith("S1-1"))
    assert "& 20 &" in s11 and "& $-9$ &" in s11  # n and E*


def test_summary_cells(runs):
    df, _ = runs
    s11 = next(line for line in tb.summary_table(df).splitlines() if line.startswith("S1-1"))
    assert "$-8.50 \\pm 0.71$" in s11  # GA mean +- sample std
    assert "{\\boldmath$-9.00 \\pm 0.00$}" in s11  # ACO has the best mean -> bold math
    assert "& 1/2 &" in s11 and "& 2/2 &" in s11 and s11.rstrip(" \\").endswith("0/2")  # exact success counts
    assert "$-7.00$\\textsuperscript{1 NA}" in s11  # PSO: one valid run, one NA (never 0)


def test_stats_table_uses_global_holm_and_formats_p(runs):
    df, _ = runs
    tex = tb.stats_table(df)
    assert "\\toprule" in tex and "A_{12}" in tex
    tests = an.pairwise(df)
    row = tests.loc[("S1-4", "GA", "ACO")]
    line = next(x for x in tex.splitlines() if x.startswith("S1-4") and "GA vs ACO" in x)
    assert f"{row['a12']:.2f}" in line
    assert tb.format_p(row["p_holm_global"]) in line


def test_p_formatting_and_significance_mark():
    assert tb.format_p(0.0004) == "$<0.001$"
    assert tb.format_p(0.0123) == "$0.012^{*}$"
    assert tb.format_p(0.2) == "$0.200$"


def test_latex_escaping():
    assert tb.escape("GA_tuned & 5% #1") == "GA\\_tuned \\& 5\\% \\#1"


def test_cli_writes_both_tables(runs, tmp_path):
    _, path = runs
    out = tmp_path / "tex"
    assert tb.main([str(path), "--out", str(out)]) == 0
    assert "\\begin{tabular}" in (out / "summary.tex").read_text()
    assert "\\begin{tabular}" in (out / "stats.tex").read_text()


# ---------------------------------------------------------------- review round 1 ----

def _columns(spec_line):
    return len(spec_line.split("{")[2].rstrip("}"))


def _check_column_counts(tex):
    lines = tex.splitlines()
    width = _columns(lines[0])
    for line in lines:
        if line.endswith("\\\\") and "multicolumn" not in line:
            assert line.count("&") == width - 1, line


def test_every_row_has_as_many_cells_as_the_column_spec(runs):
    df, _ = runs
    _check_column_counts(tb.summary_table(df))
    _check_column_counts(tb.summary_table(df, variants=["GA", "PSO"]))
    _check_column_counts(tb.stats_table(df))


def test_variant_selection_keeps_tables_narrow(runs):
    df, _ = runs
    tex = tb.summary_table(df, variants=["ACO", "GA"])
    assert "PSO" not in tex and tex.index("ACO") < tex.index("GA")
    assert tex.splitlines()[0] == "\\begin{tabular}{lrrrrrr}"  # 3 + 2 x 2 columns


def test_unknown_variant_is_rejected(runs):
    df, _ = runs
    with pytest.raises(ValueError, match="variant"):
        tb.summary_table(df, variants=["SA"])


def test_bold_follows_the_printed_values(tmp_path):
    rows = [_row("GA", 0, -9), _row("GA", 1, -9), _row("ACO", 0, -9), _row("ACO", 1, -9)]
    path = tmp_path / "tie.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    s11 = next(x for x in tb.summary_table(an.load_runs(path)).splitlines() if x.startswith("S1-1"))
    assert s11.count("\\boldmath") == 2  # tied printed means are both bold


def test_all_na_variant_and_missing_variant(tmp_path):
    rows = ([_row("GA", s, -9) for s in range(2)] + [_row("PSO", s, None) for s in range(2)]
            + [_row("GA", s, -10, seq_id="S1-4", n=36, best_known=-14) for s in range(2)])
    path = tmp_path / "na.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tex = tb.summary_table(an.load_runs(path))
    s11 = next(x for x in tex.splitlines() if x.startswith("S1-1"))
    s14 = next(x for x in tex.splitlines() if x.startswith("S1-4"))
    assert "--\\textsuperscript{2 NA} & 0/2" in s11  # all NA: no mean, never 0
    assert s14.rstrip(" \\").endswith("-- & --")  # PSO never ran on S1-4
    _check_column_counts(tex)


def test_cli_variants_flag(runs, tmp_path):
    _, path = runs
    out = tmp_path / "sel"
    assert tb.main([str(path), "--out", str(out), "--variants", "GA,ACO"]) == 0
    assert "PSO" not in (out / "summary.tex").read_text()
