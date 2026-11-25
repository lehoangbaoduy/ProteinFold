"""LaTeX (booktabs) tables generated from experiment CSVs (PRD FR-A7; spec PFOLD-DOM-008).

No number in the paper is typed by hand: `make tables` regenerates every table.

* summary.tex — one row per sequence (ordered by length): n, E*, and per variant
  the mean ± sample std of the final best energy (best mean in bold) and the
  success rate. Runs without a valid fold are never averaged as 0: they are
  excluded from the mean and reported as a superscript "k NA".
* stats.tex — per sequence and pair: Vargha-Delaney A12 and the Holm-adjusted
  p-value over the whole table (`p_holm_global`, decision D23); p < 0.05 marked *.

Usage:  python -m hpfold_tools.tables results/main_2d.csv [--out paper/tables/main_2d]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from hpfold_tools.analysis import load_runs, pairwise, summarize
from hpfold_tools.experiments import ROOT

SIGNIFICANCE = 0.05
P_FLOOR = 0.001
DECIMALS = 2
LATEX_SPECIALS = {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#", "$": r"\$"}


def escape(text: str) -> str:
    return "".join(LATEX_SPECIALS.get(ch, ch) for ch in str(text))


def format_p(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p < P_FLOOR:
        return "$<0.001$"
    mark = "^{*}" if p < SIGNIFICANCE else ""
    return f"${p:.3f}{mark}$"


def _ordered(df: pd.DataFrame, variants: list[str] | None = None) -> tuple[list[str], list[str]]:
    sequences = df.drop_duplicates("seq_id").sort_values(["n", "seq_id"])["seq_id"].tolist()
    present = list(dict.fromkeys(df["variant"]))
    if variants is None:
        return sequences, present
    unknown = [v for v in variants if v not in present]
    if unknown:
        raise ValueError(f"unknown variant(s) {unknown}; present: {present}")
    return sequences, list(variants)


def _mean_cell(stats: pd.Series, bold: bool) -> str:
    if stats["found"] == 0:
        cell = "--"
    elif math.isnan(stats["std"]):
        cell = f"${stats['mean']:.2f}$"
    else:
        cell = f"${stats['mean']:.2f} \\pm {stats['std']:.2f}$"
    if bold and stats["found"]:
        cell = f"{{\\boldmath{cell}}}"  # \\textbf does not embolden math
    failed = int(stats["runs"] - stats["found"])
    return cell + (f"\\textsuperscript{{{failed} NA}}" if failed else "")


def summary_table(df: pd.DataFrame, variants: list[str] | None = None) -> str:
    """Summary table; pass `variants` to keep it narrow (one column pair per variant)."""
    table = summarize(df)
    sequences, variants = _ordered(df, variants)
    spec = "lrr" + "rr" * len(variants)
    names = " & ".join(f"\\multicolumn{{2}}{{c}}{{{escape(v)}}}" for v in variants)
    rules = " ".join(f"\\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(variants)))
    sub = " & ".join("Mean $\\pm$ SD & Succ." for _ in variants)  # Succ. = successful runs / runs
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             f"Seq & $n$ & $E^*$ & {names} \\\\", rules, f" & & & {sub} \\\\", "\\midrule"]
    for seq_id in sequences:
        part = df.loc[df["seq_id"] == seq_id].iloc[0]
        rows = {v: table.loc[(v, seq_id)] for v in variants if (v, seq_id) in table.index}
        means = [round(r["mean"], DECIMALS) for r in rows.values() if r["found"]]
        best = min(means) if means else None  # compare printed values, so ties are all bold
        cells = []
        for v in variants:
            stats = rows.get(v)
            if stats is None:
                cells += ["--", "--"]
                continue
            is_best = best is not None and stats["found"] and round(stats["mean"], DECIMALS) == best
            cells.append(_mean_cell(stats, is_best))
            successes = round(stats["success_rate"] * stats["runs"])
            cells.append(f"{successes}/{int(stats['runs'])}")  # exact count: never rounds 1/300 to 0%
        lines.append(f"{escape(seq_id)} & {int(part['n'])} & ${int(part['best_known'])}$ & "
                     + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def stats_table(df: pd.DataFrame) -> str:
    tests = pairwise(df)
    sequences, _ = _ordered(df)
    lines = ["\\begin{tabular}{llrr}", "\\toprule",
             "Seq & Pair & $A_{12}$ & $p_{\\mathrm{Holm}}$ \\\\", "\\midrule"]
    for seq_id in sequences:
        if tests.empty or seq_id not in tests.index.get_level_values(0):
            continue
        for (a, b), row in tests.loc[seq_id].iterrows():
            lines.append(f"{escape(seq_id)} & {escape(a)} vs {escape(b)} & {row['a12']:.2f} & "
                         f"{format_p(row['p_holm_global'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write LaTeX tables for one experiment CSV.")
    parser.add_argument("csv", help="results/<experiment>.csv")
    parser.add_argument("--out", help="output directory (default: paper/tables/<experiment>)")
    parser.add_argument("--variants", help="comma-separated variant labels for the summary table")
    args = parser.parse_args(argv)
    selected = args.variants.split(",") if args.variants else None
    csv_path = Path(args.csv)
    out = Path(args.out) if args.out else ROOT / "paper" / "tables" / csv_path.stem
    out.mkdir(parents=True, exist_ok=True)
    df = load_runs(csv_path)
    (out / "summary.tex").write_text(summary_table(df, selected), encoding="utf-8")
    (out / "stats.tex").write_text(stats_table(df), encoding="utf-8")
    print(f"tables written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
