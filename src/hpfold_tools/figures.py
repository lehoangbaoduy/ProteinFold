"""Paper figures from experiment results (PRD FR-A3, FR-A4, FR-A5; spec PFOLD-DOM-007).

* Convergence: median best-so-far energy vs evaluations with an interquartile
  band, one panel per sequence, one line per variant, dashed E* line. Histories
  are step functions (decision D26): values are looked up, never interpolated.
* Boxes: final best energies per variant and sequence; runs that never found a
  valid fold (NA) are reported as text, never drawn as energy 0.
* Best folds: the best fold found by each variant on each sequence.

Usage:  python -m hpfold_tools.figures results/main_2d.csv [--out results/figures/main_2d]
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from hpfold_tools.analysis import load_runs
from hpfold_tools.experiments import load_benchmarks
from hpfold_tools.visualize import plot_fold

GRID_POINTS = 200
MAX_COLUMNS = 3
PANEL_SIZE = (4.4, 3.3)
FOLD_PANEL = 2.8
BAND_ALPHA = 0.12
LEGEND_COLUMNS = 6
NA_LABEL_Y = 0.02  # axes fraction: NA counts sit at the bottom, clear of data and E*
SAVE_DPI = 200
QUARTILES = (25, 75)
MAX_X_TICKS = 5
KILO = 1000
BEST_KNOWN_STYLE = {"linestyle": "--", "color": "black", "linewidth": 1.0}


# ------------------------------------------------------------------ data ----

def step_values(points: Sequence[Sequence[object]], grid: Sequence[float]) -> list[object]:
    """Best-so-far at each grid evaluation count (right-continuous step lookup; None before any value)."""
    xs = [p[0] for p in points]
    values = []
    for g in grid:
        index = bisect.bisect_right(xs, g) - 1
        values.append(points[index][1] if index >= 0 else None)
    return values


def convergence_bands(histories: Sequence[Sequence[Sequence[object]]], grid: Sequence[float]) -> dict[str, list]:
    """Median and interquartile range across runs at each grid point (runs without a value excluded)."""
    columns = list(zip(*(step_values(h, grid) for h in histories), strict=True)) if histories else []
    bands: dict[str, list] = {"median": [], "q25": [], "q75": [], "count": []}
    for column in columns:
        values = [float(v) for v in column if v is not None]
        bands["count"].append(len(values))
        if values:
            q25, median, q75 = np.percentile(values, [QUARTILES[0], 50, QUARTILES[1]])
        else:
            q25 = median = q75 = math.nan
        bands["median"].append(float(median))
        bands["q25"].append(float(q25))
        bands["q75"].append(float(q75))
    return bands


def masked_bands(histories: Sequence[Sequence[Sequence[object]]], grid: Sequence[float]) -> dict[str, list]:
    """Bands shown only where EVERY run already has a valid fold.

    Before that, a median over the runs that happen to have a value would be
    biased toward the better runs, so those grid points are left blank (NaN).
    """
    bands = convergence_bands(histories, grid)
    full = len(histories)
    for key in ("median", "q25", "q75"):
        bands[key] = [v if c == full else math.nan for v, c in zip(bands[key], bands["count"], strict=True)]
    return bands


def load_histories(path: str | Path) -> dict[tuple[str, str], list[list[list[object]]]]:
    histories: dict[tuple[str, str], list] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        histories[(record["variant"], record["seq_id"])].append(record["history"])
    return histories


def _default_history(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}.history.jsonl")


def _sequences(df) -> list[str]:
    order = df.drop_duplicates("seq_id").sort_values(["n", "seq_id"])
    return order["seq_id"].tolist()


def _variants(df) -> list[str]:
    return list(dict.fromkeys(df["variant"]))


def _panels(count: int, size=PANEL_SIZE, **subplot_kw):
    cols = min(count, MAX_COLUMNS)
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(size[0] * cols, size[1] * rows), squeeze=False,
                             subplot_kw=subplot_kw)
    flat = axes.ravel().tolist()
    for ax in flat[count:]:
        ax.set_visible(False)
    return fig, flat[:count]


def _kilo(value: float, _position: int) -> str:
    """Tick label in thousands of evaluations: 0, 10k, 25k, 0.5k ..."""
    return "0" if value == 0 else f"{value / KILO:g}k"


def _single_dim(df) -> int:
    dims = set(df["dim"].astype(int))
    if len(dims) != 1:
        raise ValueError(f"one experiment CSV must have a single dim, found {sorted(dims)}")
    return dims.pop()


def _info(df, seq_id: str) -> Mapping[str, object]:
    row = df.loc[df["seq_id"] == seq_id].iloc[0]
    return {"n": int(row["n"]), "best_known": int(row["best_known"]), "budget": int(row["budget"]),
            "dim": int(row["dim"])}


# ------------------------------------------------------------------ figures ----

def convergence_figure(csv_path: str | Path, history_path: str | Path | None = None):
    csv_path = Path(csv_path)
    df = load_runs(csv_path)
    histories = load_histories(history_path or _default_history(csv_path))
    sequences, variants = _sequences(df), _variants(df)
    fig, axes = _panels(len(sequences))
    for ax, seq_id in zip(axes, sequences, strict=True):
        info = _info(df, seq_id)
        grid = np.linspace(0, info["budget"], GRID_POINTS)
        for variant in variants:
            bands = masked_bands(histories.get((variant, seq_id), []), grid)
            if not bands["median"]:
                continue
            (line,) = ax.plot(grid, bands["median"], drawstyle="steps-post", label=variant)
            ax.fill_between(grid, bands["q25"], bands["q75"], step="post", alpha=BAND_ALPHA,
                            color=line.get_color())
        ax.axhline(info["best_known"], gid="best_known", label="E*", **BEST_KNOWN_STYLE)
        ax.set_title(f"{seq_id} (n = {info['n']}, E* = {info['best_known']})")
        ax.xaxis.set_major_locator(MaxNLocator(MAX_X_TICKS))
        ax.xaxis.set_major_formatter(FuncFormatter(_kilo))
        ax.set_xlabel("fitness evaluations")
        ax.set_ylabel("best energy (median, IQR of all runs)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    # Legend hangs BELOW the figure (top edge at y = 0), so it can never cover a panel;
    # savefig(bbox_inches="tight") grows the image to include it.
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), LEGEND_COLUMNS),
               fontsize="small", bbox_to_anchor=(0.5, 0.0))
    return fig


def box_figure(csv_path: str | Path):
    df = load_runs(csv_path)
    sequences, variants = _sequences(df), _variants(df)
    fig, axes = _panels(len(sequences))
    for ax, seq_id in zip(axes, sequences, strict=True):
        info = _info(df, seq_id)
        part = df.loc[df["seq_id"] == seq_id]
        data, na_counts = [], []
        for variant in variants:
            runs = part.loc[part["variant"] == variant, "best_energy"]
            data.append(runs.dropna().astype(float).to_numpy())
            na_counts.append(int(runs.isna().sum()))
        positions = [i + 1 for i, d in enumerate(data) if len(d)]
        if positions:  # a sequence where every run is NA has nothing to draw
            ax.boxplot([d for d in data if len(d)], positions=positions, widths=0.6)
        ax.set_xticks(range(1, len(variants) + 1), variants, rotation=30, ha="right")
        ax.axhline(info["best_known"], gid="best_known", **BEST_KNOWN_STYLE)
        for i, count in enumerate(na_counts, start=1):
            if count:
                ax.text(i, NA_LABEL_Y, f"NA: {count}", ha="center", va="bottom", fontsize="small",
                        transform=ax.get_xaxis_transform())
        ax.set_title(f"{seq_id} (n = {info['n']}, E* = {info['best_known']})")
        ax.set_ylabel("final best energy")
    fig.tight_layout()
    return fig


def best_folds_figure(csv_path: str | Path):
    df = load_runs(csv_path)
    benchmarks = load_benchmarks()
    sequences, variants = _sequences(df), _variants(df)
    dim = _single_dim(df)
    fig = plt.figure(figsize=(FOLD_PANEL * len(sequences), FOLD_PANEL * len(variants)))
    for r, variant in enumerate(variants):
        for c, seq_id in enumerate(sequences):
            ax = fig.add_subplot(len(variants), len(sequences), r * len(sequences) + c + 1,
                                 projection="3d" if dim == 3 else None)
            runs = df.loc[(df["variant"] == variant) & (df["seq_id"] == seq_id)].dropna(subset=["best_energy"])
            if runs.empty:
                ax.set_title(f"{variant} · {seq_id}\nno valid fold", fontsize="small")
                ax.set_axis_off()
                continue
            # best energy, then fastest, then lowest seed: deterministic tie-break
            best = runs.sort_values(["best_energy", "evals_to_best", "seed"]).iloc[0]
            plot_fold(benchmarks[seq_id]["sequence"], best["best_moves"], dim, ax=ax, title=f"{variant} · {seq_id}")
            ax.title.set_fontsize("small")
    fig.tight_layout()
    return fig


def save(fig, out_dir: str | Path, name: str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.pdf"]
    try:
        for path in paths:
            fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    finally:
        plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render paper figures for one experiment CSV.")
    parser.add_argument("csv", help="results/<experiment>.csv")
    parser.add_argument("--out", help="output directory (default: results/figures/<experiment>)")
    parser.add_argument("--history", help="history JSONL (default: next to the CSV)")
    args = parser.parse_args(argv)
    csv_path = Path(args.csv)
    out = Path(args.out) if args.out else csv_path.parent / "figures" / csv_path.stem
    save(convergence_figure(csv_path, args.history), out, "convergence")
    save(box_figure(csv_path), out, "boxes")
    save(best_folds_figure(csv_path), out, "best_folds")
    print(f"figures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
