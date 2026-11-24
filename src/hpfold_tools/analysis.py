"""Statistics for the comparative study (PRD FR-A1, FR-A2; spec PFOLD-DOM-006).

Conventions
-----------
* Lower energy is better.
* A run that never found a valid fold (NA, empty best_energy) is a FAILURE:
  it counts in `runs` and lowers `success_rate`, it is excluded from the energy
  statistics (it has no energy), and in rank-based comparisons it is ranked
  worse than every valid energy. It is never treated as energy 0.
* success = best_energy <= best_known (E* from data/benchmarks.json).
* evals_to_best is averaged over successful runs only.
* Pairwise comparisons per sequence: two-sided Wilcoxon rank-sum (Mann-Whitney
  U) with PERMUTATION p-values (exact when the number of arrangements is at most
  PERMUTATION_RESAMPLES, otherwise seeded Monte Carlo). The asymptotic normal
  approximation is anticonservative under the heavy ties typical here (many runs
  reach E*), so it is not used. Two Holm-Bonferroni corrections are reported:
  `p_holm` within each sequence's family of pairs, and `p_holm_global` across
  every (sequence, pair) test of the table (controls the family-wise error rate
  of the whole results table; this is the one to report in the paper).
* Effect size: Vargha-Delaney A12 = P(A better than B) + 0.5 P(tie)
  (0.5 = no difference; >0.71 or <0.29 is conventionally a "large" effect).
* Overall ranking: Friedman test across sequences (blocks) on each variant's
  median energy per sequence (NA ranked worst), with mean ranks.

Usage:  python -m hpfold_tools.analysis results/main_2d.csv
"""

from __future__ import annotations

import argparse
import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import PermutationMethod, friedmanchisquare, mannwhitneyu, rankdata

NA_ENERGY = math.inf  # rank NA (no valid fold) below every valid energy
GROUP = ["variant", "seq_id"]
PERMUTATION_RESAMPLES = 20_000
PERMUTATION_SEED = 20261124  # fixed: p-values are reproducible
MIN_FRIEDMAN_VARIANTS = 3
MIN_FRIEDMAN_BLOCKS = 2
PAIRWISE_COLUMNS = ["p_raw", "p_holm", "p_holm_global", "a12", "n_a", "n_b"]
BOOL_TEXT = {"True": True, "False": False, "true": True, "false": False}


@dataclass(frozen=True)
class FriedmanResult:
    statistic: float
    p_value: float
    mean_ranks: dict[str, float]  # 1 = best
    n_blocks: int


def load_runs(path: str | Path) -> pd.DataFrame:
    """Runs CSV -> DataFrame; best_energy / evals_to_best are nullable integers (NA = no valid fold)."""
    df = pd.read_csv(path, dtype={"best_moves": str, "variant": str, "seq_id": str})
    for column in ("best_energy", "evals_to_best"):
        df[column] = pd.to_numeric(df[column], errors="raise").astype("Int64")
    if df["target_reached"].dtype != bool:
        df["target_reached"] = df["target_reached"].map(lambda v: v if isinstance(v, bool) else BOOL_TEXT[str(v)])
    return df


def _success(group: pd.DataFrame) -> pd.Series:
    return (group["best_energy"] <= group["best_known"]).fillna(False).astype(bool)


def _group_stats(group: pd.DataFrame) -> dict[str, float]:
    energies = group["best_energy"].dropna().astype(float)
    success = _success(group)
    found = len(energies)
    return {
        "runs": len(group),
        "found": found,
        "best": energies.min() if found else math.nan,
        "mean": energies.mean() if found else math.nan,
        "std": energies.std(ddof=1) if found > 1 else math.nan,
        "median": energies.median() if found else math.nan,
        "worst": energies.max() if found else math.nan,
        "success_rate": float(success.mean()) if len(group) else math.nan,
        "evals_to_best_success": group.loc[success, "evals_to_best"].astype(float).mean()
        if success.any() else math.nan,
        "runtime_mean": float(group["runtime_s"].mean()),
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, seq_id) summary table (PRD FR-A1)."""
    records = {key: _group_stats(group) for key, group in df.groupby(GROUP, sort=True)}
    table = pd.DataFrame.from_dict(records, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index, names=GROUP)
    return table


def _energies(values: Iterable[object]) -> np.ndarray:
    return np.array([NA_ENERGY if v is None or pd.isna(v) else float(v) for v in values])


def a12(a: Sequence[object], b: Sequence[object]) -> float:
    """Vargha-Delaney A12 for minimization: P(a < b) + 0.5 * P(a == b); NA counts as worst."""
    xs, ys = _energies(a), _energies(b)
    if not len(xs) or not len(ys):
        return math.nan
    wins = (xs[:, None] < ys[None, :]).sum()
    ties = (xs[:, None] == ys[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(xs) * len(ys)))


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order (monotone, capped at 1)."""
    m = len(pvalues)
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted.tolist()


def _rank_sum_p(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(np.unique(np.concatenate([xs, ys]))) == 1:
        return 1.0  # every value identical: no evidence of any difference
    method = PermutationMethod(n_resamples=PERMUTATION_RESAMPLES, random_state=PERMUTATION_SEED)
    return float(mannwhitneyu(xs, ys, alternative="two-sided", method=method).pvalue)


def pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """Per sequence, every pair of variants (in order of first appearance): p_raw, p_holm, a12."""
    records = []
    for seq_id, group in df.groupby("seq_id", sort=True):
        variants = list(dict.fromkeys(group["variant"]))
        samples = {v: _energies(group.loc[group["variant"] == v, "best_energy"]) for v in variants}
        pairs = list(itertools.combinations(variants, 2))
        raw = [_rank_sum_p(samples[a], samples[b]) for a, b in pairs]
        for (a, b), p, p_adj in zip(pairs, raw, holm(raw), strict=True):
            records.append({"seq_id": seq_id, "a": a, "b": b, "p_raw": p, "p_holm": p_adj,
                            "a12": a12(samples[a], samples[b]),
                            "n_a": len(samples[a]), "n_b": len(samples[b])})
    if not records:
        empty = pd.DataFrame(columns=["seq_id", "a", "b", *PAIRWISE_COLUMNS])
        return empty.set_index(["seq_id", "a", "b"])
    table = pd.DataFrame.from_records(records)
    table["p_holm_global"] = holm(table["p_raw"].tolist())
    return table.set_index(["seq_id", "a", "b"])[PAIRWISE_COLUMNS]


def friedman(df: pd.DataFrame) -> FriedmanResult:
    """Friedman test over sequences (blocks) using each variant's median energy (NA worst).

    Only sequences on which every variant was run are used (a complete block design).
    """
    medians = df.assign(score=_energies(df["best_energy"])).pivot_table(
        index="seq_id", columns="variant", values="score", aggfunc="median").dropna()
    variants = list(medians.columns)
    ranks = np.vstack([rankdata(row) for row in medians.to_numpy()]) if len(medians) else np.empty((0, 0))
    mean_ranks = {v: float(ranks[:, i].mean()) for i, v in enumerate(variants)} if len(medians) else {}
    if len(variants) < MIN_FRIEDMAN_VARIANTS or len(medians) < MIN_FRIEDMAN_BLOCKS:
        return FriedmanResult(math.nan, math.nan, mean_ranks, len(medians))
    stat = friedmanchisquare(*(medians[v].to_numpy() for v in variants))
    return FriedmanResult(float(stat.statistic), float(stat.pvalue), mean_ranks, len(medians))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize and compare runs from an experiment CSV.")
    parser.add_argument("csv", help="results/<experiment>.csv")
    args = parser.parse_args(argv)
    df = load_runs(args.csv)
    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.precision", 3):
        print(summarize(df))
        print()
        print(pairwise(df))
        print()
        print(friedman(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
