"""Static data for the web page (PRD FR-B5, FR-E6, FR-T1, FR-T2, FR-W2; spec PFOLD-DATA-003).

Writes into web/public/data/ (or --out):

* defaults.json    tuned parameters per algorithm (first config; one variant per
                   algorithm), effort presets (budget = max(min, c n^2), D21), limits
* benchmarks.json  benchmark sequences with E*, citations and source notes
* experiments.json one entry per experiment: settings, variants (params + hash), trace index
* summary.json     per-experiment summary and pairwise statistics (dashboard)
* traces/<experiment>/index.json + one JSON per replayed run: the best and the
  (lower) median run of every (variant, sequence).

Replays are trustworthy by construction: a pre-flight check compares the CSV with
the config (experiment name, variants, params hash, budget) before anything runs,
and every re-run must reproduce the CSV row exactly (best energy, fold,
evaluations, evaluations to best, target reached). Re-runs are deterministic, so
running them in parallel gives byte-identical files.

Usage:  python -m hpfold_tools.export_web --config configs/main_2d.toml --config configs/ext_3d.toml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from hpfold.algorithms import ALGORITHMS
from hpfold_tools.analysis import load_runs, pairwise, summarize
from hpfold_tools.experiments import (
    DEFAULT_BENCHMARKS,
    ROOT,
    budget_for,
    load_config,
    params_hash,
    resolve_params,
)

DEFAULT_OUT = ROOT / "web" / "public" / "data"
EFFORT_PRESETS = ({"name": "Quick", "c": 2, "min": 5000},
                  {"name": "Standard", "c": 5, "min": 5000},
                  {"name": "Thorough", "c": 12, "min": 5000})
LIMITS = {"min_length": 3, "max_length": 100}
MAX_TRACE_BYTES = 2_000_000
MAX_FRAMES = 300
MIN_FRAMES = 2
FRAME_EVERY = 10
KINDS = ("best", "median")
MEDIAN_RULE = "lower"  # for an even number of runs, the lower of the two middle runs (a real run)
CHECKED_FIELDS = ("best_energy", "best_moves", "evaluations", "evals_to_best", "target_reached")


def _json_safe(value: Any) -> Any:
    """Recursively replace NaN/inf/pd.NA with None and numpy scalars with Python numbers."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if value is pd.NA:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _dumps(data: Any) -> str:
    return json.dumps(_json_safe(data), allow_nan=False, separators=(",", ":"))


def _write(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dumps(data), encoding="utf-8")


# ------------------------------------------------------------------ small files ----

def write_defaults(config_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    algorithms: dict[str, Any] = {}
    for variant in cfg.variants:
        if variant.algorithm in algorithms:
            raise ValueError(f"{Path(config_path).name}: more than one variant of {variant.algorithm}; "
                             "web defaults need exactly one per algorithm")
        algorithms[variant.algorithm] = asdict(resolve_params(variant.algorithm, variant.params))
    data = {"source_config": Path(config_path).name, "algorithms": algorithms,
            "effort": [dict(e) for e in EFFORT_PRESETS], "limits": dict(LIMITS)}
    _write(Path(out_dir) / "defaults.json", data)
    return _json_safe(data)


def write_benchmarks(out_dir: str | Path, benchmarks_path: str | Path = DEFAULT_BENCHMARKS) -> dict[str, Any]:
    raw = json.loads(Path(benchmarks_path).read_text(encoding="utf-8"))
    instances = [{"id": b["id"], "dim": b["dim"], "length": b["length"], "sequence": b["sequence"],
                  "best_known_energy": b["best_known_energy"], "notes": b.get("notes", ""),
                  "sources": [raw["sources"][key] for key in b["sources"]]}
                 for b in raw["instances"]]
    data = {"instances": instances}
    _write(Path(out_dir) / "benchmarks.json", data)
    return data


def write_experiments(config_paths: Iterable[str | Path], out_dir: str | Path) -> dict[str, Any]:
    entries = []
    for path in config_paths:
        cfg = load_config(path)
        variants = {}
        for v in cfg.variants:
            resolved = asdict(resolve_params(v.algorithm, v.params))
            variants[v.label] = {"algorithm": v.algorithm, "params": resolved,
                                 "params_hash": params_hash(v.algorithm, resolved)}
        entries.append({"name": cfg.name, "dim": cfg.dim, "instances": list(cfg.instances), "seeds": cfg.seeds,
                        "budget_c": cfg.budget_c, "budget_min": cfg.budget_min,
                        "stop_at_best_known": cfg.stop_at_best_known, "variants": variants,
                        "trace_index": f"traces/{cfg.name}/index.json"})
    data = {"experiments": entries}
    _write(Path(out_dir) / "experiments.json", data)
    return _json_safe(data)


def write_summary(csv_paths: Iterable[str | Path], out_dir: str | Path) -> dict[str, Any]:
    experiments = {}
    for csv_path in csv_paths:
        df = load_runs(csv_path)
        summary = summarize(df).reset_index().to_dict(orient="records")
        tests = pairwise(df)
        pairs = tests.reset_index().to_dict(orient="records") if not tests.empty else []
        experiments[str(df["experiment"].iloc[0])] = {"summary": summary, "pairwise": pairs}
    data = _json_safe({"experiments": experiments})
    _write(Path(out_dir) / "summary.json", data)
    return data


# ------------------------------------------------------------------ traces ----

def _na(value: Any) -> Any:
    return None if pd.isna(value) else value


def select_runs(df: pd.DataFrame, kinds: Sequence[str] = KINDS) -> list[dict[str, Any]]:
    """Best and lower-median run per (variant, seq_id): ordered by energy (NA last),
    then evaluations to best, then seed, so the choice is deterministic."""
    picks = []
    for (variant, seq_id), group in df.groupby(["variant", "seq_id"], sort=True):
        ordered = group.assign(
            _e=group["best_energy"].astype("Float64").fillna(math.inf),
            _t=group["evals_to_best"].astype("Float64").fillna(math.inf),
        ).sort_values(["_e", "_t", "seed"])
        positions = {"best": 0, "median": (len(ordered) - 1) // 2}
        for kind in kinds:
            row = ordered.iloc[positions[kind]]
            energy, moves, to_best = _na(row["best_energy"]), _na(row["best_moves"]), _na(row["evals_to_best"])
            picks.append({
                "variant": variant, "algorithm": row["algorithm"], "seq_id": seq_id, "seed": int(row["seed"]),
                "kind": kind, "rank": positions[kind] + 1, "runs": len(ordered), "median_rule": MEDIAN_RULE,
                "dim": int(row["dim"]), "n": int(row["n"]), "budget": int(row["budget"]),
                "best_known": int(row["best_known"]),
                "best_energy": None if energy is None else int(energy),
                "best_moves": None if moves is None else str(moves),
                "evaluations": int(row["evaluations"]),
                "evals_to_best": None if to_best is None else int(to_best),
                "target_reached": bool(row["target_reached"]),
            })
    return picks


def thin_frames(frames: Sequence[Mapping[str, Any]], every: int = FRAME_EVERY,
                max_frames: int = MAX_FRAMES) -> list[Mapping[str, Any]]:
    """Keep the first and last frame and every improvement, plus every `every`-th frame,
    at most `max_frames`. The first, the LAST improvement and the final frame are always
    kept; if improvements alone exceed the cap, the others are sampled evenly."""
    if not frames:
        return []
    last = len(frames) - 1
    improvements = [i for i in range(1, len(frames)) if frames[i]["best_energy"] != frames[i - 1]["best_energy"]]
    always = {0, last} | ({improvements[-1]} if improvements else set())
    must = sorted(always | set(improvements))
    if len(must) > max_frames:
        pool = [i for i in must if i not in always]
        slots = max(max_frames - len(always), 0)
        step = len(pool) / slots if slots else math.inf
        must = sorted(always | {pool[int(k * step)] for k in range(slots)})
    taken = set(must)
    extra = [i for i in range(0, len(frames), max(every, 1)) if i not in taken]
    room = max_frames - len(must)
    if room > 0 and extra:
        step = max(len(extra) / room, 1)
        must = sorted(taken | {extra[int(k * step)] for k in range(min(room, len(extra)))})
    return [frames[i] for i in must]


def _preflight(df: pd.DataFrame, cfg, csv_path: Path, sequences: Mapping[str, str]) -> dict[str, Any]:
    hint = (f"; the CSV was made with a different config or code — re-run the experiment, or use the "
            f"config/commit recorded in results/{cfg.name}.meta.json")
    names = sorted(set(df["experiment"]))
    if names != [cfg.name]:
        raise ValueError(f"{csv_path.name} is from experiment(s) {names}, config is {cfg.name!r}{hint}")
    params = {v.label: resolve_params(v.algorithm, v.params) for v in cfg.variants}
    hashes = {v.label: params_hash(v.algorithm, asdict(params[v.label])) for v in cfg.variants}
    unknown = sorted(set(df["variant"]) - set(params))
    if unknown:
        raise ValueError(f"{csv_path.name} has variant(s) {unknown} not in config {cfg.name!r}{hint}")
    missing = sorted(set(df["seq_id"]) - set(sequences))
    if missing:
        raise ValueError(f"{csv_path.name} has unknown benchmark id(s) {missing}")
    drift = sorted(set(df.loc[df["params_hash"] != df["variant"].map(hashes), "variant"]))
    if drift:
        raise ValueError(f"params_hash in {csv_path.name} differs from config for {drift}{hint}")
    budgets = df.apply(lambda r: budget_for(int(r["n"]), cfg.budget_c, cfg.budget_min), axis=1)
    if (budgets != df["budget"]).any():
        raise ValueError(f"budgets in {csv_path.name} do not match config {cfg.name!r}{hint}")
    return params


def _trace_job(job: tuple) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Re-run one pick (top-level so it can run in a worker process)."""
    algorithm, sequence, dim, params, seed, budget, target = job
    frames: list[dict[str, Any]] = []
    result = ALGORITHMS[algorithm].run(sequence, dim, params=params, seed=seed, budget=budget, target=target,
                                       on_progress=lambda f: frames.append(f.to_dict()))
    return result.to_dict(), frames


def _check_reproduced(pick: Mapping[str, Any], result: Mapping[str, Any], csv_name: str) -> None:
    for field in CHECKED_FIELDS:
        if result[field] != pick[field]:
            raise ValueError(f"could not reproduce {pick['variant']} {pick['seq_id']} seed {pick['seed']} "
                             f"from {csv_name}: {field} = {result[field]!r} on re-run, {pick[field]!r} in CSV")


def _serialize(trace: dict[str, Any], frames: list[dict[str, Any]], name: str) -> str:
    cap = MAX_FRAMES
    while True:
        trace["frames"] = thin_frames(frames, max_frames=cap)
        text = _dumps(trace)
        if len(text.encode()) < MAX_TRACE_BYTES:
            return text
        if cap <= MIN_FRAMES:
            raise ValueError(f"trace {name} exceeds the size limit of {MAX_TRACE_BYTES} bytes")
        cap = max(cap // 2, MIN_FRAMES)


def export_traces(csv_path: str | Path, config_path: str | Path, out_dir: str | Path,
                  kinds: Sequence[str] = KINDS, workers: int = 1) -> dict[str, Any]:
    csv_path = Path(csv_path)
    cfg = load_config(config_path)
    sequences = {b["id"]: b["sequence"] for b in json.loads(DEFAULT_BENCHMARKS.read_text())["instances"]}
    df = load_runs(csv_path)
    params = _preflight(df, cfg, csv_path, sequences)
    picks = select_runs(df, kinds)
    jobs = [(p["algorithm"], sequences[p["seq_id"]], p["dim"], params[p["variant"]], p["seed"], p["budget"],
             p["best_known"] if cfg.stop_at_best_known else None) for p in picks]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(_trace_job, jobs))  # ordered: identical to sequential
    else:
        outputs = [_trace_job(job) for job in jobs]
    folder = Path(out_dir) / "traces" / cfg.name
    if folder.exists():
        shutil.rmtree(folder)  # no stale traces from an earlier export
    folder.mkdir(parents=True)
    entries = []
    for pick, (result, frames) in zip(picks, outputs, strict=True):
        _check_reproduced(pick, result, csv_path.name)
        name = f"{pick['seq_id']}_{pick['variant']}_{pick['kind']}_s{pick['seed']}.json"
        meta_keys = ("variant", "algorithm", "seq_id", "dim", "n", "seed", "kind", "rank", "runs", "median_rule",
                     "budget", "best_known", "best_energy")
        trace = {"meta": {"experiment": cfg.name, **{k: pick[k] for k in meta_keys},
                          "sequence": sequences[pick["seq_id"]], "params": asdict(params[pick["variant"]])},
                 # wall-clock time is host-dependent: dropped so exports are byte-for-byte reproducible
                 "result": {k: v for k, v in result.items() if k != "runtime_s"}}
        (folder / name).write_text(_serialize(trace, frames, name), encoding="utf-8")
        entries.append({"file": f"traces/{cfg.name}/{name}", **{k: pick[k] for k in (
            "variant", "algorithm", "seq_id", "seed", "kind", "rank", "n", "dim", "best_energy", "best_known")}})
    index = {"experiment": cfg.name, "traces": entries}
    _write(folder / "index.json", index)
    return _json_safe(index)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export static JSON data for the web page.")
    parser.add_argument("--config", action="append", required=True,
                        help="experiment config(s); the first one provides the web defaults")
    parser.add_argument("--csv", action="append", help="results CSV per config (default: results/<name>.csv)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (default: web/public/data)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="parallel re-runs")
    args = parser.parse_args(argv)
    configs = [Path(c) for c in args.config]
    csvs = [Path(c) for c in args.csv] if args.csv else [ROOT / "results" / f"{load_config(c).name}.csv"
                                                          for c in configs]
    if len(csvs) != len(configs):
        parser.error("give one --csv per --config (or none)")
    out = Path(args.out)
    write_defaults(configs[0], out)
    write_benchmarks(out)
    write_experiments(configs, out)
    write_summary(csvs, out)
    for config, csv_path in zip(configs, csvs, strict=True):
        index = export_traces(csv_path, config, out, workers=max(args.workers, 1))
        print(f"{index['experiment']}: {len(index['traces'])} traces")
    print(f"web data written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
