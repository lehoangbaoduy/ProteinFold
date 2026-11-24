"""Experiment runner (PRD FR-E1..E5; spec PFOLD-DATA-002).

An experiment is a TOML file (all keys below except seed_offset and
stop_at_best_known are required; unknown keys are rejected)::

    name = "main_2d"            # used for output file names
    dim = 2
    instances = ["S1-1", ...]   # ids from data/benchmarks.json (must match dim)
    seeds = 30                  # seeds seed_offset .. seed_offset + seeds - 1
    seed_offset = 0
    budget_c = 10               # budget B(n) = max(budget_min, ceil(budget_c * n^2))
    budget_min = 10000
    stop_at_best_known = true   # target = E* (the run may stop early)

    [[variants]]                # one entry per compared configuration
    label = "GA"
    algorithm = "GA"
    params = { pop_size = 100 } # overrides of the algorithm's Params defaults

Every (variant, instance, seed) job is run once; seeds are shared across
variants. Outputs in the output dir:

    <name>.csv            one row per run (columns: COLUMNS)
    <name>.history.jsonl  convergence history per run
    <name>.meta.json      config, resolved params + hashes, config hash, sessions

Resume rules (safe by construction):
* jobs already in the CSV are skipped; more seeds or instances may be added;
* a cut-off last line (crash while writing) is trimmed and that run redone;
  any other malformed row is an error;
* the config hash (dim, budget rule, stopping rule, every variant's resolved
  params) must match the one stored in meta.json, otherwise resuming is refused,
  so two parameterizations can never be mixed under one label.

Usage:  python -m hpfold_tools.experiments configs/main_2d.toml --workers 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import numbers
import os
import re
import subprocess  # nosec B404 - only used for a fixed `git` invocation
import tomllib
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hpfold.algorithms import ALGORITHMS

ROOT = Path(__file__).resolve().parents[2]  # assumes the source-tree layout (dev tooling)
DEFAULT_BENCHMARKS = ROOT / "data" / "benchmarks.json"
DEFAULT_OUT = ROOT / "results"
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
HASH_LENGTH = 12
RUNTIME_DECIMALS = 4
INTERRUPTED_EXIT = 130
TOP_LEVEL_KEYS = frozenset({"name", "dim", "instances", "seeds", "seed_offset", "budget_c", "budget_min",
                            "stop_at_best_known", "variants"})
REQUIRED_KEYS = frozenset({"name", "dim", "instances", "seeds", "budget_c", "budget_min", "variants"})
VARIANT_KEYS = frozenset({"label", "algorithm", "params"})
COLUMNS = ("experiment", "variant", "algorithm", "dim", "seq_id", "n", "seed", "budget", "best_known",
           "best_energy", "best_moves", "evaluations", "evals_to_best", "target_reached", "runtime_s",
           "params_hash")


@dataclass(frozen=True)
class Variant:
    label: str
    algorithm: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dim: int
    instances: tuple[str, ...]
    seeds: int
    seed_offset: int
    budget_c: float
    budget_min: int
    stop_at_best_known: bool
    variants: tuple[Variant, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class Job:
    experiment: str
    variant: str
    algorithm: str
    params: Mapping[str, Any]
    params_hash: str
    dim: int
    seq_id: str
    sequence: str
    seed: int
    budget: int
    best_known: int
    target: int | None


@dataclass(frozen=True)
class Summary:
    new: int
    skipped: int
    csv_path: Path


# ------------------------------------------------------------------ config ----

def load_benchmarks(path: str | Path = DEFAULT_BENCHMARKS) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {b["id"]: b for b in data["instances"]}


def _int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _name(value: object, what: str) -> str:
    if not isinstance(value, str) or not NAME_PATTERN.fullmatch(value):
        raise ValueError(f"{what} must match {NAME_PATTERN.pattern}, got {value!r}")
    return value


def _check_keys(table: Mapping[str, Any], allowed: frozenset[str], required: frozenset[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in {where} (allowed: {sorted(allowed)})")
    missing = sorted(required - set(table))
    if missing:
        raise ValueError(f"missing required key(s) {missing} in {where}")


def resolve_params(algorithm: str, overrides: Mapping[str, Any]):
    """The algorithm's Params with `overrides` applied (validated by the Params class)."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm!r} (known: {sorted(ALGORITHMS)})")
    try:
        return ALGORITHMS[algorithm].Params(**dict(overrides))
    except TypeError as exc:  # unknown parameter name
        raise ValueError(f"invalid params for {algorithm}: {exc}") from exc


def params_hash(algorithm: str, params: Mapping[str, Any]) -> str:
    blob = json.dumps({"algorithm": algorithm, "params": dict(params)}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:HASH_LENGTH]


def _variants(raw_variants: object) -> tuple[Variant, ...]:
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("at least one [[variants]] entry is required")
    variants, labels = [], set()
    for entry in raw_variants:
        if not isinstance(entry, dict):
            raise ValueError("each [[variants]] entry must be a table")  # noqa: TRY004 - config errors are ValueError
        _check_keys(entry, VARIANT_KEYS, frozenset({"label", "algorithm"}), "[[variants]]")
        label = _name(entry["label"], "variant label")
        if label in labels:
            raise ValueError(f"duplicate variant label {label!r}")
        labels.add(label)
        params = entry.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"params of {label!r} must be a table")  # noqa: TRY004 - config errors are ValueError
        resolve_params(entry["algorithm"], params)  # validate early
        variants.append(Variant(label, entry["algorithm"], dict(params)))
    return tuple(variants)


def _instances(raw: object, dim: int, benchmarks: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("instances must be a non-empty list of benchmark ids")
    for seq_id in raw:
        if seq_id not in benchmarks:
            raise ValueError(f"unknown benchmark instance {seq_id!r}")
        if benchmarks[seq_id]["dim"] != dim:
            raise ValueError(f"instance {seq_id} is {benchmarks[seq_id]['dim']}D, experiment is {dim}D")
    return tuple(raw)


def load_config(path: str | Path, benchmarks_path: str | Path = DEFAULT_BENCHMARKS) -> ExperimentConfig:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    _check_keys(raw, TOP_LEVEL_KEYS, REQUIRED_KEYS, "the experiment config")
    dim = raw["dim"]
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim!r}")
    budget_c = raw["budget_c"]
    if isinstance(budget_c, bool) or not isinstance(budget_c, numbers.Real) or not math.isfinite(budget_c) \
            or budget_c < 0:
        raise ValueError(f"budget_c must be a finite number >= 0, got {budget_c!r}")
    stop = raw.get("stop_at_best_known", True)
    if not isinstance(stop, bool):
        raise ValueError("stop_at_best_known must be true or false")  # noqa: TRY004 - config errors are ValueError
    return ExperimentConfig(
        name=_name(raw["name"], "name"),
        dim=dim,
        instances=_instances(raw["instances"], dim, load_benchmarks(benchmarks_path)),
        seeds=_int(raw["seeds"], "seeds", 1),
        seed_offset=_int(raw.get("seed_offset", 0), "seed_offset", 0),
        budget_c=float(budget_c),
        budget_min=_int(raw["budget_min"], "budget_min", 0),
        stop_at_best_known=stop,
        variants=_variants(raw["variants"]),
        raw=raw,
    )


def budget_for(n: int, budget_c: float, budget_min: int) -> int:
    """B(n) = max(budget_min, ceil(budget_c * n^2)): the same for every algorithm (PRD FR-E4)."""
    return max(budget_min, math.ceil(budget_c * n * n))


def config_hash(cfg: ExperimentConfig) -> str:
    """Hash of every setting that changes results (not the grid extent: seeds/instances may grow)."""
    variants = [(v.label, v.algorithm, asdict(resolve_params(v.algorithm, v.params))) for v in cfg.variants]
    blob = json.dumps({"dim": cfg.dim, "budget_c": cfg.budget_c, "budget_min": cfg.budget_min,
                       "stop_at_best_known": cfg.stop_at_best_known, "variants": variants}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:HASH_LENGTH]


# ------------------------------------------------------------------ running ----

def plan_jobs(cfg: ExperimentConfig, benchmarks: Mapping[str, Mapping[str, Any]]) -> list[Job]:
    jobs = []
    for variant in cfg.variants:
        resolved = asdict(resolve_params(variant.algorithm, variant.params))
        digest = params_hash(variant.algorithm, resolved)
        for seq_id in cfg.instances:
            bench = benchmarks[seq_id]
            target = bench["best_known_energy"] if cfg.stop_at_best_known else None
            for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.seeds):
                jobs.append(Job(cfg.name, variant.label, variant.algorithm, dict(variant.params), digest,
                                cfg.dim, seq_id, bench["sequence"], seed,
                                budget_for(len(bench["sequence"]), cfg.budget_c, cfg.budget_min),
                                bench["best_known_energy"], target))
    return jobs


def _blank(value: object) -> object:
    return "" if value is None else value


def run_job(job: Job) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one job; returns (csv row, history record). Top-level so it can be pickled."""
    module = ALGORITHMS[job.algorithm]
    result = module.run(job.sequence, job.dim, params=module.Params(**job.params), seed=job.seed,
                        budget=job.budget, target=job.target)
    row = {
        "experiment": job.experiment, "variant": job.variant, "algorithm": job.algorithm,
        "dim": job.dim, "seq_id": job.seq_id, "n": len(job.sequence), "seed": job.seed,
        "budget": job.budget, "best_known": job.best_known,
        "best_energy": _blank(result.best_energy), "best_moves": _blank(result.best_moves),
        "evaluations": result.evaluations, "evals_to_best": _blank(result.evals_to_best),
        "target_reached": result.target_reached, "runtime_s": round(result.runtime_s, RUNTIME_DECIMALS),
        "params_hash": job.params_hash,
    }
    history = {"variant": job.variant, "seq_id": job.seq_id, "seed": job.seed,
               "history": [list(point) for point in result.history]}
    return row, history


def _key(variant: str, seq_id: str, seed: int) -> tuple[str, str, int]:
    return variant, seq_id, int(seed)


def _load_rows(csv_path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    """Existing rows by key; trims a cut-off last line, rejects any other malformed row."""
    if not csv_path.exists():
        return {}
    text = csv_path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):  # crash while writing the last row: drop it, it is re-run
        text = text[: text.rfind("\n") + 1]
        csv_path.write_text(text, encoding="utf-8")
    rows: dict[tuple[str, str, int], dict[str, str]] = {}
    for line_no, row in enumerate(csv.DictReader(io.StringIO(text)), start=2):
        if None in row.values() or None in row or len(row) != len(COLUMNS):
            raise ValueError(f"corrupt row at line {line_no} of {csv_path}; fix or remove it")
        try:
            rows[_key(row["variant"], row["seq_id"], row["seed"])] = row
        except ValueError as exc:
            raise ValueError(f"corrupt row at line {line_no} of {csv_path}: {exc}") from exc
    return rows


def _sync_history(path: Path, keys: set[tuple[str, str, int]]) -> None:
    """Keep exactly one history record per completed CSV row (drops orphans from a crash)."""
    if not path.exists():
        return
    kept: dict[tuple[str, str, int], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            key = _key(record["variant"], record["seq_id"], record["seed"])
        except (ValueError, KeyError, TypeError):
            continue
        if key in keys:
            kept[key] = line
    path.write_text("".join(f"{line}\n" for line in kept.values()), encoding="utf-8")


def _check_consistent(existing: Mapping[tuple[str, str, int], Mapping[str, str]], plan: list[Job]) -> None:
    for job in plan:
        row = existing.get((job.variant, job.seq_id, job.seed))
        if row and (row["params_hash"] != job.params_hash or int(row["budget"]) != job.budget):
            raise ValueError(f"config changed for variant {job.variant!r} since these results were "
                             "written; use a new experiment name or output directory")


def _git_commit() -> str:
    """Short HEAD hash, '-dirty' if tracked sources differ (result files are ignored)."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,  # nosec B603 B607
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", ".", ":!results"], cwd=ROOT,  # nosec B603 B607
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}-dirty" if dirty else head


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _start_meta(cfg: ExperimentConfig, path: Path, git_commit: str) -> dict[str, Any]:
    digest = config_hash(cfg)
    session = {"started_at": _now(), "git_commit": git_commit}
    if path.exists():
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("config_hash") != digest:
            raise ValueError(f"config changed since {path.name} was written (hash {meta.get('config_hash')} "
                             f"-> {digest}); use a new experiment name or output directory")
        meta["sessions"] = [*meta.get("sessions", []), session]
    else:
        meta = {"started_at": session["started_at"], "sessions": [session], "config_hash": digest}
    meta["config"] = cfg.raw
    meta["git_commit"] = git_commit
    meta["variants"] = {
        v.label: {"algorithm": v.algorithm, "params": (resolved := asdict(resolve_params(v.algorithm, v.params))),
                  "params_hash": params_hash(v.algorithm, resolved)}
        for v in cfg.variants
    }
    _write_json(path, meta)
    return meta


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _results(jobs: list[Job], workers: int) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield results in completion order, so a finished run is written immediately."""
    if workers <= 1 or len(jobs) <= 1:
        yield from map(run_job, jobs)
        return
    executor = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            yield future.result()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def run_experiment(cfg: ExperimentConfig, out_dir: str | Path = DEFAULT_OUT, workers: int = 1,
                   benchmarks_path: str | Path = DEFAULT_BENCHMARKS) -> Summary:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{cfg.name}.csv"
    history_path = out / f"{cfg.name}.history.jsonl"
    git_commit = _git_commit()  # before any output exists
    plan = plan_jobs(cfg, load_benchmarks(benchmarks_path))
    existing = _load_rows(csv_path)
    _check_consistent(existing, plan)
    meta = _start_meta(cfg, out / f"{cfg.name}.meta.json", git_commit)
    _sync_history(history_path, set(existing))
    todo = [j for j in plan if (j.variant, j.seq_id, j.seed) not in existing]
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as table, history_path.open("a", encoding="utf-8") as hist:
        writer = csv.DictWriter(table, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for row, history in _results(todo, workers):
            hist.write(json.dumps(history) + "\n")  # history first: an orphan is dropped on resume
            hist.flush()
            writer.writerow(row)
            table.flush()
    meta["finished_at"] = meta["sessions"][-1]["finished_at"] = _now()
    _write_json(out / f"{cfg.name}.meta.json", meta)
    return Summary(new=len(todo), skipped=len(plan) - len(todo), csv_path=csv_path)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an HP folding experiment from a TOML config.")
    parser.add_argument("config", help="path to the experiment TOML file")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (default: results/)")
    parser.add_argument("--workers", type=_positive_int, default=os.cpu_count() or 1, help="parallel processes")
    parser.add_argument("--benchmarks", default=str(DEFAULT_BENCHMARKS), help="benchmark JSON file")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.benchmarks)
    try:
        summary = run_experiment(cfg, args.out, args.workers, args.benchmarks)
    except KeyboardInterrupt:
        print(f"\ninterrupted; completed runs are saved — re-run the same command to resume {cfg.name}")
        return INTERRUPTED_EXIT
    print(f"{cfg.name}: {summary.new} runs done, {summary.skipped} already present -> {summary.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
