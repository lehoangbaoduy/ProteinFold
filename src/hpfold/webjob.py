"""JSON entry point for the browser solver (PRD FR-L3, FR-L4, FR-L5, FR-T1).

The web worker loads this package in Pyodide and calls::

    run_job_json(job_json, emit) -> result_json

so the page runs exactly the code used for the paper. The job comes from the page and is
treated as untrusted: every field is checked, and error messages are fixed plain text that
never repeat the input (the page shows them with textContent).

Frames are throttled for the UI: a frame is emitted on every change of the best energy and
otherwise at most once per `min_interval_s`. The first frame is always emitted. Because every
improvement is emitted, the page can terminate the worker to cancel a run and still keep the
best fold found so far (Python cannot be interrupted from JavaScript without
SharedArrayBuffer, which static hosting cannot enable). Algorithms emit one frame per
iteration, so "so far" means "up to the last finished iteration"; the population-size caps
below keep an iteration short enough (a fraction of a second at n = 100) for this to hold.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import fields
from typing import Any

from hpfold.algorithms import ALGORITHMS
from hpfold.algorithms.base import Frame
from hpfold.hp_model import contacts, fold_coords, is_valid

MIN_LENGTH = 3
MAX_LENGTH = 100  # browser cap (D21); the Python package itself has no cap
MAX_SEED = 2**31 - 1
MAX_BUDGET = 200_000  # Thorough at n = 100 needs 120,000; caps a run at a few minutes
MAX_POPULATION = 1000  # pop_size / swarm_size / n_ants: bounds the time between frames
MAX_LOOP_PARAM = 1000  # ACO backtrack_limit / max_restarts: bounds the time per ant
MAX_ABS_NUMBER = 1e6  # any other numeric parameter
POPULATION_PARAMS = frozenset({"pop_size", "swarm_size", "n_ants"})
LOOP_PARAMS = frozenset({"backtrack_limit", "max_restarts"})
MAX_JOB_CHARS = 10_000  # a valid job is well under 1,000 characters
MAX_TEXT_PARAM = 32
DEFAULT_MIN_INTERVAL_S = 0.1

REQUIRED_KEYS = frozenset({"algorithm", "sequence", "dim", "seed", "budget", "params"})
OPTIONAL_KEYS = frozenset({"target"})
SEQUENCE = re.compile(r"[HP]+")

EmitFn = Callable[[str], None]
ClockFn = Callable[[], float]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    # ints are never passed to math.isfinite: a 400-digit int would raise OverflowError
    return _is_int(value) or (isinstance(value, float) and math.isfinite(value))


def _limit_for(key: str) -> float:
    if key in POPULATION_PARAMS:
        return MAX_POPULATION
    if key in LOOP_PARAMS:
        return MAX_LOOP_PARAM
    return MAX_ABS_NUMBER


def _check_param(default: Any, value: Any) -> bool:
    """A value is accepted if it has the same JSON type as the parameter's default."""
    if default is None:  # optional numbers such as the GA mutation rate pm
        return value is None or _is_number(value)
    if isinstance(default, str):
        return isinstance(value, str) and len(value) <= MAX_TEXT_PARAM
    if _is_int(default):
        return _is_int(value)
    if isinstance(default, float):
        return _is_number(value)
    return False


def _build_params(algorithm: str, raw: Any) -> Any:
    module = ALGORITHMS[algorithm]
    if not isinstance(raw, dict):
        raise ValueError("The parameters must be an object.")  # noqa: TRY004 - contract: invalid job -> ValueError
    defaults = {f.name: getattr(module.Params(), f.name) for f in fields(module.Params)}
    for key, value in raw.items():
        if key not in defaults:
            raise ValueError(f"Unknown parameter for {algorithm}.")
        if not _check_param(defaults[key], value):
            raise ValueError(f"Parameter {key} for {algorithm} has the wrong type.")
        if _is_number(value) and abs(value) > _limit_for(key):
            raise ValueError(f"Parameter {key} for {algorithm} is too large.")
    try:
        return module.Params(**raw)
    except ValueError as err:  # Params messages are fixed text naming the parameter
        raise ValueError(f"Invalid parameters for {algorithm}: {err}") from None


def parse_job(job_json: str) -> dict[str, Any]:
    """Validate a job; returns {algorithm, sequence, dim, seed, budget, params, target}."""
    if not isinstance(job_json, str) or len(job_json) > MAX_JOB_CHARS:
        raise ValueError("The job is too large or not text.")
    try:
        job = json.loads(job_json)
    except (ValueError, RecursionError):  # deep nesting can exhaust the smaller wasm stack
        raise ValueError("The job is not valid JSON.") from None
    if not isinstance(job, dict):
        raise ValueError("The job must be a JSON object.")  # noqa: TRY004 - contract: invalid job -> ValueError
    keys = set(job)
    if not REQUIRED_KEYS <= keys:
        raise ValueError("The job is missing a field.")
    if keys - REQUIRED_KEYS - OPTIONAL_KEYS:
        raise ValueError("The job has an unknown field.")

    algorithm = job["algorithm"]
    if algorithm not in ALGORITHMS:
        raise ValueError("Unknown algorithm (use GA, ACO or PSO).")
    sequence = job["sequence"]
    if not isinstance(sequence, str) or not SEQUENCE.fullmatch(sequence):
        raise ValueError("The sequence may contain only H and P.")
    if not MIN_LENGTH <= len(sequence) <= MAX_LENGTH:
        raise ValueError(f"The sequence length must be {MIN_LENGTH} to {MAX_LENGTH}.")
    dim = job["dim"]
    if not _is_int(dim) or dim not in (2, 3):
        raise ValueError("The lattice must be 2 or 3.")
    seed = job["seed"]
    if not _is_int(seed) or not 0 <= seed <= MAX_SEED:
        raise ValueError(f"The seed must be an integer from 0 to {MAX_SEED}.")
    budget = job["budget"]
    if not _is_int(budget) or not 1 <= budget <= MAX_BUDGET:
        raise ValueError(f"The budget must be an integer from 1 to {MAX_BUDGET}.")
    target = job.get("target")
    if target is not None and (not _is_int(target) or target > 0):
        raise ValueError("The target energy must be zero, a negative integer, or null.")
    params = _build_params(algorithm, job["params"])
    return {"algorithm": algorithm, "sequence": sequence, "dim": dim, "seed": seed,
            "budget": budget, "params": params, "target": target}


def fold_payload(sequence: str, moves: str | None) -> dict[str, Any]:
    """Coordinates and H-H contacts of a fold, ready for drawing (None = no valid fold)."""
    if moves is None:
        return {"coords": None, "contacts": [], "valid": False, "n_contacts": None}
    coords = fold_coords(sequence, moves)
    valid = is_valid(coords)
    pairs = contacts(sequence, coords) if valid else ()
    return {"coords": [list(c) for c in coords], "contacts": [list(p) for p in pairs],
            "valid": valid, "n_contacts": len(pairs) if valid else None}


class _Throttle:
    """Decides which frames reach the page: the first, every improvement, then by time."""

    def __init__(self, min_interval_s: float, clock: ClockFn) -> None:
        self._interval = min_interval_s
        self._clock = clock
        self._last_emit: float | None = None
        self._last_best: object = object()  # sentinel: differs from any energy, even None

    def wants(self, frame: Frame) -> bool:
        now = self._clock()
        improved = frame.best_energy != self._last_best
        self._last_best = frame.best_energy
        if self._last_emit is None or improved or now - self._last_emit >= self._interval:
            self._last_emit = now
            return True
        return False


def run_job_json(job_json: str, emit: EmitFn, min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
                 clock: ClockFn = time.monotonic) -> str:
    """Run one browser job; frames go to `emit`, the result is returned (both as JSON text)."""
    job = parse_job(job_json)
    sequence = job["sequence"]
    throttle = _Throttle(min_interval_s, clock)

    def on_progress(frame: Frame) -> None:
        if throttle.wants(frame):
            payload = frame.to_dict()
            geometry = fold_payload(sequence, frame.best_moves)
            payload["best_coords"] = geometry["coords"]
            payload["best_contacts"] = geometry["contacts"]
            emit(json.dumps({"type": "frame", "frame": payload}))

    module = ALGORITHMS[job["algorithm"]]
    result = module.run(sequence, job["dim"], params=job["params"], seed=job["seed"],
                        budget=job["budget"], target=job["target"], on_progress=on_progress)
    return json.dumps({"type": "result",
                       "result": {**result.to_dict(), **fold_payload(sequence, result.best_moves)}})
