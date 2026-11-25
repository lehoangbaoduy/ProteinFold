"""hpfold.webjob: the JSON entry point the browser worker calls through Pyodide (PFOLD-DOM-010).

The job JSON comes from the page, so it is untrusted: every field is validated and errors
are plain text that never echo the input. The result must equal a native run exactly,
which is what makes the live solver a faithful copy of the paper's code (T5.12 parity).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from hpfold import webjob
from hpfold.algorithms import ALGORITHMS
from hpfold.hp_model import contacts, fold_coords

SEQ = "HPHPPHHPHPPHPHHPPHPH"  # S1-1, n = 20
JOB = {"algorithm": "GA", "sequence": SEQ, "dim": 2, "seed": 7, "budget": 3000, "params": {}}


def run(job, **kw):
    frames: list[str] = []
    out = webjob.run_job_json(json.dumps(job), frames.append, **kw)
    return json.loads(out), [json.loads(f) for f in frames]


def native(job):
    module = ALGORITHMS[job["algorithm"]]
    frames = []
    result = module.run(job["sequence"], job["dim"], params=module.Params(**job["params"]),
                        seed=job["seed"], budget=job["budget"], target=job.get("target"),
                        on_progress=frames.append)
    return result, frames


@pytest.mark.parametrize("algorithm", ["GA", "ACO", "PSO"])
@pytest.mark.parametrize("dim", [2, 3])
def test_result_equals_native_run(algorithm, dim):
    job = {**JOB, "algorithm": algorithm, "dim": dim, "budget": 2000}
    out, _ = run(job)
    assert out["type"] == "result"
    expected, _ = native(job)
    got = dict(out["result"])
    want = expected.to_dict()
    for d in (got, want):
        d.pop("runtime_s")
    for key in want:
        assert got[key] == want[key], key


def test_result_carries_the_fold_geometry():
    out, _ = run(JOB)
    r = out["result"]
    coords = fold_coords(SEQ, r["best_moves"])
    assert r["coords"] == [list(c) for c in coords]
    assert r["contacts"] == [list(p) for p in contacts(SEQ, coords)]
    assert r["valid"] is True
    assert r["n_contacts"] == -r["best_energy"]


def test_params_override_is_applied():
    out, _ = run({**JOB, "params": {"pop_size": 20, "tournament_k": 2}})
    assert out["result"]["params"]["pop_size"] == 20
    assert out["result"]["params"]["tournament_k"] == 2


def test_target_stops_early():
    out, _ = run({**JOB, "budget": 50_000, "target": -2})
    assert out["result"]["target_reached"] is True
    assert out["result"]["evaluations"] < 50_000


def test_every_frame_is_emitted_without_throttling():
    _, frames = run(JOB, min_interval_s=0.0)
    _, native_frames = native(JOB)
    assert len(frames) == len(native_frames)
    for got, want in zip(frames, native_frames, strict=True):
        assert got["type"] == "frame"
        expected = want.to_dict()
        for key in expected:
            assert got["frame"][key] == expected[key]


def test_throttled_frames_keep_every_improvement_and_the_first_frame():
    _, native_frames = native(JOB)
    _, frames = run(JOB, min_interval_s=1e9, clock=lambda: 0.0)
    emitted = [f["frame"] for f in frames]
    assert emitted[0]["iter"] == native_frames[0].iter
    improvements = {f.best_energy for f in native_frames}
    assert {f["best_energy"] for f in emitted} == improvements
    assert len(emitted) < len(native_frames)  # the throttle really drops frames
    energies = [f["best_energy"] for f in emitted]
    assert all(a is None or (b is not None and b <= a) for a, b in itertools.pairwise(energies))


def test_throttle_emits_again_after_the_interval():
    ticks = iter(range(10_000))
    _, frames = run(JOB, min_interval_s=1.0, clock=lambda: float(next(ticks)))
    _, native_frames = native(JOB)
    assert len(frames) == len(native_frames)  # the clock advances 1 s per frame


def test_frames_carry_best_fold_geometry():
    _, frames = run(JOB, min_interval_s=0.0)
    for f in frames:
        frame = f["frame"]
        if frame["best_moves"] is None:
            assert frame["best_coords"] is None
            assert frame["best_contacts"] == []
        else:
            coords = fold_coords(SEQ, frame["best_moves"])
            assert frame["best_coords"] == [list(c) for c in coords]
            assert frame["best_contacts"] == [list(p) for p in contacts(SEQ, coords)]


def test_fold_payload_for_na():
    assert webjob.fold_payload(SEQ, None) == {"coords": None, "contacts": [], "valid": False,
                                             "n_contacts": None}


def test_default_params_match_the_algorithm_defaults():
    out, _ = run(JOB)
    assert out["result"]["params"] == asdict(ALGORITHMS["GA"].Params())


EVIL = "<script>alert(1)</script>"


@pytest.mark.parametrize(("label", "job_json"), [
    ("not json", "{not json"),
    ("not an object", json.dumps([1, 2])),
    ("missing key", json.dumps({k: v for k, v in JOB.items() if k != "seed"})),
    ("extra key", json.dumps({**JOB, "evil": EVIL})),
    ("unknown algorithm", json.dumps({**JOB, "algorithm": EVIL})),
    ("bad sequence", json.dumps({**JOB, "sequence": EVIL})),
    ("sequence not a string", json.dumps({**JOB, "sequence": 5})),
    ("too short", json.dumps({**JOB, "sequence": "HP"})),
    ("too long", json.dumps({**JOB, "sequence": "H" * 101})),
    ("bad dim", json.dumps({**JOB, "dim": 4})),
    ("bool dim", json.dumps({**JOB, "dim": True})),
    ("negative seed", json.dumps({**JOB, "seed": -1})),
    ("huge seed", json.dumps({**JOB, "seed": 2**31})),
    ("float seed", json.dumps({**JOB, "seed": 1.5})),
    ("zero budget", json.dumps({**JOB, "budget": 0})),
    ("huge budget", json.dumps({**JOB, "budget": 1_000_001})),
    ("params not an object", json.dumps({**JOB, "params": [1]})),
    ("unknown param", json.dumps({**JOB, "params": {EVIL: 1}})),
    ("invalid param value", json.dumps({**JOB, "params": {"pop_size": -1}})),
    ("positive target", json.dumps({**JOB, "target": 1})),
    ("string target", json.dumps({**JOB, "target": "x"})),
    ("oversized json", json.dumps({**JOB, "pad": "x" * 100_000})),
])
def test_rejects_invalid_jobs_with_plain_messages(label, job_json):
    with pytest.raises(ValueError) as info:
        webjob.run_job_json(job_json, lambda _f: None)
    message = str(info.value)
    assert message
    assert EVIL not in message
    assert "<" not in message


def test_boundaries_are_accepted():
    run({**JOB, "sequence": "HPH", "seed": 0, "budget": 1})
    run({**JOB, "seed": 2**31 - 1, "budget": 10, "target": None})


@pytest.mark.parametrize("params", [
    {"pop_size": "100"}, {"pop_size": 100.0}, {"pop_size": True},
    {"pc": "0.9"}, {"pc": True}, {"pc": float("nan")},
    {"crossover": 5}, {"crossover": "x" * 40},
    {"pm": "0.1"}, {"pm": True},
])
def test_parameter_types_must_match_the_defaults(params):
    with pytest.raises(ValueError, match="wrong type"):
        webjob.parse_job(json.dumps({**JOB, "params": params}))


@pytest.mark.parametrize("params", [{"pm": None}, {"pm": 0.05}, {"pc": 1}, {"crossover": "uniform"}])
def test_parameter_values_of_the_right_type_are_accepted(params):
    job = webjob.parse_job(json.dumps({**JOB, "params": params}))
    for key, value in params.items():
        assert getattr(job["params"], key) == value


def test_job_must_be_text():
    with pytest.raises(ValueError):
        webjob.parse_job(b"{}")  # type: ignore[arg-type]


# ---------------------------------------------------------------- review run 34 ----

@pytest.mark.parametrize("params", [{"pc": 10**400}, {"pc": -(10**400)}, {"penalty": 10**400}])
def test_huge_integers_are_a_value_error_not_overflow(params):
    with pytest.raises(ValueError):
        webjob.parse_job(json.dumps({**JOB, "params": params}))


@pytest.mark.parametrize(("algorithm", "params"), [
    ("GA", {"pop_size": webjob.MAX_POPULATION + 1}),
    ("PSO", {"swarm_size": webjob.MAX_POPULATION + 1}),
    ("ACO", {"n_ants": webjob.MAX_POPULATION + 1}),
    ("ACO", {"backtrack_limit": webjob.MAX_LOOP_PARAM + 1}),
    ("ACO", {"max_restarts": webjob.MAX_LOOP_PARAM + 1}),
    ("GA", {"penalty": 1e300}),
])
def test_parameters_are_bounded_so_frames_keep_coming(algorithm, params):
    with pytest.raises(ValueError, match="too large"):
        webjob.parse_job(json.dumps({**JOB, "algorithm": algorithm, "params": params}))


def test_population_cap_is_accepted():
    webjob.parse_job(json.dumps({**JOB, "params": {"pop_size": webjob.MAX_POPULATION}}))


def test_budget_cap_matches_the_thorough_preset():
    assert webjob.MAX_BUDGET == 200_000  # Thorough at n = 100 needs 120,000
    with pytest.raises(ValueError):
        webjob.parse_job(json.dumps({**JOB, "budget": webjob.MAX_BUDGET + 1}))


def test_rejected_param_value_does_not_echo_input():
    with pytest.raises(ValueError) as info:
        webjob.parse_job(json.dumps({**JOB, "params": {"crossover": "<b>x</b>"}}))
    assert "<" not in str(info.value)


def test_deeply_nested_json_is_a_value_error():
    with pytest.raises(ValueError):
        webjob.parse_job("[" * 4900 + "]" * 4900)


@pytest.mark.parametrize("field", ["budget", "seed", "target", "dim"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 2.0])
def test_non_integer_numbers_are_rejected(field, value):
    with pytest.raises(ValueError):
        webjob.parse_job(json.dumps({**JOB, field: value}))


def test_mid_range_throttle_emits_every_third_quiet_frame():
    ticks = iter(range(10_000))
    _, frames = run(JOB, min_interval_s=2.5, clock=lambda: float(next(ticks)))
    _, native_frames = native(JOB)
    # replay the rule by hand: first, every change of best energy, else every >= 2.5 s
    expected, last_emit, last_best = [], None, object()
    for t, f in enumerate(native_frames):
        changed = f.best_energy != last_best
        last_best = f.best_energy
        if last_emit is None or changed or t - last_emit >= 2.5:
            expected.append(f.iter)
            last_emit = t
    assert [f["frame"]["iter"] for f in frames] == expected


def test_last_emitted_frame_holds_the_final_best_fold():
    out, frames = run(JOB, min_interval_s=1e9, clock=lambda: 0.0)
    last = frames[-1]["frame"]
    assert last["best_moves"] == out["result"]["best_moves"]
    assert last["best_coords"] == out["result"]["coords"]


def test_target_zero_and_n3_in_3d():
    out, _ = run({**JOB, "sequence": "HPH", "dim": 3, "budget": 50, "target": 0})
    assert out["result"]["target_reached"] is True


GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "web_golden.json").read_text())


@pytest.mark.parametrize("case", GOLDEN, ids=lambda c: f"{c['job']['algorithm']}-{c['job']['dim']}d")
def test_golden_results_pin_the_browser_numpy(case):
    """Fixed expected folds, shared with the Pyodide parity check (T5.12): native numpy and
    Pyodide's numpy (2.2.5) must both reproduce them, so a numpy drift is caught."""
    out, _ = run(case["job"])
    assert out["result"]["best_energy"] == case["best_energy"]
    assert out["result"]["best_moves"] == case["best_moves"]
    assert out["result"]["evaluations"] == case["evaluations"]
