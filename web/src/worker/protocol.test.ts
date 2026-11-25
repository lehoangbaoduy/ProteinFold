// Worker message protocol (PFOLD-DOM-011). Messages from a worker are parsed and checked
// field by field before the page uses them; anything malformed is dropped (null).
import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { errorMessage, parseWorkerMessage, type Frame, type RunResultData } from "./protocol";

const frame: Frame = {
  iter: 3,
  evals: 300,
  budget: 5000,
  best_energy: -2,
  best_moves: "FLR",
  current_best_moves: "FLL",
  diversity: 0.41,
  matrix: [[0.333, 0.333, 0.334], [0.5, 0.25, 0.25], [1, 0, 0]],
  best_coords: [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]],
  best_contacts: [[0, 3]],
};

const result: RunResultData = {
  algorithm: "GA",
  sequence: "HPPHH",
  dim: 2,
  seed: 7,
  params: { pop_size: 100, crossover: "two_point", pm: null },
  best_moves: "FLR",
  best_energy: -2,
  evaluations: 5000,
  evals_to_best: 120,
  runtime_s: 0.5,
  history: [[0, null], [100, -1], [120, -2]],
  cancelled: false,
  target_reached: false,
  stats: { pm: 0.5 },
  coords: frame.best_coords,
  contacts: [[0, 3]],
  valid: true,
  n_contacts: 1,
};

const frameMsg = (f: unknown) => ({ type: "frame", id: "GA-2", json: JSON.stringify({ type: "frame", frame: f }) });
const resultMsg = (r: unknown) => ({ type: "result", id: "GA-2", json: JSON.stringify({ type: "result", result: r }) });

describe("parseWorkerMessage", () => {
  it("accepts the simple messages", () => {
    expect(parseWorkerMessage({ type: "loading", stage: "Loading numpy" })).toEqual({ type: "loading", stage: "Loading numpy" });
    expect(parseWorkerMessage({ type: "ready" })).toEqual({ type: "ready" });
    expect(parseWorkerMessage({ type: "error", id: "GA-2", message: "The seed must be ..." })).toEqual({
      type: "error",
      id: "GA-2",
      message: "The seed must be ...",
    });
    expect(parseWorkerMessage({ type: "error", id: null, message: "x" })).toEqual({ type: "error", id: null, message: "x" });
  });

  it("parses and checks a frame from Python's JSON", () => {
    expect(parseWorkerMessage(frameMsg(frame))).toEqual({ type: "frame", id: "GA-2", frame });
  });

  it("accepts a frame before any valid fold (NA)", () => {
    const na = { ...frame, best_energy: null, best_moves: null, best_coords: null, best_contacts: [] };
    expect(parseWorkerMessage(frameMsg(na))).toEqual({ type: "frame", id: "GA-2", frame: na });
  });

  it("parses and checks a result", () => {
    expect(parseWorkerMessage(resultMsg(result))).toEqual({ type: "result", id: "GA-2", result });
  });

  it("accepts an NA result", () => {
    const na = { ...result, best_moves: null, best_energy: null, evals_to_best: null, coords: null, contacts: [], valid: false, n_contacts: null };
    expect(parseWorkerMessage(resultMsg(na))).toEqual({ type: "result", id: "GA-2", result: na });
  });

  it.each([
    ["unknown type", { type: "boom" }],
    ["not an object", "ready"],
    ["null", null],
    ["loading without stage", { type: "loading" }],
    ["error without message", { type: "error", id: "a" }],
    ["error with huge message", { type: "error", id: "a", message: "x".repeat(5000) }],
    ["frame with bad JSON", { type: "frame", id: "a", json: "{" }],
    ["frame without id", { type: "frame", json: JSON.stringify({ type: "frame", frame }) }],
    ["frame wrapped as result", { type: "frame", id: "a", json: JSON.stringify({ type: "result", result }) }],
    ["frame with fractional evals", frameMsg({ ...frame, evals: 1.5 })],
    ["frame with positive energy", frameMsg({ ...frame, best_energy: 1 })],
    ["frame with markup in moves", frameMsg({ ...frame, best_moves: "<img>" })],
    ["frame with 2-number coords", frameMsg({ ...frame, best_coords: [[0, 0]] })],
    ["frame with probability > 1", frameMsg({ ...frame, matrix: [[2, 0, 0]] })],
    ["frame with negative contact index", frameMsg({ ...frame, best_contacts: [[-1, 3]] })],
    ["frame with missing field", frameMsg({ ...frame, diversity: undefined })],
    ["result with string energy", resultMsg({ ...result, best_energy: "-2" })],
    ["result with bad dim", resultMsg({ ...result, dim: 4 })],
    ["result with bad history", resultMsg({ ...result, history: [[0]] })],
    ["result with unknown algorithm", resultMsg({ ...result, algorithm: "SA" })],
  ])("drops %s", (_label, msg) => {
    expect(parseWorkerMessage(msg)).toBeNull();
  });

  it("never throws on arbitrary input (fuzz)", () => {
    fc.assert(
      fc.property(fc.anything(), (value) => {
        parseWorkerMessage(value);
        parseWorkerMessage({ type: "frame", id: "a", json: JSON.stringify(value) });
        parseWorkerMessage({ type: "result", id: "a", json: JSON.stringify(value) });
      }),
      { numRuns: 1000 },
    );
  });
});

describe("errorMessage", () => {
  it("keeps only the ValueError text from a Python traceback", () => {
    const err = new Error(
      'Traceback (most recent call last):\n  File "/lib/hpfold/webjob.py", line 98, in parse_job\nValueError: The seed must be an integer from 0 to 2147483647.\n',
    );
    expect(errorMessage(err)).toBe("The seed must be an integer from 0 to 2147483647.");
  });

  it.each([
    new Error("Traceback ...\nRuntimeError: something internal at 0x7f"),
    new Error("ValueError: " + "x".repeat(1000)),
    "a string",
    null,
    { message: 42 },
  ])("falls back to a fixed message for %j", (err) => {
    expect(errorMessage(err)).toBe("The solver failed unexpectedly.");
  });
});
