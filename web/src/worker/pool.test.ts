// SolverPool (PFOLD-DOM-011): runs jobs on at most `size` workers, FIFO queue, cancel by
// terminating busy workers while keeping each running job's best-so-far frame.
import { describe, expect, it } from "vitest";

import { defaultPoolSize, SolverPool, type PoolEvent, type WorkerLike } from "./pool";
import type { Frame, JobSpec, RunResultData, ToWorker } from "./protocol";

class FakeWorker implements WorkerLike {
  posted: ToWorker[] = [];
  terminated = false;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  postMessage(msg: ToWorker): void {
    this.posted.push(msg);
  }
  terminate(): void {
    this.terminated = true;
  }
  send(data: unknown): void {
    this.onmessage?.({ data });
  }
  get running(): string | null {
    const runs = this.posted.filter((m) => m.type === "run");
    const last = runs[runs.length - 1];
    return last?.type === "run" ? last.id : null;
  }
}

function nth<T>(items: readonly T[], i: number): T {
  const item = items.at(i);
  if (item === undefined) throw new Error(`no item ${String(i)}`);
  return item;
}

const job: JobSpec = { algorithm: "GA", sequence: "HPPHH", dim: 2, seed: 1, budget: 5000, params: {}, target: null };

const frame = (evals: number, energy: number): Frame => ({
  iter: evals / 100,
  evals,
  budget: 5000,
  best_energy: energy,
  best_moves: "FLR",
  current_best_moves: "FLR",
  diversity: 0.5,
  matrix: [[1, 0, 0]],
  best_coords: [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]],
  best_contacts: [[0, 3]],
});

const result: RunResultData = {
  algorithm: "GA", sequence: "HPPHH", dim: 2, seed: 1, params: {}, best_moves: "FLR", best_energy: -1,
  evaluations: 5000, evals_to_best: 100, runtime_s: 0.1, history: [[0, null], [100, -1]], cancelled: false,
  target_reached: false, stats: {}, coords: [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]],
  contacts: [[0, 3]], valid: true, n_contacts: 1,
};

const frameMsg = (id: string, f: Frame) => ({ type: "frame", id, json: JSON.stringify({ type: "frame", frame: f }) });
const resultMsg = (id: string) => ({ type: "result", id, json: JSON.stringify({ type: "result", result }) });

function setup(size: number) {
  const workers: FakeWorker[] = [];
  const events: PoolEvent[] = [];
  const pool = new SolverPool({
    size,
    baseUrl: "https://example.test/app/",
    createWorker: () => {
      const w = new FakeWorker();
      workers.push(w);
      return w;
    },
    onEvent: (e) => events.push(e),
  });
  return { pool, workers, events };
}

const jobs = (...ids: string[]) => ids.map((id) => ({ id, job }));

describe("SolverPool", () => {
  it("starts at most `size` workers, initializes them, and dispatches after ready", () => {
    const { pool, workers, events } = setup(2);
    pool.run(jobs("a", "b", "c"));
    expect(workers).toHaveLength(2);
    for (const w of workers) expect(w.posted).toEqual([{ type: "init", baseUrl: "https://example.test/app/" }]);
    expect(events).toEqual([]);
    nth(workers, 0).send({ type: "ready" });
    expect(nth(workers, 0).running).toBe("a");
    expect(events).toEqual([{ kind: "started", id: "a" }]);
    nth(workers, 1).send({ type: "ready" });
    expect(nth(workers, 1).running).toBe("b");
    expect(pool.busy).toBe(true);
  });

  it("does not start more workers than there are jobs", () => {
    const { pool, workers } = setup(4);
    pool.run(jobs("a"));
    expect(workers).toHaveLength(1);
  });

  it("forwards frames and results, then gives the next queued job to the free worker (FIFO)", () => {
    const { pool, workers, events } = setup(1);
    pool.run(jobs("a", "b"));
    const w = nth(workers, 0);
    w.send({ type: "ready" });
    w.send(frameMsg("a", frame(100, -1)));
    w.send(resultMsg("a"));
    expect(events).toEqual([
      { kind: "started", id: "a" },
      { kind: "frame", id: "a", frame: frame(100, -1) },
      { kind: "done", id: "a", result },
      { kind: "started", id: "b" },
    ]);
    expect(w.running).toBe("b");
    w.send(resultMsg("b"));
    expect(pool.busy).toBe(false);
  });

  it("reuses an idle, already-loaded worker for a later run", () => {
    const { pool, workers, events } = setup(1);
    pool.run(jobs("a"));
    nth(workers, 0).send({ type: "ready" });
    nth(workers, 0).send(resultMsg("a"));
    pool.run(jobs("b"));
    expect(workers).toHaveLength(1);
    expect(nth(workers, 0).running).toBe("b");
    expect(events.at(-1)).toEqual({ kind: "started", id: "b" });
  });

  it("forwards loading progress", () => {
    const { pool, workers, events } = setup(1);
    pool.run(jobs("a"));
    nth(workers, 0).send({ type: "loading", stage: "Loading numpy" });
    expect(events).toEqual([{ kind: "loading", stage: "Loading numpy" }]);
  });

  it("cancel keeps each running job's last frame, cancels the queue, and terminates busy workers only", () => {
    const { pool, workers, events } = setup(2);
    pool.run(jobs("a", "b", "c"));
    nth(workers, 0).send({ type: "ready" });
    nth(workers, 0).send(frameMsg("a", frame(100, -1)));
    nth(workers, 0).send(frameMsg("a", frame(200, -2)));
    // worker 1 is still loading: it has no job and is not terminated
    pool.cancel();
    expect(events.slice(-3)).toEqual([
      { kind: "cancelled", id: "a", lastFrame: frame(200, -2) },
      { kind: "cancelled", id: "b", lastFrame: null },
      { kind: "cancelled", id: "c", lastFrame: null },
    ]);
    expect(nth(workers, 0).terminated).toBe(true);
    expect(nth(workers, 1).terminated).toBe(false);
    expect(pool.busy).toBe(false);

    // late messages from the terminated worker are ignored
    const before = events.length;
    nth(workers, 0).send(resultMsg("a"));
    expect(events).toHaveLength(before);

    // a new run uses the surviving worker and replaces the terminated one
    pool.run(jobs("d", "e"));
    expect(workers).toHaveLength(3);
    nth(workers, 1).send({ type: "ready" });
    expect(nth(workers, 1).running).toBe("d");
  });

  it("cancelling a job that has no frame yet reports lastFrame null", () => {
    const { pool, workers, events } = setup(1);
    pool.run(jobs("a"));
    nth(workers, 0).send({ type: "ready" });
    pool.cancel();
    expect(events.at(-1)).toEqual({ kind: "cancelled", id: "a", lastFrame: null });
  });

  it("a Python error fails that job only; the worker continues with the queue", () => {
    const { pool, workers, events } = setup(1);
    pool.run(jobs("a", "b"));
    nth(workers, 0).send({ type: "ready" });
    nth(workers, 0).send({ type: "error", id: "a", message: "The seed must be an integer." });
    expect(events.slice(-2)).toEqual([
      { kind: "error", id: "a", message: "The seed must be an integer." },
      { kind: "started", id: "b" },
    ]);
    expect(nth(workers, 0).terminated).toBe(false);
  });

  it("a worker crash fails its job, replaces the worker and continues the queue", () => {
    const { pool, workers, events } = setup(1);
    pool.run(jobs("a", "b"));
    nth(workers, 0).send({ type: "ready" });
    nth(workers, 0).onerror?.(new Event("error"));
    expect(events.at(-1)).toEqual({ kind: "error", id: "a", message: "The solver stopped unexpectedly." });
    expect(nth(workers, 0).terminated).toBe(true);
    expect(workers).toHaveLength(2);
    nth(workers, 1).send({ type: "ready" });
    expect(nth(workers, 1).running).toBe("b");
  });

  it("a runtime that fails to load fails the queued jobs with its message", () => {
    const { pool, workers, events } = setup(1);
    pool.run(jobs("a", "b"));
    nth(workers, 0).send({ type: "error", id: null, message: "Could not load the Python runtime." });
    expect(events).toEqual([
      { kind: "error", id: "a", message: "Could not load the Python runtime." },
      { kind: "error", id: "b", message: "Could not load the Python runtime." },
    ]);
    expect(nth(workers, 0).terminated).toBe(true);
    expect(pool.busy).toBe(false);
  });

  it("ignores malformed messages and messages for another job id", () => {
    const { pool, workers, events } = setup(1);
    pool.run(jobs("a"));
    nth(workers, 0).send({ type: "ready" });
    nth(workers, 0).send({ type: "frame", id: "a", json: "{oops" });
    nth(workers, 0).send(frameMsg("zzz", frame(100, -1)));
    nth(workers, 0).send(resultMsg("zzz"));
    nth(workers, 0).send({ type: "ready" }); // a duplicate ready must not start a second job
    expect(events).toEqual([{ kind: "started", id: "a" }]);
  });

  it("rejects duplicate job ids", () => {
    const { pool } = setup(1);
    pool.run(jobs("a"));
    expect(() => {
      pool.run(jobs("a"));
    }).toThrow(/duplicate/i);
  });
});

describe("defaultPoolSize", () => {
  it.each([
    [undefined, 2],
    [0, 1],
    [1, 1],
    [3, 3],
    [16, 4],
  ])("cores %j -> %i workers", (cores, size) => {
    expect(defaultPoolSize(cores)).toBe(size);
  });
});
