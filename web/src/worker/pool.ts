// Worker pool for the live solver (PRD FR-L3; PFOLD-DOM-011).
//
// One job = one (algorithm, lattice) run. At most `size` workers run at once; each worker
// loads Pyodide once and then runs jobs one at a time, so extra jobs wait in a FIFO queue.
//
// Cancel: Python inside a worker cannot be interrupted from the page (that would need
// SharedArrayBuffer, which static hosting cannot enable), so cancel terminates every busy
// worker. hpfold.webjob emits a frame on every improvement, so the last frame received is the
// job's best fold so far; it is reported with the "cancelled" event. Idle workers survive.

import { parseWorkerMessage, type Frame, type JobSpec, type RunResultData, type ToWorker } from "./protocol";

/** The part of the Worker API the pool uses (lets tests drive it with a fake). */
export interface WorkerLike {
  postMessage(msg: ToWorker): void;
  terminate(): void;
  onmessage: ((ev: { data: unknown }) => void) | null;
  onerror: ((ev: unknown) => void) | null;
}

export type PoolEvent =
  | { kind: "loading"; stage: string }
  | { kind: "started"; id: string }
  | { kind: "frame"; id: string; frame: Frame }
  | { kind: "done"; id: string; result: RunResultData }
  | { kind: "error"; id: string; message: string }
  | { kind: "cancelled"; id: string; lastFrame: Frame | null };

export interface QueuedJob {
  id: string;
  job: JobSpec;
}

export interface PoolOptions {
  size: number;
  baseUrl: string; // where pyodide/ and py/ are served (the site root)
  createWorker: () => WorkerLike;
  onEvent: (event: PoolEvent) => void;
}

export const MAX_POOL_SIZE = 4; // each worker holds its own Pyodide (~100 MB of memory)
const CRASH_MESSAGE = "The solver stopped unexpectedly.";

/** min(cores, 4), at least 1; 2 when the browser does not say. */
export function defaultPoolSize(cores: number | undefined): number {
  if (cores === undefined) return 2;
  return Math.min(MAX_POOL_SIZE, Math.max(1, Math.floor(cores)));
}

interface Slot {
  worker: WorkerLike;
  ready: boolean;
  job: QueuedJob | null;
  lastFrame: Frame | null;
}

export class SolverPool {
  private readonly slots: Slot[] = [];
  private queue: QueuedJob[] = [];
  private readonly active = new Set<string>(); // ids queued or running
  private readonly opts: PoolOptions;

  constructor(opts: PoolOptions) {
    this.opts = opts;
  }

  /** True while any job is queued or running. */
  get busy(): boolean {
    return this.queue.length > 0 || this.slots.some((s) => s.job !== null);
  }

  run(jobs: readonly QueuedJob[]): void {
    const ids = new Set<string>();
    for (const { id } of jobs) {
      if (this.active.has(id) || ids.has(id)) throw new Error(`Duplicate job id: ${id}`);
      ids.add(id);
    }
    for (const queued of jobs) {
      this.queue.push(queued);
      this.active.add(queued.id);
    }
    this.grow();
    this.dispatch();
  }

  cancel(): void {
    for (const slot of [...this.slots]) {
      if (slot.job === null) continue; // idle or still loading: keep it for the next run
      this.remove(slot);
      this.opts.onEvent({ kind: "cancelled", id: slot.job.id, lastFrame: slot.lastFrame });
    }
    for (const queued of this.queue) this.opts.onEvent({ kind: "cancelled", id: queued.id, lastFrame: null });
    this.queue = [];
    this.active.clear();
  }

  /** Start workers until every queued job has one (up to `size`). */
  private grow(): void {
    const free = () => this.slots.filter((s) => s.job === null).length;
    while (this.slots.length < this.opts.size && free() < this.queue.length) this.spawn();
  }

  private spawn(): void {
    const slot: Slot = { worker: this.opts.createWorker(), ready: false, job: null, lastFrame: null };
    this.slots.push(slot);
    slot.worker.onmessage = (ev) => {
      this.onMessage(slot, ev.data);
    };
    slot.worker.onerror = () => {
      this.onCrash(slot);
    };
    slot.worker.postMessage({ type: "init", baseUrl: this.opts.baseUrl });
  }

  private dispatch(): void {
    for (const slot of this.slots) {
      if (!slot.ready || slot.job !== null) continue;
      const next = this.queue.shift();
      if (next === undefined) return;
      slot.job = next;
      slot.lastFrame = null;
      slot.worker.postMessage({ type: "run", id: next.id, job: next.job });
      this.opts.onEvent({ kind: "started", id: next.id });
    }
  }

  private remove(slot: Slot): void {
    slot.worker.onmessage = null;
    slot.worker.onerror = null;
    slot.worker.terminate();
    this.slots.splice(this.slots.indexOf(slot), 1);
  }

  private finish(slot: Slot): void {
    if (slot.job) this.active.delete(slot.job.id);
    slot.job = null;
    slot.lastFrame = null;
    this.dispatch();
  }

  private onMessage(slot: Slot, data: unknown): void {
    if (!this.slots.includes(slot)) return; // late message from a terminated worker
    const msg = parseWorkerMessage(data);
    if (msg === null) return;
    const current = slot.job?.id;
    switch (msg.type) {
      case "loading":
        this.opts.onEvent({ kind: "loading", stage: msg.stage });
        return;
      case "ready":
        if (!slot.ready) {
          slot.ready = true;
          this.dispatch();
        }
        return;
      case "frame":
        if (msg.id !== current) return;
        slot.lastFrame = msg.frame;
        this.opts.onEvent({ kind: "frame", id: msg.id, frame: msg.frame });
        return;
      case "result":
        if (msg.id !== current) return;
        this.opts.onEvent({ kind: "done", id: msg.id, result: msg.result });
        this.finish(slot);
        return;
      case "error":
        if (msg.id === null && current === undefined) {
          this.failRuntime(slot, msg.message);
        } else if (current !== undefined && (msg.id === current || msg.id === null)) {
          this.opts.onEvent({ kind: "error", id: current, message: msg.message });
          this.finish(slot);
        }
        return;
    }
  }

  /** The Python runtime failed to load: every waiting job would fail the same way. */
  private failRuntime(slot: Slot, message: string): void {
    this.remove(slot);
    for (const queued of this.queue) {
      this.active.delete(queued.id);
      this.opts.onEvent({ kind: "error", id: queued.id, message });
    }
    this.queue = [];
  }

  private onCrash(slot: Slot): void {
    if (!this.slots.includes(slot)) return;
    const job = slot.job;
    this.remove(slot);
    if (job) {
      this.active.delete(job.id);
      this.opts.onEvent({ kind: "error", id: job.id, message: CRASH_MESSAGE });
    }
    this.grow();
    this.dispatch();
  }
}
