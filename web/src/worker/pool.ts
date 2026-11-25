// Worker pool for the live solver (PRD FR-L3; PFOLD-DOM-011).
//
// One job = one (algorithm, lattice) run. At most `size` workers run at once; each worker
// loads Pyodide once and then runs jobs one at a time, so extra jobs wait in a FIFO queue.
//
// Cancel: Python inside a worker cannot be interrupted from the page (that would need
// SharedArrayBuffer, which static hosting cannot enable), so cancel terminates every busy
// worker. hpfold.webjob emits a frame on every improvement, so the last frame received is the
// job's best fold so far; it is reported with the "cancelled" event. Idle workers survive.
//
// Every job ends with exactly one terminal event (done, error or cancelled). The pool updates
// its own state before emitting any event, so an event handler may call run() or cancel().

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
  /** Absolute URL of the directory serving pyodide/ and py/ (the site root). */
  baseUrl: string;
  createWorker: () => WorkerLike;
  onEvent: (event: PoolEvent) => void;
  /** A worker that is not ready after this long counts as failed to load. */
  loadTimeoutMs?: number;
}

export const MAX_POOL_SIZE = 4; // each worker holds its own Pyodide (~100 MB of memory)
export const DEFAULT_LOAD_TIMEOUT_MS = 180_000; // ~15 MB download on a slow connection
const CRASH_MESSAGE = "The solver stopped unexpectedly.";
const START_MESSAGE = "Could not start the solver.";
const TIMEOUT_MESSAGE = "Loading the Python runtime timed out.";
const UNREADABLE_MESSAGE = "The solver returned an unreadable result.";

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
  loadTimer: ReturnType<typeof setTimeout> | null;
}

/** A result/error message addressed to `id` (used when its content fails validation). */
function isTerminalFor(data: unknown, id: string): boolean {
  if (typeof data !== "object" || data === null) return false;
  const msg = data as { type?: unknown; id?: unknown };
  return (msg.type === "result" || msg.type === "error") && msg.id === id;
}

export class SolverPool {
  private slots: Slot[] = [];
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
    const running = this.slots.filter((s) => s.job !== null);
    const waiting = this.queue;
    // reset first: a handler of the events below may start new jobs
    for (const slot of running) this.remove(slot);
    this.queue = [];
    this.active.clear();
    for (const slot of running) {
      if (slot.job) this.emit({ kind: "cancelled", id: slot.job.id, lastFrame: slot.lastFrame });
    }
    for (const queued of waiting) this.emit({ kind: "cancelled", id: queued.id, lastFrame: null });
  }

  private emit(event: PoolEvent): void {
    this.opts.onEvent(event);
  }

  /** Start workers until every queued job has one (up to `size`). */
  private grow(): void {
    const free = () => this.slots.filter((s) => s.job === null).length;
    while (this.slots.length < this.opts.size && free() < this.queue.length) this.spawn();
  }

  private spawn(): void {
    const slot: Slot = { worker: this.opts.createWorker(), ready: false, job: null, lastFrame: null, loadTimer: null };
    this.slots.push(slot);
    slot.loadTimer = setTimeout(() => {
      this.loadFailed(slot, TIMEOUT_MESSAGE);
    }, this.opts.loadTimeoutMs ?? DEFAULT_LOAD_TIMEOUT_MS);
    slot.worker.onmessage = (ev) => {
      this.onMessage(slot, ev.data);
    };
    slot.worker.onerror = () => {
      this.onCrash(slot);
    };
    slot.worker.postMessage({ type: "init", baseUrl: this.opts.baseUrl });
  }

  private dispatch(): void {
    for (const slot of [...this.slots]) {
      if (!this.slots.includes(slot) || !slot.ready || slot.job !== null) continue;
      const next = this.queue.shift();
      if (next === undefined) return;
      slot.job = next;
      slot.lastFrame = null;
      slot.worker.postMessage({ type: "run", id: next.id, job: next.job });
      this.emit({ kind: "started", id: next.id });
    }
  }

  private remove(slot: Slot): void {
    if (slot.loadTimer !== null) clearTimeout(slot.loadTimer);
    slot.loadTimer = null;
    slot.worker.onmessage = null;
    slot.worker.onerror = null;
    slot.worker.terminate();
    this.slots = this.slots.filter((s) => s !== slot);
  }

  /** End the slot's current job with `event` (state is cleared first), then continue. */
  private finish(slot: Slot, event: PoolEvent): void {
    if (slot.job) this.active.delete(slot.job.id);
    slot.job = null;
    slot.lastFrame = null;
    this.emit(event);
    this.dispatch();
  }

  private onMessage(slot: Slot, data: unknown): void {
    if (!this.slots.includes(slot)) return; // late message from a terminated worker
    const current = slot.job?.id;
    const msg = parseWorkerMessage(data);
    if (msg === null) {
      // a malformed frame is skipped, but a malformed end of the job must still end it
      if (current !== undefined && isTerminalFor(data, current)) {
        this.finish(slot, { kind: "error", id: current, message: UNREADABLE_MESSAGE });
      }
      return;
    }
    switch (msg.type) {
      case "loading":
        this.emit({ kind: "loading", stage: msg.stage });
        return;
      case "ready":
        if (!slot.ready) {
          slot.ready = true;
          if (slot.loadTimer !== null) clearTimeout(slot.loadTimer);
          slot.loadTimer = null;
          this.dispatch();
        }
        return;
      case "frame":
        if (msg.id !== current) return;
        slot.lastFrame = msg.frame;
        this.emit({ kind: "frame", id: msg.id, frame: msg.frame });
        return;
      case "result":
        if (msg.id !== current) return;
        this.finish(slot, { kind: "done", id: msg.id, result: msg.result });
        return;
      case "error":
        if (current === undefined) {
          if (msg.id === null) this.loadFailed(slot, msg.message);
        } else if (msg.id === current || msg.id === null) {
          this.finish(slot, { kind: "error", id: current, message: msg.message });
        }
        return;
    }
  }

  /**
   * A worker could not load the runtime (error, crash or timeout before ready). If another
   * worker is still alive, it takes over the queue; otherwise every waiting job fails with the
   * message. No replacement is spawned here, so a broken deployment cannot respawn forever;
   * the next run() tries again.
   */
  private loadFailed(slot: Slot, message: string): void {
    if (!this.slots.includes(slot)) return;
    this.remove(slot);
    if (this.slots.length > 0) return;
    const waiting = this.queue;
    this.queue = [];
    for (const queued of waiting) this.active.delete(queued.id);
    for (const queued of waiting) this.emit({ kind: "error", id: queued.id, message });
  }

  private onCrash(slot: Slot): void {
    if (!this.slots.includes(slot)) return;
    if (!slot.ready) {
      this.loadFailed(slot, START_MESSAGE);
      return;
    }
    const job = slot.job;
    this.remove(slot);
    if (job) {
      this.active.delete(job.id);
      this.emit({ kind: "error", id: job.id, message: CRASH_MESSAGE });
    }
    this.grow(); // a crash consumes its job, so respawning here cannot loop
    this.dispatch();
  }
}
