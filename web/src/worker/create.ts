// Adapter from a real module Worker to the pool's WorkerLike interface (PFOLD-DOM-011).
import type { WorkerLike } from "./pool";
import type { ToWorker } from "./protocol";

export function createSolverWorker(): WorkerLike {
  const worker = new Worker(new URL("./solver.worker.ts", import.meta.url), { type: "module" });
  const like: WorkerLike = {
    postMessage: (msg: ToWorker) => {
      worker.postMessage(msg);
    },
    terminate: () => {
      worker.terminate();
    },
    onmessage: null,
    onerror: null,
  };
  worker.onmessage = (ev: MessageEvent<unknown>) => like.onmessage?.({ data: ev.data });
  worker.onerror = (ev: ErrorEvent) => {
    ev.preventDefault();
    like.onerror?.(ev);
  };
  return like;
}
