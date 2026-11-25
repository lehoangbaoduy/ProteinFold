// Solver worker (PRD FR-L3, FR-L7; PFOLD-DOM-011): loads the self-hosted Pyodide runtime and
// the hpfold package, then runs jobs through hpfold.webjob.run_job_json. Checked in a real
// browser by the Playwright parity test (T5.12).
//
// Protocol (see protocol.ts): init {baseUrl} -> loading..., ready | error(id null);
// run {id, job} -> frame*, result | error(id). Frames and results are forwarded as the JSON
// text Python wrote; the page validates them (parseWorkerMessage).

import type { loadPyodide } from "pyodide";

import { errorMessage, type ToWorker } from "./protocol";
import { runtimeUrls } from "./runtime";

interface WorkerScope {
  postMessage(message: unknown): void;
  onmessage: ((ev: MessageEvent<unknown>) => void) | null;
  location: { origin: string };
}

type RunJobFn = (jobJson: string, emit: (frameJson: string) => void) => string;

const scope = self as unknown as WorkerScope;
const LOAD_ERROR = "Could not load the Python runtime. Reload the page to try again.";
const LIB_DIR = "/home/pyodide/lib";

let runJob: RunJobFn | null = null;

const post = (message: unknown): void => {
  scope.postMessage(message);
};

async function init(baseUrl: string): Promise<void> {
  const urls = runtimeUrls(baseUrl, scope.location.origin);

  post({ type: "loading", stage: "Loading the Python runtime" });
  const pyodideModule = (await import(/* @vite-ignore */ urls.pyodideModule)) as { loadPyodide: typeof loadPyodide };
  const py = await pyodideModule.loadPyodide({ indexURL: urls.pyodideIndex });

  post({ type: "loading", stage: "Loading numpy" });
  await py.loadPackage("numpy", { messageCallback: () => undefined });

  post({ type: "loading", stage: "Loading the solver" });
  const response = await fetch(urls.hpfoldZip);
  if (!response.ok) throw new Error(`hpfold.zip: HTTP ${String(response.status)}`);
  py.unpackArchive(await response.arrayBuffer(), "zip", { extractDir: LIB_DIR });
  py.runPython(`import sys\nif "${LIB_DIR}" not in sys.path: sys.path.insert(0, "${LIB_DIR}")`);
  runJob = py.runPython("from hpfold.webjob import run_job_json\nrun_job_json") as RunJobFn;
  post({ type: "ready" });
}

function run(msg: Extract<ToWorker, { type: "run" }>): void {
  const { id } = msg;
  if (runJob === null) {
    post({ type: "error", id, message: "The solver is not ready yet." });
    return;
  }
  try {
    const json = runJob(JSON.stringify(msg.job), (frameJson) => {
      post({ type: "frame", id, json: frameJson });
    });
    post({ type: "result", id, json });
  } catch (err) {
    post({ type: "error", id, message: errorMessage(err) });
  }
}

let initStarted = false;

scope.onmessage = (ev) => {
  const data: unknown = ev.data;
  if (typeof data !== "object" || data === null) return;
  const msg = data as Partial<ToWorker>;
  if (msg.type === "init" && typeof msg.baseUrl === "string" && !initStarted) {
    initStarted = true; // load the runtime once per worker
    init(msg.baseUrl).catch(() => {
      post({ type: "error", id: null, message: LOAD_ERROR });
    });
  } else if (msg.type === "run" && typeof msg.id === "string" && typeof msg.job === "object") {
    run(msg as Extract<ToWorker, { type: "run" }>);
  }
};
