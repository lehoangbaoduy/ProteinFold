// Messages between the page and the solver workers (PFOLD-DOM-011).
//
// Page -> worker: ToWorker. Worker -> page: raw objects that parseWorkerMessage checks
// field by field. Frames and results travel as the JSON text written by hpfold.webjob, so
// the page sees exactly what Python produced; anything malformed is dropped (null).

export type Algorithm = "GA" | "ACO" | "PSO";
export type ParamValue = number | string | boolean | null;

export interface JobSpec {
  algorithm: Algorithm;
  sequence: string;
  dim: 2 | 3;
  seed: number;
  budget: number;
  params: Record<string, ParamValue>;
  target: number | null;
}

export type Coord = [number, number, number];
export type Pair = [number, number];

/** Progress frame (PRD FR-T1) plus the best fold's geometry. */
export interface Frame {
  iter: number;
  evals: number;
  budget: number;
  best_energy: number | null;
  best_moves: string | null;
  current_best_moves: string;
  diversity: number;
  matrix: number[][];
  best_coords: Coord[] | null;
  best_contacts: Pair[];
}

/** RunResult.to_dict() plus the fold geometry (hpfold.webjob). */
export interface RunResultData {
  algorithm: Algorithm;
  sequence: string;
  dim: 2 | 3;
  seed: number;
  params: Record<string, ParamValue>;
  best_moves: string | null;
  best_energy: number | null;
  evaluations: number;
  evals_to_best: number | null;
  runtime_s: number;
  history: [number, number | null][];
  cancelled: boolean;
  target_reached: boolean;
  stats: Record<string, number>;
  coords: Coord[] | null;
  contacts: Pair[];
  valid: boolean;
  n_contacts: number | null;
}

export type ToWorker = { type: "init"; baseUrl: string } | { type: "run"; id: string; job: JobSpec };

export type WorkerMessage =
  | { type: "loading"; stage: string }
  | { type: "ready" }
  | { type: "frame"; id: string; frame: Frame }
  | { type: "result"; id: string; result: RunResultData }
  | { type: "error"; id: string | null; message: string };

const MAX_TEXT = 500;
const MAX_ITEMS = 10_000; // beads (<= 100), matrix rows (<= 98), contacts
const MAX_HISTORY = 200_001; // one point per iteration: up to the 200,000 budget, plus (0, null)
const MOVES = /^[FLRUD]*$/;
const SEQUENCE = /^[HP]{3,100}$/;
const ALGORITHMS: readonly string[] = ["GA", "ACO", "PSO"];
const GENERIC_ERROR = "The solver failed unexpectedly.";

type Obj = Record<string, unknown>;

const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const isInt = (v: unknown): v is number => Number.isInteger(v);
const isCount = (v: unknown): v is number => isInt(v) && v >= 0;
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isText = (v: unknown): v is string => typeof v === "string" && v.length <= MAX_TEXT;
const isEnergy = (v: unknown): v is number | null => v === null || (isInt(v) && v <= 0);
const isMoves = (v: unknown): v is string => typeof v === "string" && v.length <= MAX_TEXT && MOVES.test(v);
const isList = (v: unknown): v is unknown[] => Array.isArray(v) && v.length <= MAX_ITEMS;
const isProb = (v: unknown): boolean => isNum(v) && v >= 0 && v <= 1;
const isCoord = (v: unknown): v is Coord => isList(v) && v.length === 3 && v.every(isInt);
const isPair = (v: unknown): v is Pair => isList(v) && v.length === 2 && v.every(isCount);
const isCoords = (v: unknown): v is Coord[] | null => v === null || (isList(v) && v.every(isCoord));
const isPairs = (v: unknown): v is Pair[] => isList(v) && v.every(isPair);
const isParamValue = (v: unknown): v is ParamValue =>
  v === null || typeof v === "boolean" || isNum(v) || isText(v);
const isParams = (v: unknown): v is Record<string, ParamValue> =>
  isObj(v) && Object.keys(v).length <= 50 && Object.values(v).every(isParamValue);
const isHistoryPoint = (v: unknown): boolean => isList(v) && v.length === 2 && isCount(v[0]) && isEnergy(v[1]);
const isHistory = (v: unknown): boolean => Array.isArray(v) && v.length <= MAX_HISTORY && v.every(isHistoryPoint);

/** Rows all of one alphabet's width (3 moves in 2D, 5 in 3D), each a probability vector. */
function isMatrix(v: unknown): boolean {
  if (!isList(v)) return false;
  const width = (v[0] as unknown[] | undefined)?.length;
  return (width === 3 || width === 5 || v.length === 0) && v.every((row) => isList(row) && row.length === width && row.every(isProb));
}

/** Coordinates of a fold with `beads` beads (or none) and contacts that index into it. */
function isGeometry(coords: unknown, contacts: unknown, beads: number | null): boolean {
  if (beads === null) return coords === null && isPairs(contacts) && contacts.length === 0;
  return isCoords(coords) && coords !== null && coords.length === beads && isPairs(contacts) && contacts.every(([i, j]) => i < beads && j < beads);
}

function toFrame(f: unknown): Frame | null {
  if (!isObj(f)) return null;
  const ok =
    isCount(f.iter) &&
    isCount(f.evals) &&
    isCount(f.budget) &&
    isEnergy(f.best_energy) &&
    (f.best_moves === null || isMoves(f.best_moves)) &&
    isMoves(f.current_best_moves) &&
    isProb(f.diversity) &&
    isMatrix(f.matrix) &&
    isGeometry(f.best_coords, f.best_contacts, typeof f.best_moves === "string" ? f.best_moves.length + 2 : null);
  if (!ok) return null;
  return {
    iter: f.iter as number,
    evals: f.evals as number,
    budget: f.budget as number,
    best_energy: f.best_energy as number | null,
    best_moves: f.best_moves as string | null,
    current_best_moves: f.current_best_moves as string,
    diversity: f.diversity as number,
    matrix: f.matrix as number[][],
    best_coords: f.best_coords as Coord[] | null,
    best_contacts: f.best_contacts as Pair[],
  };
}

function toResult(r: unknown): RunResultData | null {
  if (!isObj(r)) return null;
  const ok =
    typeof r.algorithm === "string" &&
    ALGORITHMS.includes(r.algorithm) &&
    typeof r.sequence === "string" &&
    SEQUENCE.test(r.sequence) &&
    (r.dim === 2 || r.dim === 3) &&
    isCount(r.seed) &&
    isParams(r.params) &&
    (r.best_moves === null || isMoves(r.best_moves)) &&
    isEnergy(r.best_energy) &&
    isCount(r.evaluations) &&
    (r.evals_to_best === null || isCount(r.evals_to_best)) &&
    isNum(r.runtime_s) &&
    r.runtime_s >= 0 &&
    isHistory(r.history) &&
    typeof r.cancelled === "boolean" &&
    typeof r.target_reached === "boolean" &&
    isObj(r.stats) &&
    Object.values(r.stats).every(isNum) &&
    isGeometry(r.coords, r.contacts, r.best_moves === null ? null : r.sequence.length) &&
    (r.dim === 3 || r.coords === null || (r.coords as Coord[]).every((c) => c[2] === 0)) &&
    typeof r.valid === "boolean" &&
    (r.n_contacts === null || isCount(r.n_contacts));
  if (!ok) return null;
  return {
    algorithm: r.algorithm as Algorithm,
    sequence: r.sequence as string,
    dim: r.dim as 2 | 3,
    seed: r.seed as number,
    params: r.params as Record<string, ParamValue>,
    best_moves: r.best_moves as string | null,
    best_energy: r.best_energy as number | null,
    evaluations: r.evaluations as number,
    evals_to_best: r.evals_to_best as number | null,
    runtime_s: r.runtime_s as number,
    history: r.history as [number, number | null][],
    cancelled: r.cancelled as boolean,
    target_reached: r.target_reached as boolean,
    stats: r.stats as Record<string, number>,
    coords: r.coords as Coord[] | null,
    contacts: r.contacts as Pair[],
    valid: r.valid as boolean,
    n_contacts: r.n_contacts as number | null,
  };
}

/** Python's JSON text {type, <key>: payload} -> payload, or undefined if malformed. */
function unwrap(json: unknown, type: "frame" | "result"): unknown {
  if (typeof json !== "string") return undefined;
  const parsed: unknown = JSON.parse(json);
  if (!isObj(parsed) || parsed.type !== type) return undefined;
  return type === "frame" ? parsed.frame : parsed.result;
}

function parse(data: unknown): WorkerMessage | null {
  if (!isObj(data)) return null;
  switch (data.type) {
    case "loading":
      return isText(data.stage) ? { type: "loading", stage: data.stage } : null;
    case "ready":
      return { type: "ready" };
    case "error":
      if (!isText(data.message) || !(data.id === null || isText(data.id))) return null;
      return { type: "error", id: data.id, message: data.message };
    case "frame": {
      if (!isText(data.id)) return null;
      const frame = toFrame(unwrap(data.json, "frame"));
      return frame ? { type: "frame", id: data.id, frame } : null;
    }
    case "result": {
      if (!isText(data.id)) return null;
      const result = toResult(unwrap(data.json, "result"));
      return result ? { type: "result", id: data.id, result } : null;
    }
    default:
      return null;
  }
}

/** Checked message from a worker, or null (malformed messages are dropped). Never throws. */
export function parseWorkerMessage(data: unknown): WorkerMessage | null {
  try {
    return parse(data);
  } catch {
    return null; // e.g. JSON.parse on malformed text
  }
}

/**
 * Text to show for an error thrown in the worker. hpfold.webjob raises ValueError with fixed,
 * plain messages; only that text is shown. Anything else (internal errors, tracebacks) becomes
 * a fixed message, so no internals reach the page.
 */
export function errorMessage(err: unknown): string {
  const text = err instanceof Error ? err.message : isObj(err) ? err.message : undefined;
  if (typeof text !== "string") return GENERIC_ERROR;
  // only the final exception counts (a chained traceback lists earlier ones first)
  const lastLine = text.trimEnd().split("\n").at(-1) ?? "";
  const match = /^ValueError: (.+)$/.exec(lastLine);
  const message = match?.[1]?.trim();
  return message && message.length <= 300 ? message : GENERIC_ERROR;
}
