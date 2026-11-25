// Live-solver options (PRD FR-L1 presets/random, FR-L2, FR-E6, FR-L5 badge; PFOLD-DOM-012).
//
// The page's defaults come from web/public/data/defaults.json (the tuned parameters of the
// paper, written by hpfold_tools.export_web) and the presets from benchmarks.json (cited
// literature instances). Both are parsed and checked here before use.

import type { QueuedJob } from "../worker/pool";
import type { Algorithm, ParamValue } from "../worker/protocol";
import { budgetFor, parseSequence, type EffortPreset } from "./input";

export type Dim = 2 | 3;
export type EffortName = "Quick" | "Standard" | "Thorough";
export type Params = Record<string, ParamValue>;

export interface Defaults {
  source_config: string;
  algorithms: Record<Algorithm, Params>;
  effort: EffortPreset[];
  limits: { min_length: number; max_length: number };
}

export interface Source {
  citation: string;
  url: string;
}

export interface Benchmark {
  id: string;
  dim: Dim;
  length: number;
  sequence: string;
  best_known_energy: number;
  notes: string;
  sources: Source[];
}

export interface SolverState {
  sequence: string;
  dims: Dim[];
  algorithms: Algorithm[];
  effort: EffortName;
  seed: number;
  /** Stop a benchmark run once it reaches the best-known energy (default true). */
  stopAtBestKnown?: boolean;
}

export type Overrides = Partial<Record<Algorithm, Params>>;
export type BuildResult = { ok: true; jobs: QueuedJob[] } | { ok: false; error: string };
export type Badge = { kind: "reached" } | { kind: "better" } | { kind: "gap"; gap: number } | { kind: "na" };

export const ALGORITHM_ORDER: readonly Algorithm[] = ["GA", "ACO", "PSO"];
const DIM_ORDER: readonly Dim[] = [2, 3];
const EFFORT_NAMES: readonly string[] = ["Quick", "Standard", "Thorough"];
const MAX_SEED = 2 ** 31 - 1;
const MINUS = "−";
const PARAM_KEY = /^[a-z][a-z0-9_]{0,31}$/;
const BENCHMARK_ID = /^[A-Za-z0-9-]{1,20}$/;

// Mirrors of hpfold.webjob's checks, so an invalid Advanced-panel value fails here, before any
// job starts, instead of in one worker while sibling jobs are already running.
export const MAX_BUDGET = 200_000;
export const MAX_BROWSER_LENGTH = 100;
const MAX_POPULATION = 1000; // pop_size / swarm_size / n_ants
const MAX_LOOP_PARAM = 1000; // ACO backtrack_limit / max_restarts
const MAX_ABS_NUMBER = 1e6;
const POPULATION_PARAMS = new Set(["pop_size", "swarm_size", "n_ants"]);
const LOOP_PARAMS = new Set(["backtrack_limit", "max_restarts"]);
// Integer fields of the Params dataclasses (JSON cannot tell 1.0 from 1).
const INT_PARAMS = new Set(["pop_size", "tournament_k", "elite", "n_ants", "backtrack_limit", "max_restarts", "swarm_size"]);
// Allowed values of the categorical fields (fitness.STRATEGIES, ga.CROSSOVERS, pso.TOPOLOGIES, aco.DEPOSITS).
const CHOICES = new Map<string, readonly string[]>([
  ["strategy", ["penalty", "repair"]],
  ["crossover", ["one_point", "two_point", "uniform"]],
  ["topology", ["global", "ring"]],
  ["deposit", ["iteration_best", "global_best", "both"]],
]);

// Pyodide throughput at n = 100 (decision D21), in evaluations per second; cost per
// evaluation grows linearly with n, so rate(n) = K / n.
const THROUGHPUT_K: Readonly<Record<Algorithm, number>> = { GA: 360_000, ACO: 120_000, PSO: 410_000 };

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const isInt = (v: unknown): v is number => Number.isInteger(v);
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isShortText = (v: unknown, max: number): v is string => typeof v === "string" && v.length <= max;
const isParamValue = (v: unknown): v is ParamValue => v === null || typeof v === "boolean" || isNum(v) || isShortText(v, 32);

function isParams(v: unknown): v is Params {
  if (!isObj(v)) return false;
  const entries = Object.entries(v);
  return entries.length <= 30 && entries.every(([k, value]) => PARAM_KEY.test(k) && isParamValue(value));
}

function paramsOf(algorithms: Obj | Record<Algorithm, Params>, algorithm: Algorithm): unknown {
  switch (algorithm) {
    case "GA":
      return algorithms.GA;
    case "ACO":
      return algorithms.ACO;
    case "PSO":
      return algorithms.PSO;
  }
}

function isEffort(v: unknown): v is EffortPreset {
  return isObj(v) && typeof v.name === "string" && EFFORT_NAMES.includes(v.name) && isNum(v.c) && v.c > 0 && isInt(v.min) && v.min >= 0;
}

/** Checked defaults.json, or null. Never throws. */
export function parseDefaults(json: unknown): Defaults | null {
  if (!isObj(json) || !isShortText(json.source_config, 100) || !isObj(json.algorithms)) return null;
  const algorithms = json.algorithms;
  if (!ALGORITHM_ORDER.every((a) => isParams(paramsOf(algorithms, a)))) return null;
  const effort = json.effort;
  // exactly the three presets, each once
  if (!Array.isArray(effort) || effort.length !== EFFORT_NAMES.length || !effort.every(isEffort)) return null;
  if (new Set(effort.map((e) => e.name)).size !== EFFORT_NAMES.length) return null;
  const limits = json.limits;
  if (!isObj(limits) || !isInt(limits.min_length) || !isInt(limits.max_length)) return null;
  if (limits.min_length < 3 || limits.min_length > limits.max_length || limits.max_length > MAX_BROWSER_LENGTH) return null;
  const maxLength = limits.max_length;
  if (effort.some((e) => budgetFor(maxLength, e) > MAX_BUDGET)) return null; // the worker would refuse it
  return {
    source_config: json.source_config,
    algorithms: { GA: algorithms.GA as Params, ACO: algorithms.ACO as Params, PSO: algorithms.PSO as Params },
    effort: effort.map((e) => ({ name: e.name, c: e.c, min: e.min })),
    limits: { min_length: limits.min_length, max_length: limits.max_length },
  };
}

function isSource(v: unknown): v is Source {
  if (!isObj(v) || !isShortText(v.citation, 500) || typeof v.url !== "string") return false;
  try {
    return new URL(v.url).protocol === "https:";
  } catch {
    return false;
  }
}

function toBenchmark(v: unknown): Benchmark | null {
  if (!isObj(v)) return null;
  const ok =
    typeof v.id === "string" &&
    BENCHMARK_ID.test(v.id) &&
    (v.dim === 2 || v.dim === 3) &&
    typeof v.sequence === "string" &&
    /^[HP]{1,200}$/.test(v.sequence) &&
    v.length === v.sequence.length &&
    isInt(v.best_known_energy) &&
    v.best_known_energy <= 0 &&
    isShortText(v.notes, 2000) &&
    Array.isArray(v.sources) &&
    v.sources.length >= 1 &&
    v.sources.length <= 10 &&
    v.sources.every(isSource);
  if (!ok) return null;
  return {
    id: v.id as string,
    dim: v.dim as Dim,
    length: v.length as number,
    sequence: v.sequence as string,
    best_known_energy: v.best_known_energy as number,
    notes: v.notes as string,
    sources: (v.sources as Source[]).map((s) => ({ citation: s.citation, url: s.url })),
  };
}

/** Checked benchmarks.json instances, or null if any instance is malformed. Never throws. */
export function parseBenchmarks(json: unknown): Benchmark[] | null {
  if (!isObj(json) || !Array.isArray(json.instances) || json.instances.length > 500) return null;
  const parsed = json.instances.map(toBenchmark);
  return parsed.every((b): b is Benchmark => b !== null) ? parsed : null;
}

export function benchmarkFor(sequence: string, dim: Dim, benchmarks: readonly Benchmark[]): Benchmark | undefined {
  return benchmarks.find((b) => b.dim === dim && b.sequence === sequence);
}

/** "Thachuk C, Shmygelska A, Hoos HH. Title... 2007;..." -> "Thachuk et al. 2007". */
export function sourceLabel(source: Source): string {
  const dot = source.citation.indexOf(". ");
  const authors = (dot >= 0 ? source.citation.slice(0, dot) : source.citation).split(", ");
  const surnames = authors.map((a) => a.trim().split(" ")[0] ?? "").filter((s) => s.length > 0);
  const year = /\b(?:19|20)\d{2}\b/.exec(dot >= 0 ? source.citation.slice(dot) : source.citation)?.[0] ?? "";
  const first = surnames[0] ?? "";
  const who = surnames.length >= 3 ? `${first} et al.` : surnames.length === 2 ? `${first} & ${surnames[1] ?? ""}` : first;
  return year ? `${who} ${year}` : who;
}

export function formatEnergy(energy: number | null): string {
  if (energy === null) return "NA";
  return energy < 0 ? `${MINUS}${String(-energy)}` : String(energy);
}

export function presetLabel(b: Benchmark): string {
  const source = b.sources[0];
  const cite = source ? ` · ${sourceLabel(source)}` : "";
  return `${b.id} · ${String(b.dim)}D · n = ${String(b.length)} · best known ${formatEnergy(b.best_known_energy)}${cite}`;
}

/** Result badge against a benchmark's best-known energy (FR-L5). */
export function gapBadge(energy: number | null, bestKnown: number): Badge {
  if (energy === null) return { kind: "na" };
  if (energy === bestKnown) return { kind: "reached" };
  if (energy < bestKnown) return { kind: "better" };
  return { kind: "gap", gap: energy - bestKnown };
}

export function randomSequence(length: number, rng: () => number): string {
  if (!Number.isInteger(length) || length < 3 || length > 100) throw new RangeError("length must be an integer from 3 to 100");
  let out = "";
  for (let i = 0; i < length; i++) out += rng() < 0.5 ? "H" : "P";
  return out;
}

export function randomSeed(rng: () => number): number {
  return Math.min(MAX_SEED, Math.floor(rng() * 2 ** 31));
}

/** Rough wall-clock estimate in the browser (one worker per job). */
export function estimateSeconds(n: number, budget: number, algorithm: Algorithm): number {
  const k = algorithm === "GA" ? THROUGHPUT_K.GA : algorithm === "ACO" ? THROUGHPUT_K.ACO : THROUGHPUT_K.PSO;
  return (budget * n) / k;
}

/** Error text for an override value, or null if hpfold.webjob would accept it. */
function checkOverride(algorithm: Algorithm, key: string, base: ParamValue | undefined, value: unknown): string | null {
  const wrongType = `Parameter ${key} for ${algorithm} has the wrong type.`;
  if (base === undefined || !isParamValue(value)) return wrongType;
  if (typeof base === "string") {
    if (typeof value !== "string") return wrongType;
    const allowed = CHOICES.get(key);
    return allowed && !allowed.includes(value) ? `Parameter ${key} for ${algorithm} must be one of ${allowed.join(", ")}.` : null;
  }
  if (typeof base === "boolean") return typeof value === "boolean" ? null : wrongType;
  // numbers; null only where the default is null (the GA mutation rate pm = automatic)
  if (value === null) return base === null ? null : wrongType;
  if (typeof value !== "number" || (INT_PARAMS.has(key) && !Number.isInteger(value))) return wrongType;
  const limit = POPULATION_PARAMS.has(key) ? MAX_POPULATION : LOOP_PARAMS.has(key) ? MAX_LOOP_PARAM : MAX_ABS_NUMBER;
  return Math.abs(value) > limit ? `Parameter ${key} for ${algorithm} is too large.` : null;
}

function mergeParams(algorithm: Algorithm, base: Params, override: Params | undefined): Params | string {
  const merged = new Map(Object.entries(base));
  for (const [key, value] of Object.entries(override ?? {})) {
    if (!PARAM_KEY.test(key) || !merged.has(key)) {
      return PARAM_KEY.test(key) ? `Unknown parameter ${key} for ${algorithm}.` : `Unknown parameter for ${algorithm}.`;
    }
    const problem = checkOverride(algorithm, key, merged.get(key), value);
    if (problem !== null) return problem;
    merged.set(key, value);
  }
  return Object.fromEntries<ParamValue>(merged);
}

/** One job per (lattice, algorithm), in canonical order, ready for SolverPool.run. */
export function buildJobs(state: SolverState, defaults: Defaults, benchmarks: readonly Benchmark[], overrides: Overrides = {}): BuildResult {
  const parsed = parseSequence(state.sequence, { minLength: defaults.limits.min_length, maxLength: defaults.limits.max_length });
  if (!parsed.ok) return { ok: false, error: parsed.error };
  const dims = DIM_ORDER.filter((d) => state.dims.includes(d));
  if (dims.length === 0) return { ok: false, error: "Choose at least one lattice." };
  const algorithms = ALGORITHM_ORDER.filter((a) => state.algorithms.includes(a));
  if (algorithms.length === 0) return { ok: false, error: "Choose at least one algorithm." };
  const preset = defaults.effort.find((e) => e.name === state.effort);
  if (!preset) return { ok: false, error: "Unknown effort level." };
  if (!Number.isInteger(state.seed) || state.seed < 0 || state.seed > MAX_SEED) return { ok: false, error: "The seed must be an integer from 0 to 2147483647." };

  const params = new Map<Algorithm, Params>();
  for (const algorithm of algorithms) {
    const merged = mergeParams(algorithm, paramsOf(defaults.algorithms, algorithm) as Params, paramsOf(overrides, algorithm) as Params | undefined);
    if (typeof merged === "string") return { ok: false, error: merged };
    params.set(algorithm, merged);
  }
  const sequence = parsed.sequence;
  const budget = budgetFor(sequence.length, preset);
  const jobs: QueuedJob[] = [];
  for (const dim of dims) {
    const known = benchmarkFor(sequence, dim, benchmarks)?.best_known_energy ?? null;
    const target = state.stopAtBestKnown === false ? null : known;
    for (const algorithm of algorithms) {
      jobs.push({
        id: `${algorithm}-${String(dim)}d`,
        job: { algorithm, sequence, dim, seed: state.seed, budget, params: params.get(algorithm) ?? {}, target },
      });
    }
  }
  return { ok: true, jobs };
}
