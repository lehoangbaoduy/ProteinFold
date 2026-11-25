// Share-link codec (PRD FR-L6): the run settings live in the query string so a run can be
// reproduced from a link. Query strings are untrusted: decodeShare validates every value,
// ignores unknown keys and never throws.
import { DEFAULT_LIMITS, parseSequence } from "../solver/input";

export type Dim = 2 | 3;
export type AlgorithmName = "GA" | "ACO" | "PSO";
export type EffortName = "Quick" | "Standard" | "Thorough";

export interface ShareState {
  sequence: string;
  dims: Dim[];
  algorithms: AlgorithmName[];
  effort: EffortName;
  seed: number;
}

export type DecodeResult = { ok: true; state: ShareState } | { ok: false; error: string };

const DIMS: readonly Dim[] = [2, 3];
const ALGORITHMS: readonly AlgorithmName[] = ["GA", "ACO", "PSO"];
const EFFORTS: readonly EffortName[] = ["Quick", "Standard", "Thorough"];
const MAX_SEED = 2 ** 31 - 1;
const SEED_PATTERN = /^(0|[1-9]\d{0,9})$/; // canonical decimal: no sign, no leading zeros
const LINK_SEQUENCE = /^[HP]+$/; // links are machine-written: no case or whitespace folding
const MAX_QUERY_LENGTH = 2048; // a valid link is under ~200 characters
const KEYS = ["seq", "dims", "algs", "effort", "seed"] as const;

export function encodeShare(state: ShareState): string {
  const params = new URLSearchParams();
  params.set("seq", state.sequence);
  params.set("dims", state.dims.join(","));
  params.set("algs", state.algorithms.join(","));
  params.set("effort", state.effort);
  params.set("seed", String(state.seed));
  return params.toString();
}

function pickList<T extends string | number>(raw: string | null, allowed: readonly T[]): T[] | null {
  if (!raw) return null;
  const wanted = new Set(raw.split(","));
  const chosen = allowed.filter((value) => wanted.has(String(value)));
  // every requested item must be known, and at least one must remain
  if (chosen.length === 0 || [...wanted].some((w) => !allowed.some((a) => String(a) === w))) return null;
  return chosen; // canonical order, duplicates removed
}

export function decodeShare(query: string): DecodeResult {
  if (query.length > MAX_QUERY_LENGTH) return { ok: false, error: "The link is too long." };
  // URLSearchParams never throws (lone surrogates become U+FFFD).
  const params = new URLSearchParams(query.replace(/^\?/, ""));
  // A key given twice could be read differently by another parser: refuse it.
  const repeated = KEYS.find((key) => params.getAll(key).length > 1);
  if (repeated) return { ok: false, error: `The link sets "${repeated}" more than once.` };

  const rawSequence = params.get("seq") ?? "";
  const parsed = parseSequence(rawSequence, DEFAULT_LIMITS);
  if (!parsed.ok) return { ok: false, error: `Link sequence: ${parsed.error}` };
  if (!LINK_SEQUENCE.test(rawSequence)) {
    return { ok: false, error: "Link sequence: use uppercase H and P without spaces." };
  }

  const dims = pickList(params.get("dims"), DIMS);
  if (!dims) return { ok: false, error: "The link has an invalid lattice choice." };

  const algorithms = pickList(params.get("algs"), ALGORITHMS);
  if (!algorithms) return { ok: false, error: "The link has an invalid algorithm choice." };

  const effort = params.get("effort");
  if (!effort || !EFFORTS.includes(effort as EffortName)) {
    return { ok: false, error: "The link has an invalid effort level." };
  }

  const seedText = params.get("seed") ?? "";
  const seed = SEED_PATTERN.test(seedText) ? Number(seedText) : NaN;
  if (!Number.isInteger(seed) || seed < 0 || seed > MAX_SEED) {
    return { ok: false, error: "The link has an invalid seed." };
  }

  return { ok: true, state: { sequence: parsed.sequence, dims, algorithms, effort: effort as EffortName, seed } };
}
