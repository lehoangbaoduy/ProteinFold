// Live-solver options (PFOLD-DOM-012). The tests read the REAL generated data files
// (web/public/data, written by hpfold_tools.export_web) - no mock data (T5.13).
import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { DEFAULT_LIMITS } from "./input";
import {
  benchmarkFor,
  buildJobs,
  estimateSeconds,
  formatEnergy,
  gapBadge,
  parseBenchmarks,
  parseDefaults,
  presetLabel,
  randomSeed,
  randomSequence,
  sourceLabel,
  type Benchmark,
  type Defaults,
} from "./options";

import benchmarksJson from "../../public/data/benchmarks.json";
import defaultsJson from "../../public/data/defaults.json";

// fresh deep copies, so a test that mutates one cannot affect another
const readJson = (name: "defaults.json" | "benchmarks.json"): unknown =>
  structuredClone(name === "defaults.json" ? defaultsJson : benchmarksJson);

function load(): { defaults: Defaults; benchmarks: Benchmark[] } {
  const defaults = parseDefaults(readJson("defaults.json"));
  const benchmarks = parseBenchmarks(readJson("benchmarks.json"));
  if (!defaults || !benchmarks) throw new Error("shipped data files do not parse");
  return { defaults, benchmarks };
}

const S1_1 = "HPHPPHHPHPPHPHHPPHPH";

describe("shipped data files", () => {
  it("parse, and match the tuned parameters of the paper (FR-E6)", () => {
    const { defaults, benchmarks } = load();
    expect(defaults.source_config).toBe("main_2d.toml");
    expect(defaults.algorithms.GA.strategy).toBe("repair");
    expect(defaults.algorithms.ACO.rho).toBe(0.1);
    expect(defaults.algorithms.PSO.w).toBe(0.7);
    expect(benchmarks).toHaveLength(21);
    expect(benchmarks.filter((b) => b.dim === 2)).toHaveLength(11);
  });

  it("agree with the limits hard-coded in input.ts (no drift)", () => {
    const { defaults } = load();
    expect(defaults.limits).toEqual({ min_length: DEFAULT_LIMITS.minLength, max_length: DEFAULT_LIMITS.maxLength });
    expect(defaults.effort.map((e) => e.name)).toEqual(["Quick", "Standard", "Thorough"]);
  });
});

describe("parseDefaults / parseBenchmarks", () => {
  it.each([
    ["missing algorithm", (d: Record<string, unknown>) => ({ ...d, algorithms: { GA: {}, ACO: {} } })],
    ["param of the wrong type", (d: Record<string, unknown>) => ({ ...d, algorithms: { ...(d.algorithms as object), GA: { pop_size: [1] } } })],
    ["negative effort constant", (d: Record<string, unknown>) => ({ ...d, effort: [{ name: "Quick", c: -1, min: 5000 }] })],
    ["unknown effort name", (d: Record<string, unknown>) => ({ ...d, effort: [{ name: "Max", c: 1, min: 1 }] })],
    ["bad limits", (d: Record<string, unknown>) => ({ ...d, limits: { min_length: 10, max_length: 5 } })],
  ])("rejects defaults with %s", (_label, mutate) => {
    expect(parseDefaults(mutate(readJson("defaults.json") as Record<string, unknown>))).toBeNull();
  });

  it.each([
    ["non-H/P sequence", { sequence: "HPX" }],
    ["length that disagrees with the sequence", { length: 99 }],
    ["positive best-known energy", { best_known_energy: 3 }],
    ["javascript source url", { sources: [{ citation: "x", url: ["javascript", "alert(1)"].join(":") }] }],
    ["http source url", { sources: [{ citation: "x", url: "http://example.org" }] }],
    ["no sources", { sources: [] }],
    ["bad dim", { dim: 4 }],
  ])("rejects a benchmark with %s", (_label, change) => {
    const raw = readJson("benchmarks.json") as { instances: Record<string, unknown>[] };
    const first = raw.instances[0] ?? {};
    expect(parseBenchmarks({ instances: [{ ...first, ...change }] })).toBeNull();
  });

  it("never throws on arbitrary input (fuzz)", () => {
    fc.assert(
      fc.property(fc.anything(), (v) => {
        parseDefaults(v);
        parseBenchmarks(v);
      }),
      { numRuns: 500 },
    );
  });
});

describe("benchmarks and labels", () => {
  it("finds a benchmark by sequence and lattice", () => {
    const { benchmarks } = load();
    expect(benchmarkFor(S1_1, 2, benchmarks)?.id).toBe("S1-1");
    expect(benchmarkFor(S1_1, 3, benchmarks)).toBeUndefined();
    expect(benchmarkFor("HPH", 2, benchmarks)).toBeUndefined();
  });

  it("shortens citations to author-year labels", () => {
    expect(sourceLabel({ citation: "Thachuk C, Shmygelska A, Hoos HH. A replica exchange ... 2007;8:342.", url: "https://x.org" })).toBe(
      "Thachuk et al. 2007",
    );
    expect(sourceLabel({ citation: "Shmygelska A, Hoos HH. An ant colony ... 2005;6:30.", url: "https://x.org" })).toBe("Shmygelska & Hoos 2005");
    expect(sourceLabel({ citation: "Unger R. Genetic algorithms. 1993.", url: "https://x.org" })).toBe("Unger 1993");
  });

  it("labels presets with id, lattice, length, best-known energy and source", () => {
    const { benchmarks } = load();
    const s11 = benchmarks.find((b) => b.id === "S1-1");
    expect(s11 && presetLabel(s11)).toBe("S1-1 · 2D · n = 20 · best known −9 · Thachuk et al. 2007");
  });

  it("formats energies with a true minus sign and NA for no fold", () => {
    expect(formatEnergy(-9)).toBe("−9");
    expect(formatEnergy(0)).toBe("0");
    expect(formatEnergy(null)).toBe("NA");
  });

  it.each([
    [-9, -9, { kind: "reached" }],
    [-7, -9, { kind: "gap", gap: 2 }],
    [-10, -9, { kind: "better" }],
    [null, -9, { kind: "na" }],
  ])("badge for energy %j vs best known %j", (energy, best, badge) => {
    expect(gapBadge(energy, best)).toEqual(badge);
  });
});

describe("random sequence and seed", () => {
  it("makes H/P sequences of the requested length", () => {
    fc.assert(
      fc.property(fc.integer({ min: 3, max: 100 }), fc.integer(), (n, s) => {
        let x = s;
        const rng = () => ((x = (x * 1103515245 + 12345) % 2147483648) < 0 ? -x : x) / 2147483648;
        const seq = randomSequence(n, rng);
        expect(seq).toHaveLength(n);
        expect(seq).toMatch(/^[HP]+$/);
      }),
    );
  });

  it("rejects lengths outside 3..100", () => {
    expect(() => randomSequence(2, Math.random)).toThrow(RangeError);
    expect(() => randomSequence(101, Math.random)).toThrow(RangeError);
    expect(() => randomSequence(5.5, Math.random)).toThrow(RangeError);
  });

  it("draws seeds in 0..2^31-1, including the ends", () => {
    expect(randomSeed(() => 0)).toBe(0);
    expect(randomSeed(() => 0.9999999999999999)).toBe(2 ** 31 - 1);
  });
});

describe("estimateSeconds", () => {
  it("uses the measured Pyodide throughput (D21): ACO at n = 100 Thorough is about 100 s", () => {
    expect(estimateSeconds(100, 120_000, "ACO")).toBeCloseTo(100, 0);
    expect(estimateSeconds(100, 120_000, "GA")).toBeLessThan(estimateSeconds(100, 120_000, "ACO"));
  });

  it("grows with length and budget", () => {
    expect(estimateSeconds(50, 10_000, "PSO")).toBeLessThan(estimateSeconds(100, 10_000, "PSO"));
    expect(estimateSeconds(50, 10_000, "PSO")).toBeLessThan(estimateSeconds(50, 20_000, "PSO"));
  });
});

describe("buildJobs", () => {
  const state = { sequence: S1_1, dims: [2, 3] as (2 | 3)[], algorithms: ["GA", "ACO", "PSO"] as ("GA" | "ACO" | "PSO")[], effort: "Standard" as const, seed: 42 };

  it("makes one job per algorithm and lattice with tuned params, budget and benchmark target", () => {
    const { defaults, benchmarks } = load();
    const r = buildJobs(state, defaults, benchmarks);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.jobs.map((j) => j.id)).toEqual(["GA-2d", "ACO-2d", "PSO-2d", "GA-3d", "ACO-3d", "PSO-3d"]);
    const ga2 = r.jobs[0]?.job;
    expect(ga2).toEqual({ algorithm: "GA", sequence: S1_1, dim: 2, seed: 42, budget: 5000, params: defaults.algorithms.GA, target: -9 });
    expect(r.jobs[3]?.job.target).toBeNull(); // S1-1 is not a 3D benchmark
  });

  it("uses the effort preset's budget rule", () => {
    const { defaults, benchmarks } = load();
    const r = buildJobs({ ...state, sequence: "H".repeat(100), effort: "Thorough", dims: [2], algorithms: ["ACO"] }, defaults, benchmarks);
    expect(r.ok && r.jobs[0]?.job.budget).toBe(120_000);
  });

  it("applies valid overrides and rejects unknown or mistyped ones", () => {
    const { defaults, benchmarks } = load();
    const ok = buildJobs({ ...state, algorithms: ["GA"], dims: [2] }, defaults, benchmarks, { GA: { pop_size: 50 } });
    expect(ok.ok && ok.jobs[0]?.job.params.pop_size).toBe(50);
    expect(buildJobs(state, defaults, benchmarks, { GA: { evil: 1 } })).toEqual({ ok: false, error: "Unknown parameter evil for GA." });
    expect(buildJobs(state, defaults, benchmarks, { GA: { pop_size: "50" } })).toEqual({
      ok: false,
      error: "Parameter pop_size for GA has the wrong type.",
    });
  });

  it("rejects an invalid sequence or an empty selection", () => {
    const { defaults, benchmarks } = load();
    expect(buildJobs({ ...state, sequence: "HPX" }, defaults, benchmarks).ok).toBe(false);
    expect(buildJobs({ ...state, dims: [] }, defaults, benchmarks).ok).toBe(false);
    expect(buildJobs({ ...state, algorithms: [] }, defaults, benchmarks).ok).toBe(false);
  });
});
