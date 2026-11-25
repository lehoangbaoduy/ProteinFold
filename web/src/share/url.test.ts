// Share-link codec (PRD FR-L6). URL parameters are untrusted input (CONST-SEC-005).
import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { decodeShare, encodeShare, type ShareState } from "./url";

const example: ShareState = {
  sequence: "HPHPPHHPHPPHPHHPPHPH",
  dims: [2, 3],
  algorithms: ["GA", "ACO", "PSO"],
  effort: "Standard",
  seed: 42,
};

describe("share codec", () => {
  it("round-trips a state", () => {
    const query = encodeShare(example);
    expect(decodeShare(query)).toEqual({ ok: true, state: example });
  });

  it("accepts a leading '?' and ignores unknown keys", () => {
    const query = "?" + encodeShare(example) + "&utm_source=x&evil=%3Cscript%3E";
    expect(decodeShare(query)).toEqual({ ok: true, state: example });
  });

  it.each([
    ["missing sequence", "dims=2&algs=GA&effort=Quick&seed=1"],
    ["bad sequence", "seq=HPX&dims=2&algs=GA&effort=Quick&seed=1"],
    ["script in sequence", "seq=%3Cscript%3E&dims=2&algs=GA&effort=Quick&seed=1"],
    ["bad dim", "seq=HPHP&dims=4&algs=GA&effort=Quick&seed=1"],
    ["no algorithm", "seq=HPHP&dims=2&algs=&effort=Quick&seed=1"],
    ["unknown algorithm", "seq=HPHP&dims=2&algs=SA&effort=Quick&seed=1"],
    ["bad effort", "seq=HPHP&dims=2&algs=GA&effort=Max&seed=1"],
    ["negative seed", "seq=HPHP&dims=2&algs=GA&effort=Quick&seed=-1"],
    ["huge seed", "seq=HPHP&dims=2&algs=GA&effort=Quick&seed=99999999999"],
    ["non-integer seed", "seq=HPHP&dims=2&algs=GA&effort=Quick&seed=1e3"],
    ["too long", `seq=${"H".repeat(101)}&dims=2&algs=GA&effort=Quick&seed=1`],
  ])("rejects %s", (_label, query) => {
    const r = decodeShare(query);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(typeof r.error).toBe("string");
  });

  it("deduplicates and orders dims and algorithms canonically", () => {
    const r = decodeShare("seq=HPHP&dims=3,2,3&algs=PSO,GA,GA&effort=Quick&seed=0");
    expect(r).toEqual({ ok: true, state: { sequence: "HPHP", dims: [2, 3], algorithms: ["GA", "PSO"], effort: "Quick", seed: 0 } });
  });

  it("never throws on arbitrary query strings (fuzz)", () => {
    fc.assert(
      fc.property(fc.string({ maxLength: 500 }), (query) => {
        const r = decodeShare(query);
        expect(typeof r.ok).toBe("boolean");
      }),
      { numRuns: 2000 },
    );
  });

  it("round-trips every valid state (property)", () => {
    const state = fc.record({
      sequence: fc.stringMatching(/^[HP]{3,100}$/),
      dims: fc.subarray([2, 3] as const, { minLength: 1 }),
      algorithms: fc.subarray(["GA", "ACO", "PSO"] as const, { minLength: 1 }),
      effort: fc.constantFrom("Quick", "Standard", "Thorough"),
      seed: fc.integer({ min: 0, max: 2 ** 31 - 1 }),
    });
    fc.assert(
      fc.property(state, (s) => {
        const decoded = decodeShare(encodeShare(s));
        expect(decoded).toEqual({ ok: true, state: s });
      }),
      { numRuns: 500 },
    );
  });
});

describe("share codec hardening (review run 33)", () => {
  const base = { seq: "HPHP", dims: "2", algs: "GA", effort: "Quick", seed: "1" };
  const q = (over: Record<string, string>) => new URLSearchParams({ ...base, ...over }).toString();

  it.each([
    ["missing sequence", "dims=2&algs=GA&effort=Quick&seed=1", /sequence/i],
    ["bad sequence", q({ seq: "HPX" }), /sequence/i],
    ["whitespace in link sequence", q({ seq: "HP HP" }), /sequence/i],
    ["lowercase link sequence", q({ seq: "hphp" }), /sequence/i],
    ["too short", q({ seq: "HP" }), /sequence/i],
    ["too long", q({ seq: "H".repeat(101) }), /sequence/i],
    ["bad dim", q({ dims: "4" }), /lattice/i],
    ["mixed known/unknown dim", q({ dims: "2,4" }), /lattice/i],
    ["empty dim item", q({ dims: "2,,3" }), /lattice/i],
    ["padded dim", q({ dims: "02" }), /lattice/i],
    ["mixed known/unknown algorithm", q({ algs: "GA,SA" }), /algorithm/i],
    ["no algorithm", q({ algs: "" }), /algorithm/i],
    ["lowercase effort", q({ effort: "quick" }), /effort/i],
    ["prototype effort", q({ effort: "__proto__" }), /effort/i],
    ["seed above 2^31-1", q({ seed: "2147483648" }), /seed/i],
    ["seed with leading zeros", q({ seed: "007" }), /seed/i],
    ["negative seed", q({ seed: "-1" }), /seed/i],
    ["exponent seed", q({ seed: "1e3" }), /seed/i],
    ["duplicate key", q({}) + "&seq=PPPP", /more than once/i],
    ["oversized link", q({}) + "&pad=" + "x".repeat(3000), /too long/i],
  ])("rejects %s with a specific message", (_label, query, message) => {
    const r = decodeShare(query);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toMatch(message);
  });

  it("accepts the boundaries: length 3 and 100, seed 0 and 2^31-1", () => {
    expect(decodeShare(q({ seq: "HPH", seed: "0" })).ok).toBe(true);
    expect(decodeShare(q({ seq: "P".repeat(100), seed: "2147483647" })).ok).toBe(true);
  });

  it("accepts an encoded comma in list values", () => {
    const r = decodeShare("seq=HPHP&dims=2%2C3&algs=GA&effort=Quick&seed=1");
    expect(r.ok && r.state.dims).toEqual([2, 3]);
  });

  it("only ever returns canonical, in-range states (full-Unicode fuzz)", () => {
    fc.assert(
      fc.property(fc.string({ unit: "binary", maxLength: 300 }), (query) => {
        const r = decodeShare(query);
        if (r.ok) {
          expect(r.state.sequence).toMatch(/^[HP]{3,100}$/);
          expect(Number.isInteger(r.state.seed)).toBe(true);
          expect(decodeShare(encodeShare(r.state))).toEqual(r);
        } else {
          expect(r.error.length).toBeGreaterThan(0);
        }
      }),
      { numRuns: 2000 },
    );
  });
});
