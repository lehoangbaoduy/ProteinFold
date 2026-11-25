// Input validation for the live solver (PRD FR-L1, FR-L2; edge cases E1-E4).
import fc from "fast-check";
import { describe, expect, it } from "vitest";

import { budgetFor, DEFAULT_LIMITS, parseSequence } from "./input";

describe("parseSequence", () => {
  it("accepts a plain H/P sequence", () => {
    expect(parseSequence("HPPHHP")).toEqual({ ok: true, sequence: "HPPHHP" });
  });

  it("normalizes case and whitespace (E2)", () => {
    expect(parseSequence("  hp ph\nH\tp ")).toEqual({ ok: true, sequence: "HPPHHP" });
  });

  it.each(["", "   ", "\n\t"])("rejects empty input %j (E1)", (text) => {
    const r = parseSequence(text);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toMatch(/enter/i);
  });

  it.each(["HPX", "HP1", "<script>alert(1)</script>", "H-P", "HPÉ"])("rejects invalid characters %j (E1)", (text) => {
    const r = parseSequence(text);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toMatch(/only H and P/);
  });

  it("rejects sequences shorter than 3 (E3)", () => {
    const r = parseSequence("HP");
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toMatch(/at least 3/);
  });

  it("rejects sequences longer than the web cap with an explanation (E4)", () => {
    const r = parseSequence("H".repeat(101));
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toMatch(/at most 100/);
    expect(parseSequence("H".repeat(100)).ok).toBe(true);
  });

  it("honours custom limits", () => {
    expect(parseSequence("HPHPH", { minLength: 3, maxLength: 4 }).ok).toBe(false);
  });

  it("never throws and only ever accepts clean H/P strings within the limits (fuzz)", () => {
    fc.assert(
      fc.property(fc.string({ maxLength: 300 }), (text) => {
        const r = parseSequence(text);
        if (r.ok) {
          expect(r.sequence).toMatch(/^[HP]+$/);
          expect(r.sequence.length).toBeGreaterThanOrEqual(DEFAULT_LIMITS.minLength);
          expect(r.sequence.length).toBeLessThanOrEqual(DEFAULT_LIMITS.maxLength);
        } else {
          expect(typeof r.error).toBe("string");
        }
      }),
      { numRuns: 2000 },
    );
  });
});

describe("budgetFor", () => {
  const standard = { name: "Standard", c: 5, min: 5000 };

  it("is max(min, ceil(c * n^2)) like the Python budget rule", () => {
    expect(budgetFor(20, standard)).toBe(5000);
    expect(budgetFor(100, standard)).toBe(50000);
    expect(budgetFor(3, { name: "x", c: 0.5, min: 0 })).toBe(5); // 4.5 rounds up
  });
});

describe("parseSequence hardening (review run 33)", () => {
  it("rejects oversized input before normalizing it", () => {
    const r = parseSequence(" ".repeat(20_000) + "HPH");
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toMatch(/too long/i);
  });

  it.each([
    ["full-width", "ｈｐｈ"],
    ["ligature", "ﬀHPH"],
    ["dotless i", "ıHPH"],
    ["zero-width space", "H\u200BPH"],
    ["RTL override", "\u202EHPH"],
    ["combining dot", "H\u0307PH"],
    ["lone surrogate", "HP\uD800H"],
  ])("rejects Unicode look-alikes: %s", (_label, text) => {
    const r = parseSequence(text);
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toMatch(/only H and P/);
  });

  it("strips BOM and no-break space as whitespace (deliberate: pasted text)", () => {
    expect(parseSequence("\uFEFFHP\u00A0H")).toEqual({ ok: true, sequence: "HPH" });
  });

  it("only ever accepts clean H/P strings for full-Unicode input (fuzz)", () => {
    fc.assert(
      fc.property(fc.string({ unit: "binary", maxLength: 200 }), (text) => {
        const r = parseSequence(text);
        if (r.ok) expect(r.sequence).toMatch(/^[HP]{3,100}$/);
        else expect(r.error.length).toBeGreaterThan(0);
      }),
      { numRuns: 2000 },
    );
  });
});

describe("budgetFor (property)", () => {
  it("matches the reference formula for all n and presets", () => {
    fc.assert(
      fc.property(fc.integer({ min: 3, max: 100 }), fc.integer({ min: 1, max: 50 }), fc.integer({ min: 0, max: 100_000 }), (n, c, min) => {
        const b = budgetFor(n, { name: "p", c, min });
        expect(b).toBe(Math.max(min, c * n * n));
        expect(b).toBeGreaterThanOrEqual(min);
      }),
    );
  });
});
