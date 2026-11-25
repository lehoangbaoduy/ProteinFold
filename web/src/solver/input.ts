// Live-solver input handling (PRD FR-L1, FR-L2; edge cases E1-E4).
// The text box is untrusted input: this never throws, and error messages are plain
// text meant for textContent (never innerHTML).

export interface Limits {
  minLength: number;
  maxLength: number;
}

export interface EffortPreset {
  name: string;
  c: number;
  min: number;
}

export type ParseResult = { ok: true; sequence: string } | { ok: false; error: string };

// Same values as defaults.json "limits" (exported by hpfold_tools.export_web).
export const DEFAULT_LIMITS: Limits = { minLength: 3, maxLength: 100 };

// Raw-text cap, checked before any normalization so a huge paste costs O(1) (review run 33).
// Generous on purpose: 100 residues plus any reasonable amount of whitespace.
export const MAX_RAW_LENGTH = 10_000;

// JS \s also covers U+FEFF (BOM) and U+00A0 (no-break space): pasted text with those is
// accepted on purpose. Look-alikes (full-width letters, zero-width spaces) are rejected.
const WHITESPACE = /\s+/g;
const HP_ONLY = /^[HP]+$/;

export function parseSequence(text: string, limits: Limits = DEFAULT_LIMITS): ParseResult {
  if (text.length > MAX_RAW_LENGTH) {
    return { ok: false, error: `The input is too long (more than ${MAX_RAW_LENGTH} characters).` };
  }
  const sequence = text.replace(WHITESPACE, "").toUpperCase();
  if (sequence.length === 0) {
    return { ok: false, error: "Enter an HP sequence (letters H and P)." };
  }
  if (!HP_ONLY.test(sequence)) {
    return { ok: false, error: "The sequence may contain only H and P." };
  }
  if (sequence.length < limits.minLength) {
    return { ok: false, error: `The sequence needs at least ${limits.minLength} residues.` };
  }
  if (sequence.length > limits.maxLength) {
    return {
      ok: false,
      error: `The sequence may have at most ${limits.maxLength} residues in the browser (longer runs would take too long).`,
    };
  }
  return { ok: true, sequence };
}

/** Evaluation budget B(n) = max(min, ceil(c * n^2)) — the same rule as the Python experiments. */
export function budgetFor(n: number, preset: EffortPreset): number {
  return Math.max(preset.min, Math.ceil(preset.c * n * n));
}
