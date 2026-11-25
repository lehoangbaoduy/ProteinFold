// Lint is part of the test suite: the type-aware rules and the HTML-sink bans
// (eslint.config.js, CONST-SEC-005) must pass before any change counts as green.
import { ESLint } from "eslint";
import { describe, expect, it } from "vitest";

describe("eslint", () => {
  it("reports no errors or warnings in src/", async () => {
    const eslint = new ESLint({ cwd: new URL("..", import.meta.url).pathname });
    const results = await eslint.lintFiles(["src"]);
    const problems = results.flatMap((r) =>
      r.messages.map((m) => `${r.filePath.replace(/^.*\/web\//, "")}:${String(m.line)}:${String(m.column)} ${m.ruleId ?? ""} ${m.message}`),
    );
    expect(problems).toEqual([]);
  }, 60_000);
});
