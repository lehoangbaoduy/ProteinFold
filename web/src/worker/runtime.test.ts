// Runtime URLs for the solver worker (PFOLD-DOM-011): everything the worker loads must come
// from the page's own site (self-hosted Pyodide, offline mode, no third-party code).
import { describe, expect, it } from "vitest";

import { runtimeUrls } from "./runtime";

describe("runtimeUrls", () => {
  it("resolves the runtime files under the site base", () => {
    expect(runtimeUrls("https://user.github.io/proteinfold/", "https://user.github.io")).toEqual({
      pyodideModule: "https://user.github.io/proteinfold/pyodide/pyodide.mjs",
      pyodideIndex: "https://user.github.io/proteinfold/pyodide/",
      hpfoldZip: "https://user.github.io/proteinfold/py/hpfold.zip",
    });
  });

  it("works for a local preview", () => {
    expect(runtimeUrls("http://localhost:4173/", "http://localhost:4173").hpfoldZip).toBe("http://localhost:4173/py/hpfold.zip");
  });

  it.each([
    ["another site", "https://evil.example/", "https://user.github.io"],
    ["another port", "http://localhost:9999/", "http://localhost:4173"],
    ["javascript url", ["javascript", "alert(1)//"].join(":"), "https://user.github.io"],
    ["data url", "data:text/javascript,alert(1)", "https://user.github.io"],
    ["not a url", "pyodide/", "https://user.github.io"],
  ])("refuses %s", (_label, base, origin) => {
    expect(() => runtimeUrls(base, origin)).toThrow(/this site/);
  });
});
