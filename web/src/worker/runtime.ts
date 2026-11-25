// Where the solver worker loads its runtime from (PFOLD-DOM-011, FR-L7, offline mode).
// Everything comes from the page's own site: the self-hosted Pyodide (make web-runtime) and
// the hpfold package. A base URL on any other origin or scheme is refused.

export interface RuntimeUrls {
  pyodideModule: string;
  pyodideIndex: string;
  hpfoldZip: string;
}

const NOT_THIS_SITE = "The runtime must come from this site.";

export function runtimeUrls(baseUrl: string, pageOrigin: string): RuntimeUrls {
  let base: URL;
  try {
    base = new URL(baseUrl);
  } catch {
    throw new Error(NOT_THIS_SITE);
  }
  if ((base.protocol !== "https:" && base.protocol !== "http:") || base.origin !== pageOrigin || base.username || base.password) {
    throw new Error(NOT_THIS_SITE);
  }
  if (!base.pathname.endsWith("/")) base.pathname += "/"; // the base is a directory
  return {
    pyodideModule: new URL("pyodide/pyodide.mjs", base).href,
    pyodideIndex: new URL("pyodide/", base).href,
    hpfoldZip: new URL("py/hpfold.zip", base).href,
  };
}
