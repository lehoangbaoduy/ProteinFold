import js from "@eslint/js";
import security from "eslint-plugin-security";
import tseslint from "typescript-eslint";

const HTML_SINK = "User data must never reach an HTML parser: use textContent / DOM APIs.";

export default tseslint.config(
  js.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  security.configs.recommended,
  {
    languageOptions: { parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname } },
    rules: {
      // HTML-parsing sinks (CONST-SEC-005). Defence in depth: the page CSP also blocks inline script.
      "no-restricted-properties": ["error",
        { property: "innerHTML", message: HTML_SINK },
        { property: "outerHTML", message: HTML_SINK },
        { property: "insertAdjacentHTML", message: HTML_SINK },
        { property: "setHTMLUnsafe", message: HTML_SINK },
        { property: "createContextualFragment", message: HTML_SINK },
        { property: "srcdoc", message: HTML_SINK },
        { object: "document", property: "write", message: HTML_SINK },
        { object: "document", property: "writeln", message: HTML_SINK }],
      "no-restricted-syntax": ["error",
        { selector: "NewExpression[callee.name='DOMParser']", message: HTML_SINK },
        { selector: "ObjectPattern > Property[key.name=/^(innerHTML|outerHTML)$/]", message: HTML_SINK }],
      "@typescript-eslint/restrict-template-expressions": ["error", { allowNumber: true }],
      "no-eval": "error",
      "no-implied-eval": "error",
      "no-new-func": "error",
      "no-script-url": "error",
      "no-new-wrappers": "error",
    },
  },
  { ignores: ["dist/**", "eslint.config.js"] },
);
