import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "./", // relative asset paths: works from GitHub Pages, Vercel or a local folder
  build: { target: "es2022", sourcemap: true },
  worker: { format: "es" },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
