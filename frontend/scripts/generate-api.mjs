import { existsSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const frontendRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const schemaPath = resolve(frontendRoot, "../openapi.json");
const outputPath = resolve(frontendRoot, "src/api/schema.ts");

if (!existsSync(schemaPath)) {
  console.error(`OpenAPI schema not found at ${schemaPath}. Generate the backend schema first.`);
  process.exit(1);
}

await mkdir(dirname(outputPath), { recursive: true });
const executable = resolve(
  frontendRoot,
  "node_modules",
  ".bin",
  process.platform === "win32" ? "openapi-typescript.cmd" : "openapi-typescript",
);
const result = spawnSync(executable, [schemaPath, "-o", outputPath], {
  cwd: frontendRoot,
  stdio: "inherit",
});

if (result.error) throw result.error;
process.exit(result.status ?? 1);
