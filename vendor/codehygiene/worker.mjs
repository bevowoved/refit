import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";
import { unpackNpzBuffer, scoreCode, formatJsonResult } from "./engine.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const weightsPath = process.argv[2] || path.join(__dirname, "weights.npz");

async function main() {
  const buf = fs.readFileSync(weightsPath);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const weights = await unpackNpzBuffer(ab);

  // Signal ready on stderr so stdout remains pure NDJSON responses
  process.stderr.write(`refit-hygiene-worker ready (${buf.length} B weights)\n`);

  const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    if (!line || !line.trim()) continue;
    let req;
    try {
      req = JSON.parse(line);
    } catch (err) {
      process.stdout.write(JSON.stringify({ id: 0, error: String(err) }) + "\n");
      continue;
    }

    const id = req.id ?? 0;
    const content = typeof req.content === "string" ? req.content : "";
    const threshold = typeof req.threshold === "number" ? req.threshold : 0.55;
    const t0 = performance.now();
    try {
      const raw = scoreCode(content, weights, threshold);
      const formatted = formatJsonResult(raw);
      const elapsedUs = Math.round((performance.now() - t0) * 1000);
      process.stdout.write(
        JSON.stringify({
          id,
          score: formatted.score,
          squiggles: formatted.squiggles,
          latency_us: elapsedUs,
        }) + "\n"
      );
    } catch (err) {
      process.stdout.write(
        JSON.stringify({ id, score: 0.5, squiggles: [], error: String(err) }) + "\n"
      );
    }
  }
}

main().catch((err) => {
  process.stderr.write(`worker fatal: ${err?.stack || err}\n`);
  process.exit(1);
});
