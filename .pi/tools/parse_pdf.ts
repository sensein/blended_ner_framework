import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { Type } from "typebox";

const execFileAsync = promisify(execFile);

export default {
  name: "parse_pdf",
  description:
    "Parse a PDF into payload-safe chunk files by running scripts/parse_pdf.py via uv. Heavy PDF parsing and file I/O stay inside the Python script; this wrapper returns only lightweight execution metadata.",
  parameters: Type.Object({
    pdf: Type.String({ description: "Path to the input PDF." }),
    outDir: Type.Optional(
      Type.String({ description: "Directory where chunk_NNN.txt files should be written." }),
    ),
    parser: Type.Optional(
      Type.Union([
        Type.Literal("auto"),
        Type.Literal("grobid"),
        Type.Literal("pymupdf4llm"),
      ], { description: "Parser backend to use. Defaults to auto." }),
    ),
    grobidUrl: Type.Optional(
      Type.String({ description: "Base URL of the local Grobid service." }),
    ),
    maxChars: Type.Optional(
      Type.Number({ description: "Maximum characters per chunk." }),
    ),
  }),
  async execute(_toolCallId: string, params: any, signal?: AbortSignal) {
    const args = ["run", "scripts/parse_pdf.py", params.pdf];

    if (params.outDir) args.push("--out-dir", params.outDir);
    if (params.parser) args.push("--parser", params.parser);
    if (params.grobidUrl) args.push("--grobid-url", params.grobidUrl);
    if (params.maxChars !== undefined) args.push("--max-chars", String(params.maxChars));

    try {
      const { stdout, stderr } = await execFileAsync("uv", args, {
        signal,
        maxBuffer: 1024 * 1024,
      });
      const chunkPaths = stdout.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
      const summaryLine = stderr.split(/\r?\n/).find((line) => line.includes("Wrote")) ?? "parse_pdf completed";

      return {
        content: [
          {
            type: "text",
            text: `${summaryLine}\nGenerated ${chunkPaths.length} chunk file(s).`,
          },
        ],
        details: {
          command: `uv ${args.join(" ")}`,
          chunkCount: chunkPaths.length,
          outDir: params.outDir,
        },
      };
    } catch (error: any) {
      return {
        content: [
          {
            type: "text",
            text: `parse_pdf failed with exit code ${error?.code ?? "unknown"}: ${error?.stderr || error?.message}`,
          },
        ],
        details: { command: `uv ${args.join(" ")}`, stderr: error?.stderr, stdout: error?.stdout },
        isError: true,
      };
    }
  },
};
