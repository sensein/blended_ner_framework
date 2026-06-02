import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { Type } from "typebox";

const execFileAsync = promisify(execFile);

export default {
  name: "parse_pdf",
  description:
    "Parse a PDF into model-token-aware chunk files by running scripts/ingest_chunk.py via uv. The Python script tries local Grobid first, falls back to PyMuPDF4LLM, and chunks with the Hugging Face tokenizer for the provided modelId.",
  parameters: Type.Object({
    pdf: Type.String({ description: "Path to the input PDF." }),
    modelId: Type.String({
      description:
        "Hugging Face model identifier whose tokenizer should be used for token-aware chunking, e.g. bert-base-uncased or any accessible LLM tokenizer.",
    }),
    outDir: Type.Optional(
      Type.String({ description: "Directory where chunk_NNN.txt files should be written." }),
    ),
    grobidUrl: Type.Optional(
      Type.String({ description: "Base URL of the local Grobid service." }),
    ),
    grobidTimeout: Type.Optional(
      Type.Number({ description: "Timeout in seconds for Grobid parse requests." }),
    ),
    maxTokens: Type.Optional(
      Type.Number({
        description:
          "Optional maximum tokenizer tokens per chunk. If omitted, the script derives a limit from tokenizer.model_max_length.",
      }),
    ),
  }),
  async execute(_toolCallId: string, params: any, signal?: AbortSignal) {
    const args = ["run", "scripts/ingest_chunk.py", params.pdf, "--model-id", params.modelId];

    if (params.outDir) args.push("--out-dir", params.outDir);
    if (params.grobidUrl) args.push("--grobid-url", params.grobidUrl);
    if (params.grobidTimeout !== undefined) args.push("--grobid-timeout", String(params.grobidTimeout));
    if (params.maxTokens !== undefined) args.push("--max-tokens", String(params.maxTokens));

    try {
      const { stdout, stderr } = await execFileAsync("uv", args, {
        signal,
        maxBuffer: 1024 * 1024,
      });
      const chunkPaths = stdout.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
      const summaryLine =
        stderr
          .split(/\r?\n/)
          .filter((line) => line.includes("wrote") || line.includes("Wrote"))
          .pop() ?? "parse_pdf completed";

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
          modelId: params.modelId,
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
