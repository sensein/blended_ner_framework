import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { Type } from "typebox";

const execFileAsync = promisify(execFile);

export default {
  name: "llm_refinement",
  description:
    "Run an LLM refinement pass over local GLiNER outputs by injecting entities into raw chunks as [Entity](LABEL), sending decorated text to a LiteLLM model, and writing llm_pass1_entities.json.",
  parameters: Type.Object({
    chunksDir: Type.String({ description: "Directory containing chunk_NNN.txt files." }),
    glinerDir: Type.String({ description: "Directory containing GLiNER chunk_NNN.json outputs." }),
    model: Type.Optional(Type.String({ description: "LiteLLM model name, e.g. gpt-4o, gpt-5.5, or claude-3-5-sonnet-latest." })),
    output: Type.Optional(Type.String({ description: "Output JSON path. Defaults to <glinerDir>/llm_pass1_entities.json." })),
    artifactsDir: Type.Optional(Type.String({ description: "Optional directory for decorated/refined markdown and clean text artifacts." })),
    chunkIndex: Type.Optional(Type.Array(Type.Number(), { description: "Optional list of chunk indices to process." })),
    maxChars: Type.Optional(Type.Number({ description: "Optional maximum raw chunk characters to send per chunk; 0 means no truncation." })),
    temperature: Type.Optional(Type.Number({ description: "LLM sampling temperature. Defaults to 0.0." })),
    dryRun: Type.Optional(Type.Boolean({ description: "If true, write decorated artifacts and parse without calling the LLM." })),
  }),
  async execute(_toolCallId: string, params: any, signal?: AbortSignal) {
    const args = [
      "run",
      "scripts/llm_refinement.py",
      "--chunks-dir",
      params.chunksDir,
      "--gliner-dir",
      params.glinerDir,
    ];

    if (params.model) args.push("--model", params.model);
    if (params.output) args.push("--output", params.output);
    if (params.artifactsDir) args.push("--artifacts-dir", params.artifactsDir);
    if (Array.isArray(params.chunkIndex)) {
      for (const index of params.chunkIndex) args.push("--chunk-index", String(index));
    }
    if (params.maxChars !== undefined) args.push("--max-chars", String(params.maxChars));
    if (params.temperature !== undefined) args.push("--temperature", String(params.temperature));
    if (params.dryRun) args.push("--dry-run");

    try {
      const { stdout, stderr } = await execFileAsync("uv", args, {
        signal,
        maxBuffer: 1024 * 1024,
      });

      return {
        content: [
          {
            type: "text",
            text: [stdout.trim(), stderr.trim()].filter(Boolean).join("\n") || "llm_refinement completed",
          },
        ],
        details: {
          command: `uv ${args.join(" ")}`,
          stdout,
          stderr,
        },
      };
    } catch (error: any) {
      return {
        content: [
          {
            type: "text",
            text: `llm_refinement failed with exit code ${error?.code ?? "unknown"}: ${error?.stderr || error?.message}`,
          },
        ],
        details: { command: `uv ${args.join(" ")}`, stderr: error?.stderr, stdout: error?.stdout },
        isError: true,
      };
    }
  },
};
