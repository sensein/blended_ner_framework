import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { Type } from "typebox";

const execFileAsync = promisify(execFile);

export default {
  name: "llm_masked_pass",
  description:
    "Run a blind masked LLM recall pass after LLM refinement. The script masks pass-1 entities with same-length asterisks, asks a LiteLLM model to discover only new unmasked entities, merges pass-1/pass-2 entities, and writes master_extracted_entities.json.",
  parameters: Type.Object({
    llmPass1: Type.String({ description: "Path to llm_pass1_entities.json." }),
    model: Type.Optional(Type.String({ description: "LiteLLM model name, e.g. gpt-4o, gpt-5.5, or claude-3-5-sonnet-latest." })),
    output: Type.Optional(Type.String({ description: "Output path. Defaults to <llmPass1 parent>/master_extracted_entities.json." })),
    artifactsDir: Type.Optional(Type.String({ description: "Optional directory for masked text artifacts. Defaults to <llmPass1 parent>/llm_masked_pass_artifacts." })),
    chunkIndex: Type.Optional(Type.Array(Type.Number(), { description: "Optional list of chunk indices to process." })),
    temperature: Type.Optional(Type.Number({ description: "LLM sampling temperature. Defaults to 0.0." })),
    dryRun: Type.Optional(Type.Boolean({ description: "If true, create masked artifacts and merge pass-1 entities without calling the LLM." })),
  }),
  async execute(_toolCallId: string, params: any, signal?: AbortSignal) {
    const args = ["run", "scripts/llm_masked_pass.py", "--llm-pass1", params.llmPass1];

    if (params.model) args.push("--model", params.model);
    if (params.output) args.push("--output", params.output);
    if (params.artifactsDir) args.push("--artifacts-dir", params.artifactsDir);
    if (Array.isArray(params.chunkIndex)) {
      for (const index of params.chunkIndex) args.push("--chunk-index", String(index));
    }
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
            text: [stdout.trim(), stderr.trim()].filter(Boolean).join("\n") || "llm_masked_pass completed",
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
            text: `llm_masked_pass failed with exit code ${error?.code ?? "unknown"}: ${error?.stderr || error?.message}`,
          },
        ],
        details: { command: `uv ${args.join(" ")}`, stderr: error?.stderr, stdout: error?.stdout },
        isError: true,
      };
    }
  },
};
