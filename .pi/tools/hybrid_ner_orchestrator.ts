import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { Type } from "typebox";

const execFileAsync = promisify(execFile);

export default {
  name: "hybrid_ner_orchestrator",
  description:
    "Generate a hybrid GLiNER label list from explicit user intent plus a document text sample by running scripts/hybrid_ner_orchestrator.py via uv. The Python script loads repo-root .env, uses LiteLLM + Pydantic, prints the labels, and can optionally invoke scripts/ner.py.",
  parameters: Type.Object({
    prompt: Type.String({
      description:
        "Natural-language request containing the target path and desired entity focus, e.g. 'Look through ./papers, specifically searching for brain regions'.",
    }),
    model: Type.Optional(Type.String({ description: "LiteLLM model name. Defaults to LITELLM_MODEL or gpt-5.5." })),
    sampleChars: Type.Optional(Type.Number({ description: "Number of characters to sample from the target text/chunk. Defaults to 2000." })),
    nerScript: Type.Optional(Type.String({ description: "Path to local GLiNER runner. Defaults to ner.py, resolving to scripts/ner.py when present." })),
    dryRun: Type.Optional(Type.Boolean({ description: "If true, generate/print labels but do not invoke ner.py." })),
    extraNerArgs: Type.Optional(Type.Array(Type.String(), { description: "Additional args appended to the ner.py command." })),
  }),
  async execute(_toolCallId: string, params: any, signal?: AbortSignal) {
    const args = ["run", "scripts/hybrid_ner_orchestrator.py", params.prompt];

    if (params.model) args.push("--model", params.model);
    if (params.sampleChars !== undefined) args.push("--sample-chars", String(params.sampleChars));
    if (params.nerScript) args.push("--ner-script", params.nerScript);
    if (params.dryRun) args.push("--dry-run");
    if (Array.isArray(params.extraNerArgs)) {
      for (const extraArg of params.extraNerArgs) args.push("--extra-ner-arg", extraArg);
    }

    try {
      const { stdout, stderr } = await execFileAsync("uv", args, {
        signal,
        maxBuffer: 1024 * 1024,
      });

      return {
        content: [
          {
            type: "text",
            text: stdout.trim() || "hybrid_ner_orchestrator completed",
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
            text: `hybrid_ner_orchestrator failed with exit code ${error?.code ?? "unknown"}: ${error?.stderr || error?.message}`,
          },
        ],
        details: { command: `uv ${args.join(" ")}`, stderr: error?.stderr, stdout: error?.stdout },
        isError: true,
      };
    }
  },
};
