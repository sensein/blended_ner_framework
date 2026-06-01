import { spawn } from "node:child_process";
import { Type } from "typebox";

function runWithStdin(command: string, args: string[], stdinText: string, signal?: AbortSignal) {
  return new Promise<{ stdout: string; stderr: string; code: number | null }>((resolve, reject) => {
    const child = spawn(command, args, { signal });
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));
    child.on("error", reject);
    child.on("close", (code) => resolve({ stdout, stderr, code }));

    child.stdin.write(stdinText);
    child.stdin.end();
  });
}

export default {
  name: "save_chunk_entities",
  description:
    "Validate and persist one chunk's NER entity JSON by running scripts/save_chunk_entities.py via uv. Heavy validation and file I/O stay inside the Python script; this wrapper returns only lightweight success metadata.",
  parameters: Type.Object({
    paperName: Type.String({ description: "Stable paper/document id, usually the parse_pdf doc_id." }),
    runId: Type.String({ description: "Orchestrator-provided run id; use verbatim." }),
    chunkIndex: Type.Number({ description: "Zero-based chunk index." }),
    entitiesJson: Type.String({ description: "JSON array of entity mentions to pass to the Python script on stdin." }),
    outputRoot: Type.Optional(Type.String({ description: "Optional output root. Defaults to output/." })),
  }),
  async execute(_toolCallId: string, params: any, signal?: AbortSignal) {
    const args = [
      "run",
      "scripts/save_chunk_entities.py",
      "--paper-name",
      params.paperName,
      "--run-id",
      params.runId,
      "--chunk-index",
      String(params.chunkIndex),
    ];

    if (params.outputRoot) args.push("--output-root", params.outputRoot);

    const result = await runWithStdin("uv", args, params.entitiesJson, signal);
    const outputPath = result.stdout.trim().split(/\r?\n/).filter(Boolean).at(-1) ?? "";

    if (result.code !== 0) {
      return {
        content: [
          {
            type: "text",
            text: `save_chunk_entities failed with exit code ${result.code}: ${result.stderr || result.stdout}`,
          },
        ],
        details: { command: `uv ${args.join(" ")}`, stderr: result.stderr, stdout: result.stdout },
        isError: true,
      };
    }

    return {
      content: [
        {
          type: "text",
          text: `Saved entities for chunk_${String(params.chunkIndex).padStart(3, "0")} to ${outputPath}`,
        },
      ],
      details: {
        command: `uv ${args.join(" ")}`,
        outputPath,
      },
    };
  },
};
