import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { Type } from "typebox";

const execFileAsync = promisify(execFile);

export default {
  name: "audit_ner_output",
  description:
    "Audit and normalize final Neuroscience NER outputs. Preserves every mention while adding grouped entities, run statistics, span validation, and ontology IRI checks.",
  parameters: Type.Object({
    input: Type.String({ description: "Input JSON path: master_extracted_entities.json, neuro_entities_mapped.json, or an object with entities[]." }),
    output: Type.Optional(Type.String({ description: "Output JSON path. Defaults to <input_stem>_audited.json." })),
    sourceText: Type.Optional(Type.String({ description: "Optional original full text file for global_start/global_end span validation." })),
    strictIri: Type.Optional(Type.Boolean({ description: "If false, disables strict ontology IRI structural validation. Defaults to true." })),
    failOnInvalid: Type.Optional(Type.Boolean({ description: "If true, return an error when invalid spans or invalid ontology IRIs are found." })),
  }),
  async execute(_toolCallId: string, params: any, signal?: AbortSignal) {
    const args = ["run", "scripts/audit_ner_output.py", "--input", params.input];

    if (params.output) args.push("--output", params.output);
    if (params.sourceText) args.push("--source-text", params.sourceText);
    if (params.strictIri === false) args.push("--no-strict-iri");
    if (params.failOnInvalid) args.push("--fail-on-invalid");

    try {
      const { stdout, stderr } = await execFileAsync("uv", args, {
        signal,
        maxBuffer: 1024 * 1024,
      });

      return {
        content: [
          {
            type: "text",
            text: [stdout.trim(), stderr.trim()].filter(Boolean).join("\n") || "audit_ner_output completed",
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
            text: `audit_ner_output failed with exit code ${error?.code ?? "unknown"}: ${error?.stderr || error?.message}`,
          },
        ],
        details: { command: `uv ${args.join(" ")}`, stderr: error?.stderr, stdout: error?.stdout },
        isError: true,
      };
    }
  },
};
