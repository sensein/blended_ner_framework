import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { Type } from "typebox";

const execFileAsync = promisify(execFile);

export default {
  name: "map_ontology",
  description:
    "Map master extracted NER entities to ontology identifiers using migrated CrewAI concept-mapping logic without CrewAI overhead. Prefers the local /map/batch service and can fall back to BioPortal.",
  parameters: Type.Object({
    input: Type.String({ description: "Path to master_extracted_entities.json." }),
    output: Type.Optional(Type.String({ description: "Output JSON path. Defaults to <input parent>/neuro_entities_mapped.json." })),
    csv: Type.Optional(Type.String({ description: "Optional CSV output path. Use AUTO to write <output>.csv." })),
    backend: Type.Optional(Type.Union([Type.Literal("auto"), Type.Literal("local"), Type.Literal("bioportal")], { description: "Mapping backend. Defaults to auto." })),
    maxResults: Type.Optional(Type.Number({ description: "Maximum ontology results per term. Defaults to MAX_CONCEPT_MAPPING_RESULTS or 1." })),
    ontologies: Type.Optional(Type.String({ description: "Comma-separated BioPortal ontology acronyms. Ignored by local backend." })),
  }),
  async execute(_toolCallId: string, params: any, signal?: AbortSignal) {
    const args = ["run", "scripts/map_ontology.py", "--input", params.input];

    if (params.output) args.push("--output", params.output);
    if (params.csv) args.push("--csv", params.csv);
    if (params.backend) args.push("--backend", params.backend);
    if (params.maxResults !== undefined) args.push("--max-results", String(params.maxResults));
    if (params.ontologies) args.push("--ontologies", params.ontologies);

    try {
      const { stdout, stderr } = await execFileAsync("uv", args, {
        signal,
        maxBuffer: 1024 * 1024,
      });

      return {
        content: [
          {
            type: "text",
            text: [stdout.trim(), stderr.trim()].filter(Boolean).join("\n") || "map_ontology completed",
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
            text: `map_ontology failed with exit code ${error?.code ?? "unknown"}: ${error?.stderr || error?.message}`,
          },
        ],
        details: { command: `uv ${args.join(" ")}`, stderr: error?.stderr, stdout: error?.stdout },
        isError: true,
      };
    }
  },
};
