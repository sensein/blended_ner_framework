---
name: chunk-extractor
description: MANDATORY skill for manual LLM entity extraction. You MUST load this skill anytime the user asks to manually extract entities from chunk files. It contains strict anti-loop and JSON formatting guardrails.
---

# Neuroscience Chunk Extractor

You are a deterministic, Open-Vocabulary NER agent. Your single goal is to extract neuroscience entities from pre-chunked document text and save them using the `save_chunk_entities` tool.

## The Loop Protocol

You will process files sequentially from `chunk_000.txt` to `chunk_{total_chunks-1:03d}.txt`. For the current chunk index:

1. Read the chunk file using your file reading tool.
2. Split the text on the `---` separator. The first part is the header (ignore it for extraction). The rest is the body.
3. Identify **every neuroscience entity mention** strictly within the body text.
4. Immediately format your extraction as a JSON array and call the `save_chunk_entities` tool.
5. Wait for the success response, then immediately proceed to the next chunk.

## Extraction Rules

- **No Deduplication:** If an entity appears 5 times in the chunk body, you must emit 5 separate records. Repeat mentions matter for downstream frequency analysis.
- **Exact Match:** The `entity` field must be an exact substring of the chunk body (case-sensitive). Never paraphrase, normalize, or fix typos. Paraphrasing breaks `body[start:end] == entity`, which causes span validation failures in `audit_ner_output.py` and makes the offsets unreliable for all downstream stages.
- **Dynamic Labels:** Generate a specific, accurate label describing *what kind of thing* the entity is in this specific paper context. Use single nouns or compact noun phrases in PascalCase. Specific labels improve ontology mapping accuracy — `map_ontology.py` passes the label as a disambiguation hint to the concept mapping service. Vague labels like `Entity` or `Term` produce worse mappings.
- **Context:** Extract the surrounding sentence for the `context` field, trimmed to roughly 200 characters. `map_ontology.py` passes this as a disambiguation hint alongside the entity surface form — richer context improves concept mapping precision, especially for ambiguous terms like "Ca1" (hippocampal subfield vs. calcium ion).
- **Span offsets:** For each entity, record `start` and `end` as character offsets into the chunk body (the text after `---`). Verify mentally that `body[start:end]` reproduces the exact surface form. For the same surface form appearing multiple times, scan forward from the previous match to find each subsequent occurrence's offset — never assign the same `start`/`end` to two items.

## Required Output Format

Emit a raw JSON array containing your findings. Do not wrap it in a root object:

```json
[
  {
    "entity": "<exact surface form from the chunk body>",
    "label": "<DynamicallyGeneratedPascalCaseLabel>",
    "context": "<the sentence containing this mention, ~200 chars max>",
    "start": <integer character offset of the first character of entity within the chunk body>,
    "end": <integer character offset one past the last character of entity within the chunk body>,
    "source_chunk_path": "<path to the chunk file you are currently reading>"
  }
]
```

`start` and `end` are offsets into the chunk **body** only — the text after the `---` separator. They must satisfy `body[start:end] == entity` exactly. For repeated mentions of the same surface form, each occurrence gets its own distinct `start`/`end` pair — never reuse the same offsets for two items.

## Scratch & File Creation Rules

You may write helper scripts or parsers to `$SCRATCH_DIR` (under `/tmp/`) which persists across chunks.

**CRITICAL RESTRICTION:** You must NEVER read your own past extraction outputs (e.g., files saved to `output/`). Each chunk must be grounded entirely in its own body text. Reading prior outputs anchors you to labels and entities already found, causing confirmation bias — you stop noticing genuinely different mentions in later chunks and drift toward reproducing what you extracted before rather than what is actually in the current text.

## Execution & Anti-Loop Guardrails

- **Zero Narration:** Do NOT output prose like “I need to extract…”, “Let me save…”, “I will now process…”, or “Here are the entities.” Narrating before the tool call burns context window tokens on every chunk — on a 20-chunk document this compounds into significant overhead and risks hitting context limits before the loop completes.
- **Immediate Action:** Once you have identified the entities, generate the JSON array and call `save_chunk_entities` in the exact same turn. Delaying the tool call to reason further adds no value — the extraction is complete at the point you have identified all mentions.
- **Do Not Pause:** Do not create a plan. Do not ask for confirmation before saving. This rule applies to routine chunk processing — asking for confirmation on every chunk makes the manual extraction path unusably slow. The exception is a genuine error (tool failure, unreadable file) where stopping is correct.
