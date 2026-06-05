# blended-ner-framework

A local, chunk-based NER workflow for neuroscience papers.

This project is built around core Python scripts plus thin pi.dev agent wrappers:

- `scripts/ingest_chunk.py` — parses a PDF via local Grobid when available, falls back to PyMuPDF4LLM, and writes model-token-aware chunk files (`chunk_000.txt`, `chunk_001.txt`, ...).
- `scripts/parse_pdf.py` — legacy character-based PDF chunker.
- `scripts/save_chunk_entities.py` — validates a JSON entity array from `stdin` and writes one per-chunk result file under `output/<paper_name>/<run_id>/`.
- `scripts/hybrid_ner_orchestrator.py` — derives GLiNER labels from both explicit user intent and a document text sample, then invokes the local GLiNER runner.
- `scripts/ner.py` — local GLiNER runner that accepts `--input` and `--labels`, runs GLiNER over chunk/text files, and writes JSON outputs.
- `scripts/llm_refinement.py` — injects GLiNER entities inline as `[Entity](LABEL)`, sends decorated chunks to an LLM for verification/deep-pass extraction, and writes `llm_pass1_entities.json`.
- `scripts/llm_masked_pass.py` — masks pass-1 entities with `*`, runs a blind LLM recall pass for missed entities, and writes `master_extracted_entities.json`.
- `scripts/map_ontology.py` — maps extracted entities to ontology identifiers using migrated local/BioPortal concept-mapping logic from the prior CrewAI implementation, without CrewAI overhead.
- `.pi/tools/parse_pdf.ts`, `.pi/tools/hybrid_ner_orchestrator.ts`, `.pi/tools/llm_refinement.ts`, `.pi/tools/llm_masked_pass.ts`, `.pi/tools/map_ontology.ts`, and `.pi/tools/save_chunk_entities.ts` — lightweight TypeScript wrappers that invoke the Python scripts via `uv run` for the agent workflow.

## Requirements

- Python `>=3.12`
- [uv](https://docs.astral.sh/uv/) for dependency/environment management
- Optional but recommended: a running Grobid server (default: `http://localhost:8070`)
- PyMuPDF4LLM is included as a fallback parser when Grobid is unavailable
- API tokens in a local `.env` file when using LiteLLM/Hugging Face/BioPortal credentials

## Setup

```bash
uv sync
```

Optional local credentials for LiteLLM/Hugging Face can be placed in a repo-root `.env` file. The scripts load this file automatically and do not override environment variables that are already set in your shell.

```bash
OPENAI_API_KEY=your_openai_key_here
HF_TOKEN=your_huggingface_token_here
BIOPORTAL_API_KEY=your_bioportal_key_here
LOCAL_CONCEPT_MAPPING_URL=http://localhost:8000
LITELLM_CONCURRENCY=4
```

`.env` is ignored by git; do not commit real API tokens.

## LLM backend roles: pi vs LiteLLM

Pi is the interactive orchestration and code-review layer for this project. The Python pipeline scripts use [LiteLLM](https://docs.litellm.ai/) for direct, repeatable batch LLM calls during label generation, LLM refinement, and masked recall.

The scripts do **not** automatically inherit the model or authentication state from the current pi chat session. The `--model` arguments in scripts such as `hybrid_ner_orchestrator.py`, `llm_refinement.py`, and `llm_masked_pass.py` refer to the LiteLLM model used by those standalone Python processes. You can point LiteLLM at the same underlying model you use in pi if you have API-key access to that provider/model, but the execution path is separate from pi's interactive model/session.

This separation is intentional:

- pi remains responsible for orchestration, code edits, review, debugging, and deciding which deterministic tool to run.
- LiteLLM handles high-volume, structured batch inference from Python scripts without spawning nested pi agents per chunk.
- The pipeline remains portable to non-interactive environments such as remote nodes, Slurm jobs, notebooks, or CI.

For example:

```bash
LITELLM_MODEL=gpt-5.5
OPENROUTER_API_KEY=your_key_here
```

or pass a model explicitly:

```bash
uv run scripts/llm_refinement.py \
  --chunks-dir data/papers/example/chunks \
  --gliner-dir output/gliner/example_run \
  --model gpt-5.5
```

## Running the NER prompt in pi.dev

If pi is installed and running in this repository, start the workflow by entering:

```text
/ner_pipeline
```

This invokes the prompt at `.pi/prompts/ner_pipeline.md`.

## Core scripts and agent tools

The core Python scripts can be run directly. The `/ner_pipeline` workflow should use the pi.dev wrappers in `.pi/tools` instead of calling Python directly.

### 1) Parse and chunk a PDF

Recommended token-aware chunking:

```bash
uv run scripts/ingest_chunk.py data/papers/example.pdf \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --out-dir data/papers/example/<datetime>/chunks/ \
  --max-tokens 4000
```

The `--model-id` argument is the Hugging Face tokenizer/model ID used for chunking, not necessarily the model used later for NER. The tokenizer is loaded with `AutoTokenizer.from_pretrained(...)`; this may download tokenizer files into the local Hugging Face cache, but it does not download or run full model weights.

Use `--max-tokens` to choose a chunk size suitable for the downstream LLM that will process each chunk. For GPT-5.5-style downstream processing, `--max-tokens 4000` is a practical reliability-oriented default that keeps each chunk comfortably sized for extraction and saving. Chunking is sentence-aware and includes a fixed one-sentence overlap between adjacent chunks when the overlap fits under the token limit, reducing boundary-related missed entities/context.

Example request:

```text
Process data/papers/multiscale_spatial_transcriptomic/2025.12.02.691876v1.full.pdf using modelId Qwen/Qwen2.5-7B-Instruct and outDir data/papers/multiscale_spatial_transcriptomic/<datetime>/chunks/. I will be using gpt5.5 when processing the chunks so choose max-tokens accordingly.
```

Equivalent command:

```bash
dt=$(date +%Y%m%dT%H%M%S)
uv run scripts/ingest_chunk.py \
  data/papers/multiscale_spatial_transcriptomic/2025.12.02.691876v1.full.pdf \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --out-dir "data/papers/multiscale_spatial_transcriptomic/${dt}/chunks/" \
  --max-tokens 4000
```

Useful options:

- `--model-id <hugging-face-model-id>` (required tokenizer used for token-aware chunking)
- `--out-dir <path>`
- `--max-tokens 4000` (override tokenizer-derived chunk limit)
- `--grobid-url http://localhost:8070`
- `--grobid-timeout 60`

Legacy character-based chunking is still available:

```bash
uv run scripts/parse_pdf.py data/papers/example.pdf --out-dir data/papers/example.chunks
```

Chunk file format:

1. Line 1: JSON header (includes `chunk_index`, `char_start`, `char_end`, `total_chunks`, `chunking_strategy`, `sentence_overlap`, etc.)
2. Line 2: `---`
3. Line 3+: chunk body text

### 2) Generate hybrid labels and invoke local GLiNER

`scripts/hybrid_ner_orchestrator.py` accepts a natural-language request, infers the target path, samples the target text or first generated chunk, asks an LLM through LiteLLM for a single deduplicated list of 20–30 uppercase neuroscience NER labels, prints the labels, then executes the local GLiNER script at `scripts/ner.py`.

Run it with `uv` so inline dependencies (`litellm`, `pydantic`) are managed automatically:

```bash
uv run scripts/hybrid_ner_orchestrator.py \
  "Look through data/papers/multiscale_spatial_transcriptomic/20260602T143152/chunks, specifically searching for brain regions" \
  --model gpt-5.5 \
  --ner-script scripts/ner.py
```

Dry-run label generation without invoking GLiNER:

```bash
uv run scripts/hybrid_ner_orchestrator.py \
  "Look through ./papers, specifically searching for brain regions" \
  --model gpt-5.5 \
  --dry-run
```

The included local GLiNER runner accepts:

```text
--input <file-or-folder> --labels <COMMA,SEPARATED,LABELS>
```

It dynamically routes PyTorch inference to the fastest available local device:

1. `cuda:0` for NVIDIA CUDA or AMD ROCm builds
2. `mps` for Apple Silicon Metal Performance Shaders
3. `cpu` only as a warning-producing fallback

You can override detection with `--device cuda:0`, `--device mps`, or `--device cpu`. CUDA/ROCm runs use FP16 by default (`--fp16 auto`) to reduce memory and improve tensor-core throughput; MPS and CPU avoid explicit FP16 casting by default. On Apple Silicon, the runner periodically calls `torch.mps.empty_cache()` during long file loops to reduce MPS allocator fragmentation; tune with `--mps-empty-cache-every` or disable with `0`.

Direct GLiNER runner example:

```bash
uv run scripts/ner.py \
  --input data/papers/multiscale_spatial_transcriptomic/20260602T143152/chunks \
  --labels BRAIN_REGION,CELL_TYPE,GENE,TECHNIQUE \
  --device auto \
  --output-dir output/gliner/example_run
```

`uv run scripts/ner.py` uses inline dependency management for `gliner`. The first full run may download the GLiNER package, model files, and backend ML dependencies. If you want to use a different GLiNER runner, pass its location with `--ner-script` and append repeated `--extra-ner-arg` values as needed. To force device routing through the hybrid orchestrator, append arguments such as `--extra-ner-arg --device --extra-ner-arg mps`.

### 3) Refine GLiNER output with an LLM deep pass

After GLiNER has produced `output/gliner/<timestamp>/chunk_NNN.json` files, run the LLM refinement pass against the chunks directory and GLiNER output directory:

```bash
uv run scripts/llm_refinement.py \
  --chunks-dir data/papers/multiscale_spatial_transcriptomic/20260602T192339/chunks \
  --gliner-dir output/gliner/20260602T192415 \
  --model gpt-5.5 \
  --concurrency 4
```

The refinement pass uses `asyncio` plus LiteLLM `acompletion()` to process chunks concurrently. Concurrency is bounded with `--concurrency` or `LITELLM_CONCURRENCY` to avoid unbounded provider/API rate-limit pressure.

This writes by default:

```text
output/gliner/20260602T192415/llm_pass1_entities.json
```

It also writes decorated/refined markdown artifacts under:

```text
output/gliner/20260602T192415/llm_pass1_artifacts/
```

Artifact directory contents:

```text
output/gliner/<timestamp>/
├── chunk_000.json                  # raw local GLiNER entities for chunk_000
├── chunk_001.json                  # raw local GLiNER entities for chunk_001
├── ...
├── manifest.json                   # GLiNER run metadata: model, labels, input files
├── llm_pass1_entities.json         # structured entities parsed from the LLM-refined markdown
└── llm_pass1_artifacts/
    ├── decorated/
    │   └── chunk_000.md            # original chunk body with GLiNER entities injected as [Entity](LABEL)
    ├── refined_markdown/
    │   └── chunk_000.md            # LLM-reviewed annotation layer with corrected/expanded/new [Entity](LABEL) markup
    └── clean_text/
        └── chunk_000.txt           # refined_markdown with annotation syntax removed; indices in llm_pass1_entities.json refer to this clean text
```

How to interpret the LLM refinement artifacts:

- `decorated/` is the input sent to the LLM: the original chunk body plus preliminary GLiNER inline markup.
- `refined_markdown/` is the human-readable LLM output. It shows corrected labels, expanded boundaries, and newly discovered entities using `[Entity Text](LABEL)` syntax.
- `clean_text/` is produced by removing the inline markdown labels from `refined_markdown/`. It should usually be close to the original chunk body, but it can differ if the LLM changed spacing, punctuation, or wording while refining. The `start`/`end` offsets in `llm_pass1_entities.json` are relative to this clean refined text, not necessarily the original chunk file.
- `llm_pass1_entities.json` is the main structured downstream output for the LLM refinement pass.

For a preview that injects GLiNER entities and parses the markdown without calling an LLM:

```bash
uv run scripts/llm_refinement.py \
  --chunks-dir data/papers/multiscale_spatial_transcriptomic/20260602T192339/chunks \
  --gliner-dir output/gliner/20260602T192415 \
  --dry-run
```

### 4) Run masked LLM recall and write the master entity list

After `llm_pass1_entities.json` exists, run a blind masked pass to force the LLM to search only the unmasked text for missed entities:

```bash
uv run scripts/llm_masked_pass.py \
  --llm-pass1 output/gliner/20260602T192415/llm_pass1_entities.json \
  --model gpt-5.5 \
  --concurrency 4
```

The masked recall pass also uses bounded async LiteLLM calls. Increase `--concurrency` for higher-throughput providers; reduce it if you encounter rate limits or model-side queuing.

This writes by default:

```text
output/gliner/20260602T192415/master_extracted_entities.json
output/gliner/20260602T192415/llm_masked_pass_artifacts/masked_text/
```

The masked text artifacts replace already-validated pass-1 entities with same-length `*` blocks. This preserves character positions while preventing the LLM from re-extracting previously found entities. When chunk headers include `char_start`, the master merge records `global_start`/`global_end` and uses them to deduplicate duplicate entity mentions introduced by the one-sentence chunk overlap while preserving true repeated mentions at different document positions.

Dry run without calling the LLM:

```bash
uv run scripts/llm_masked_pass.py \
  --llm-pass1 output/gliner/20260602T192415/llm_pass1_entities.json \
  --dry-run
```

### 5) Map final entities to ontologies

After `master_extracted_entities.json` exists, map extracted entities to ontology identifiers:

```bash
uv run scripts/map_ontology.py \
  --input output/gliner/20260602T192415/master_extracted_entities.json \
  --backend auto
```

In `--backend auto` mode, fallback is **per term**, not per file: the script keeps successful local mappings and calls BioPortal only for terms that the local service did not map. If the local service is unavailable entirely, all terms automatically fall back to BioPortal as long as `BIOPORTAL_API_KEY` is available in `.env` or the shell environment.

This writes by default:

```text
output/gliner/20260602T192415/neuro_entities_mapped.json
```

`map_ontology.py` ports the deterministic logic from the prior CrewAI concept-mapping tools while removing CrewAI framework overhead. It preserves:

- robust text sanitization and 500-character max query length
- context-aware mapping with source sentence context truncated to 200 characters
- local concept mapping through `POST <LOCAL_CONCEPT_MAPPING_URL>/map/batch`
- batch deduplication, in-memory caching, configurable batch size/workers, and request timeouts
- per-term BioPortal fallback/search using `http://data.bioontology.org/search` when `--backend auto` and local mapping fails for an individual term
- BioPortal exact-match-first lookup, followed by fuzzy fallback when no exact result is found
- configurable BioPortal ontology filtering, defaulting to `UBERON,NIFSTD,FMA,GO,SNOMEDCT`
- tenacity exponential backoff/retries for BioPortal `429 Too Many Requests` and `5xx` errors
- final enriched fields: `extracted_text`, `llm_label`, `bioportal_prefLabel`, and `ontology_uri`

Useful environment variables:

```bash
LOCAL_CONCEPT_MAPPING_URL=http://localhost:8000
LOCAL_CONCEPT_MAPPING_TIMEOUT=30
LOCAL_CONCEPT_MAPPING_BATCH_SIZE=4000
LOCAL_CONCEPT_MAPPING_WORKERS=4
MAX_CONCEPT_MAPPING_RESULTS=1
BIOPORTAL_API_KEY=your_bioportal_key_here
# BioPortal retries are handled by tenacity exponential backoff.
BIOPORTAL_API_KEY=your_bioportal_key_here
```

Force BioPortal and restrict ontology acronyms:

```bash
uv run scripts/map_ontology.py \
  --input output/gliner/20260602T192415/master_extracted_entities.json \
  --backend bioportal \
  --ontologies UBERON,CL,GO,NCIT,MONDO \
  --max-results 3 \
  --csv
```

The `--csv` flag writes an easy-viewing CSV next to the JSON output. You can also pass an explicit CSV path, for example `--csv output/my_entities.csv`.

### 6) Save NER output for one chunk

`save_chunk_entities.py` expects a JSON **array** on `stdin`:

```json
[
  {"entity": "S1", "label": "BrainRegion", "context": "..."}
]
```

Example:

```bash
echo '[{"entity":"S1","label":"BrainRegion","context":"Layer IV of S1..."}]' \
  | uv run scripts/save_chunk_entities.py \
      --paper-name example \
      --run-id 20260528T143215_a3f1 \
      --chunk-index 0
```

This writes:

```text
output/example/20260528T143215_a3f1/chunk_000.json
```

## Project layout

```text
.
├── scripts/
│   ├── hybrid_ner_orchestrator.py
│   ├── ingest_chunk.py
│   ├── llm_masked_pass.py
│   ├── llm_refinement.py
│   ├── map_ontology.py
│   ├── ner.py
│   ├── parse_pdf.py
│   └── save_chunk_entities.py
├── .pi/
│   ├── prompts/
│   │   └── ner_pipeline.md
│   └── tools/
│       ├── hybrid_ner_orchestrator.ts
│       ├── llm_masked_pass.ts
│       ├── llm_refinement.ts
│       ├── map_ontology.ts
│       ├── parse_pdf.ts
│       └── save_chunk_entities.ts
├── data/
├── output/
├── pyproject.toml
└── uv.lock
```
