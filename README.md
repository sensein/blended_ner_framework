# blended-ner-framework

A local, chunk-based NER workflow for neuroscience papers.

This project is built around core Python scripts plus thin pi.dev agent wrappers:

- `scripts/ingest_chunk.py` — parses a PDF via local Grobid when available, falls back to PyMuPDF4LLM, and writes model-token-aware chunk files (`chunk_000.txt`, `chunk_001.txt`, ...).
- `scripts/parse_pdf.py` — legacy character-based PDF chunker.
- `scripts/save_chunk_entities.py` — validates a JSON entity array from `stdin` and writes one per-chunk result file under `output/<paper_name>/<run_id>/`.
- `.pi/tools/parse_pdf.ts` and `.pi/tools/save_chunk_entities.ts` — lightweight TypeScript wrappers that invoke the Python scripts via `uv run` for the agent workflow.

## Requirements

- Python `>=3.12`
- [uv](https://docs.astral.sh/uv/) for dependency/environment management
- Optional but recommended: a running Grobid server (default: `http://localhost:8070`)
- PyMuPDF4LLM is included as a fallback parser when Grobid is unavailable

## Setup

```bash
uv sync
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

Use `--max-tokens` to choose a chunk size suitable for the downstream LLM that will process each chunk. For GPT-5.5-style downstream processing, `--max-tokens 4000` is a practical reliability-oriented default that keeps each chunk comfortably sized for extraction and saving.

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

1. Line 1: JSON header (includes `chunk_index`, `global_offset`, `total_chunks`, etc.)
2. Line 2: `---`
3. Line 3+: chunk body text

### 2) Save NER output for one chunk

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
│   ├── ingest_chunk.py
│   ├── parse_pdf.py
│   └── save_chunk_entities.py
├── .pi/
│   ├── prompts/
│   │   └── ner_pipeline.md
│   └── tools/
│       ├── parse_pdf.ts
│       └── save_chunk_entities.ts
├── data/
├── output/
├── pyproject.toml
└── uv.lock
```
