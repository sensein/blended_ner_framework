# blended-ner-framework

A local, chunk-based NER workflow for neuroscience papers.

This project is built around two CLI tools:

- `tools/parse_pdf_grobid.py` — parses a PDF via Grobid or PyMuPDF4LLM and writes payload-safe chunk files (`chunk_000.txt`, `chunk_001.txt`, ...).
- `tools/save_chunk_entities.py` — validates a JSON entity array from `stdin` and writes one per-chunk result file under `output/<paper_name>/<run_id>/`.

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

## Usage

### 1) Parse and chunk a PDF

```bash
uv run tools/parse_pdf_grobid.py data/papers/example.pdf --out-dir data/papers/example.chunks
```

Useful options:

- `--parser auto` (default: try Grobid, fall back to PyMuPDF4LLM)
- `--parser grobid` (force Grobid)
- `--parser pymupdf4llm` (force PyMuPDF4LLM)
- `--grobid-url http://localhost:8070`
- `--max-chars 45000`

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
  | uv run tools/save_chunk_entities.py \
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
├── tools/
│   ├── parse_pdf_grobid.py
│   └── save_chunk_entities.py
├── data/
├── output/
├── pyproject.toml
└── uv.lock
```
