#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "litellm>=1.70.0",
#   "pydantic>=2.7.0",
# ]
# ///
"""Hybrid label-generation orchestrator for local GLiNER NER.

Given a natural-language request, this script:
1. infers the target file/folder path from the request,
2. samples the target text or first ingestion chunk,
3. asks an LLM for a unified deduplicated neuroscience NER label list, and
4. invokes a local GLiNER runner (`ner.py`) with those labels.

Run with uv so inline dependencies are installed in an isolated environment:

    uv run scripts/hybrid_ner_orchestrator.py \
      "Look through ./papers, specifically searching for brain regions" \
      --model gpt-5.5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from litellm import completion
from pydantic import BaseModel, Field, ValidationError, field_validator


SYSTEM_PROMPT = (
    "You are a neuroscience ontology expert. Analyze the user's request and the provided text sample. "
    "Generate a comprehensive, deduplicated list of 20-30 uppercase NER entity labels. "
    "This list must include any specific entity categories requested by the user, supplemented by relevant "
    "neuroscience categories found natively within the text sample (e.g., UBERON terms, cell types, techniques)."
)

PATH_RE = re.compile(r"(?P<path>(?:\.{1,2}/|/|~/?)[^\s,;:'\"`]+|[\w.-]+(?:/[\w .+@%=-]+)+)")
TEXT_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".xml", ".tei"}
PDF_SUFFIXES = {".pdf"}


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding existing env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class LabelResponse(BaseModel):
    labels: list[str] = Field(..., min_length=1, max_length=30)

    @field_validator("labels")
    @classmethod
    def normalize_labels(cls, labels: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for label in labels:
            cleaned = re.sub(r"[^A-Za-z0-9_ ]+", "", str(label)).strip().replace(" ", "_").upper()
            cleaned = re.sub(r"_+", "_", cleaned)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                normalized.append(cleaned)
        return normalized[:30]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hybrid GLiNER labels from user intent plus document text.")
    parser.add_argument("prompt", help="Natural-language request, e.g. 'Look through ./papers for brain regions'.")
    parser.add_argument("--model", default=os.getenv("LITELLM_MODEL", "gpt-5.5"), help="LiteLLM model name.")
    parser.add_argument("--sample-chars", type=int, default=2000, help="Number of text characters to sample.")
    parser.add_argument("--ner-script", default="ner.py", help="Path to the local GLiNER script to execute.")
    parser.add_argument("--dry-run", action="store_true", help="Print labels but do not invoke ner.py.")
    parser.add_argument("--extra-ner-arg", action="append", default=[], help="Extra argument to append to ner.py invocation; repeatable.")
    return parser.parse_args()


def infer_path(prompt: str) -> Path:
    candidates = [Path(os.path.expanduser(m.group("path"))).resolve() for m in PATH_RE.finditer(prompt)]
    existing = [p for p in candidates if p.exists()]
    if existing:
        return existing[0]
    if candidates:
        raise FileNotFoundError(f"Inferred path does not exist: {candidates[0]}")
    raise ValueError("Could not infer an input file or folder path from the prompt. Include a path like ./papers or data/papers/file.pdf.")


def first_chunk_in(path: Path) -> Path | None:
    if path.is_file() and path.name.startswith("chunk_") and path.suffix == ".txt":
        return path
    root = path if path.is_dir() else path.parent
    chunks = sorted(root.rglob("chunk_*.txt"))
    return chunks[0] if chunks else None


def first_readable_file(path: Path) -> Path:
    if path.is_file():
        return path
    chunk = first_chunk_in(path)
    if chunk:
        return chunk
    for suffixes in (TEXT_SUFFIXES, PDF_SUFFIXES):
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
        if files:
            return files[0]
    raise FileNotFoundError(f"No readable text/chunk/PDF file found under {path}")


def strip_chunk_header(text: str) -> str:
    if "\n---\n" in text:
        return text.split("\n---\n", 1)[1]
    return text


def sample_text(target: Path, sample_chars: int) -> tuple[Path, str]:
    source = first_chunk_in(target) or first_readable_file(target)
    if source.suffix.lower() in PDF_SUFFIXES:
        raise ValueError(
            f"{source} is a PDF and no chunk text was found. Run scripts/ingest_chunk.py first, then point this orchestrator at the chunks folder."
        )
    text = source.read_text(encoding="utf-8", errors="replace")
    text = strip_chunk_header(text).strip()
    return source, text[:sample_chars]


def extract_json_object(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def generate_labels(model: str, user_prompt: str, text_sample: str) -> list[str]:
    schema = LabelResponse.model_json_schema()
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Return only JSON matching this schema:\n"
                    f"{json.dumps(schema)}\n\n"
                    f"USER_REQUEST:\n{user_prompt}\n\n"
                    f"DOCUMENT_SAMPLE:\n{text_sample}"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    try:
        parsed = LabelResponse.model_validate(extract_json_object(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError(f"LLM did not return a valid label payload: {exc}\nRaw response:\n{raw}") from exc
    return parsed.labels


def resolve_ner_script(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    scripts_candidate = Path("scripts") / path
    if scripts_candidate.exists():
        return scripts_candidate
    raise FileNotFoundError(f"Local GLiNER script not found: {path}. Pass --ner-script PATH.")


def run_ner(ner_script: Path, target: Path, labels: Iterable[str], extra_args: list[str]) -> int:
    labels_arg = ",".join(labels)
    cmd = ["uv", "run", str(ner_script), "--input", str(target), "--labels", labels_arg, *extra_args]
    print("\nExecuting GLiNER command:")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def main() -> int:
    load_dotenv()
    args = parse_args()
    target = infer_path(args.prompt)
    source, text_sample = sample_text(target, args.sample_chars)

    print(f"Inferred target: {target}")
    print(f"Sample source: {source}")

    labels = generate_labels(args.model, args.prompt, text_sample)
    print("\nGenerated labels:")
    print(json.dumps(labels, indent=2))

    if args.dry_run:
        return 0

    ner_script = resolve_ner_script(args.ner_script)
    return run_ner(ner_script, target, labels, args.extra_ner_arg)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
