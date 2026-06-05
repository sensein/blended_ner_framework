#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "litellm>=1.70.0",
#   "pydantic>=2.7.0",
# ]
# ///
"""LLM refinement pass over local GLiNER outputs.

This script injects GLiNER entities into raw chunk text as inline markdown
annotations, sends the decorated text to a frontier LLM for verification and
additional extraction, then parses the refined markdown back into structured
entity spans.

Example:

    uv run scripts/llm_refinement.py \
      --chunks-dir data/papers/multiscale_spatial_transcriptomic/20260602T192339/chunks \
      --gliner-dir output/gliner/20260602T192415 \
      --model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from litellm import completion
from pydantic import BaseModel, Field, ValidationError, field_validator


SYSTEM_PROMPT = """You are an expert neuroscience annotator. You will receive a text chunk containing preliminary entity labels formatted as [Entity Text](LABEL). Your tasks are:

Verify Existing Labels: Correct any incorrect or overly generalized inline labels (e.g., changing a broad [CA1](Brain Region) to a highly specific [CA1](Hippocampal_Subfield) if context dictates).

Expand Boundaries: If the preliminary pass clipped an entity boundary, expand it (e.g., adjust dorsal [hippocampus](Brain Region) to [dorsal hippocampus](Brain Region)).

Deep-Pass Extraction: Discover and label any neuroscience entities that the preliminary pass missed entirely.

Return the output as a clean, fully refined markdown text chunk preserving all correct, modified, and newly discovered entities in the [Entity Text](LABEL) format.

Additional requirements:
- Preserve the original prose and ordering as much as possible.
- Do not summarize, omit, or rewrite non-entity text.
- Keep labels compact, specific, and ontology-like, using PascalCase or UPPER_SNAKE_CASE consistently.
- Target biomedical/neuroscientific concepts that can be mapped to ontology identifiers (IRIs) and labels.
- Useful for diseases, genes, proteins, chemicals, anatomical structures, etc.
- Prefer exact source-text entity spans and preserve enough surrounding source sentence context for downstream ontology re-ranking.
- Return only the refined markdown text chunk, with no commentary, no preamble, and no fenced code block.
"""

ANNOTATION_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\n]+)\)")
CHUNK_RE = re.compile(r"chunk_(\d+)\.txt$")


class RefinedEntity(BaseModel):
    entity: str
    label: str
    start: int = Field(description="Start index relative to clean refined text with markdown removed.")
    end: int = Field(description="End index relative to clean refined text with markdown removed.")
    markdown_start: int = Field(description="Start index of the entity text inside the refined markdown text.")
    markdown_end: int = Field(description="End index of the entity text inside the refined markdown text.")
    annotation_start: int = Field(description="Start index of the full [Entity](LABEL) annotation in refined markdown text.")
    annotation_end: int = Field(description="End index of the full [Entity](LABEL) annotation in refined markdown text.")
    context: str

    @field_validator("label")
    @classmethod
    def normalize_label(cls, label: str) -> str:
        cleaned = str(label).strip()
        cleaned = re.sub(r"\s+", "_", cleaned)
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "", cleaned)
        return cleaned


class RefinedChunk(BaseModel):
    chunk: str
    source_chunk_path: str
    gliner_path: str
    decorated_text_path: str | None = None
    refined_markdown_path: str | None = None
    clean_text_path: str | None = None
    entity_count: int
    entities: list[RefinedEntity]


class RefinementOutput(BaseModel):
    created_at: str
    model: str
    chunks_dir: str
    gliner_dir: str
    chunks: list[RefinedChunk]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine GLiNER annotations with an LLM deep pass.")
    parser.add_argument("--chunks-dir", required=True, type=Path, help="Directory containing chunk_NNN.txt files.")
    parser.add_argument("--gliner-dir", required=True, type=Path, help="Directory containing GLiNER chunk_NNN.json outputs.")
    parser.add_argument("--model", default=os.getenv("LITELLM_MODEL", "gpt-4o"), help="LiteLLM model name, e.g. gpt-4o or claude-3-5-sonnet-latest.")
    parser.add_argument("--output", type=Path, help="Output JSON path. Defaults to <gliner-dir>/llm_pass1_entities.json.")
    parser.add_argument("--artifacts-dir", type=Path, help="Optional directory for decorated/refined markdown and clean text artifacts.")
    parser.add_argument("--chunk-index", type=int, action="append", help="Only process the given chunk index; repeatable.")
    parser.add_argument("--max-chars", type=int, default=0, help="Optional maximum raw chunk characters to send per chunk; 0 means no truncation.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM sampling temperature.")
    parser.add_argument("--dry-run", action="store_true", help="Write decorated text artifacts but do not call the LLM.")
    return parser.parse_args()


def chunk_sort_key(path: Path) -> int:
    match = CHUNK_RE.search(path.name)
    return int(match.group(1)) if match else 10**9


def strip_chunk_header(text: str) -> str:
    if "\n---\n" in text:
        return text.split("\n---\n", 1)[1]
    return text


def read_chunk_body(path: Path, max_chars: int = 0) -> str:
    body = strip_chunk_header(path.read_text(encoding="utf-8", errors="replace"))
    return body[:max_chars] if max_chars and max_chars > 0 else body


def load_gliner_entities(path: Path, text_len: int) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"GLiNER output must be a JSON array: {path}")

    valid: list[dict[str, Any]] = []
    for ent in data:
        if not isinstance(ent, dict):
            continue
        try:
            start = int(ent["start"])
            end = int(ent["end"])
        except (KeyError, TypeError, ValueError):
            continue
        label = str(ent.get("label", "")).strip()
        if 0 <= start < end <= text_len and label:
            copied = dict(ent)
            copied["start"] = start
            copied["end"] = end
            copied["label"] = label
            valid.append(copied)
    return valid


def non_overlapping_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep highest-quality non-overlapping entities for safe inline injection."""
    sorted_entities = sorted(
        entities,
        key=lambda e: (int(e["start"]), -(int(e["end"]) - int(e["start"])), -float(e.get("score") or 0.0)),
    )
    kept: list[dict[str, Any]] = []
    cursor = -1
    for ent in sorted_entities:
        start = int(ent["start"])
        end = int(ent["end"])
        if start < cursor:
            continue
        kept.append(ent)
        cursor = end
    return kept


def decorate_text(raw_text: str, gliner_entities: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    cursor = 0
    for ent in non_overlapping_entities(gliner_entities):
        start = int(ent["start"])
        end = int(ent["end"])
        label = str(ent["label"]).strip().replace(" ", "_")
        surface = raw_text[start:end]
        parts.append(raw_text[cursor:start])
        parts.append(f"[{surface}]({label})")
        cursor = end
    parts.append(raw_text[cursor:])
    return "".join(parts)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def refine_with_llm(model: str, decorated_text: str, temperature: float) -> str:
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": decorated_text},
        ],
        temperature=temperature,
    )
    content = response.choices[0].message.content or ""
    return strip_code_fences(content)


def context_for(clean_text: str, start: int, end: int, window: int = 100) -> str:
    left = max(0, start - window)
    right = min(len(clean_text), end + window)
    return re.sub(r"\s+", " ", clean_text[left:right]).strip()


def parse_refined_markdown(refined_markdown: str) -> tuple[str, list[RefinedEntity]]:
    clean_parts: list[str] = []
    entities: list[RefinedEntity] = []
    markdown_cursor = 0
    clean_cursor = 0

    matches = list(ANNOTATION_RE.finditer(refined_markdown))
    for match in matches:
        prefix = refined_markdown[markdown_cursor : match.start()]
        clean_parts.append(prefix)
        clean_cursor += len(prefix)

        entity_text = match.group(1)
        label = match.group(2)
        start = clean_cursor
        end = start + len(entity_text)
        clean_parts.append(entity_text)
        clean_cursor = end

        entities.append(
            RefinedEntity(
                entity=entity_text,
                label=label,
                start=start,
                end=end,
                markdown_start=match.start(1),
                markdown_end=match.end(1),
                annotation_start=match.start(),
                annotation_end=match.end(),
                context="",  # filled after clean text is complete
            )
        )
        markdown_cursor = match.end()

    suffix = refined_markdown[markdown_cursor:]
    clean_parts.append(suffix)
    clean_text = "".join(clean_parts)

    finalized: list[RefinedEntity] = []
    for ent in entities:
        finalized.append(ent.model_copy(update={"context": context_for(clean_text, ent.start, ent.end)}))
    return clean_text, finalized


def write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def process_chunk(
    *,
    chunk_path: Path,
    gliner_path: Path,
    model: str,
    artifacts_dir: Path | None,
    max_chars: int,
    temperature: float,
    dry_run: bool,
) -> RefinedChunk:
    raw_text = read_chunk_body(chunk_path, max_chars=max_chars)
    gliner_entities = load_gliner_entities(gliner_path, len(raw_text))
    decorated = decorate_text(raw_text, gliner_entities)

    decorated_path: str | None = None
    refined_path: str | None = None
    clean_path: str | None = None

    if artifacts_dir:
        decorated_path = write_text(artifacts_dir / "decorated" / f"{chunk_path.stem}.md", decorated)

    if dry_run:
        refined_markdown = decorated
    else:
        refined_markdown = refine_with_llm(model, decorated, temperature)

    clean_text, entities = parse_refined_markdown(refined_markdown)

    if artifacts_dir:
        refined_path = write_text(artifacts_dir / "refined_markdown" / f"{chunk_path.stem}.md", refined_markdown)
        clean_path = write_text(artifacts_dir / "clean_text" / f"{chunk_path.stem}.txt", clean_text)

    return RefinedChunk(
        chunk=chunk_path.stem,
        source_chunk_path=str(chunk_path),
        gliner_path=str(gliner_path),
        decorated_text_path=decorated_path,
        refined_markdown_path=refined_path,
        clean_text_path=clean_path,
        entity_count=len(entities),
        entities=entities,
    )


def selected_chunk_paths(chunks_dir: Path, selected_indices: Iterable[int] | None) -> list[Path]:
    paths = sorted(chunks_dir.glob("chunk_*.txt"), key=chunk_sort_key)
    if selected_indices is None:
        return paths
    wanted = set(selected_indices)
    return [p for p in paths if chunk_sort_key(p) in wanted]


def main() -> int:
    load_dotenv()
    args = parse_args()

    chunks_dir = args.chunks_dir.expanduser().resolve()
    gliner_dir = args.gliner_dir.expanduser().resolve()
    output_path = args.output or (gliner_dir / "llm_pass1_entities.json")
    artifacts_dir = args.artifacts_dir.expanduser().resolve() if args.artifacts_dir else (output_path.parent / "llm_pass1_artifacts")

    if not chunks_dir.is_dir():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
    if not gliner_dir.is_dir():
        raise FileNotFoundError(f"GLiNER directory not found: {gliner_dir}")

    chunk_paths = selected_chunk_paths(chunks_dir, args.chunk_index)
    if not chunk_paths:
        raise FileNotFoundError(f"No selected chunk_*.txt files found in {chunks_dir}")

    refined_chunks: list[RefinedChunk] = []
    for chunk_path in chunk_paths:
        gliner_path = gliner_dir / f"{chunk_path.stem}.json"
        if not gliner_path.exists():
            print(f"⚠ skipping {chunk_path.name}: missing {gliner_path}", file=sys.stderr)
            continue
        print(f"Refining {chunk_path.name} with {gliner_path.name}...", file=sys.stderr)
        refined_chunks.append(
            process_chunk(
                chunk_path=chunk_path,
                gliner_path=gliner_path,
                model=args.model,
                artifacts_dir=artifacts_dir,
                max_chars=args.max_chars,
                temperature=args.temperature,
                dry_run=args.dry_run,
            )
        )

    result = RefinementOutput(
        created_at=datetime.now().isoformat(timespec="seconds"),
        model=args.model,
        chunks_dir=str(chunks_dir),
        gliner_dir=str(gliner_dir),
        chunks=refined_chunks,
    )

    try:
        payload = result.model_dump(mode="json")
    except ValidationError as exc:
        raise RuntimeError(f"Refinement output failed validation: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    total_entities = sum(chunk.entity_count for chunk in refined_chunks)
    print(f"Wrote {total_entities} refined entities across {len(refined_chunks)} chunk(s) to {output_path}")
    if artifacts_dir:
        print(f"Wrote refinement artifacts to {artifacts_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
