#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "litellm>=1.70.0",
#   "pydantic>=2.7.0",
# ]
# ///
"""Blind masked LLM recall pass after LLM refinement.

This script reads `llm_pass1_entities.json` plus the original chunk files,
masks already-validated entities with asterisks while preserving character
positions, asks an LLM to find only new unmasked neuroscience entities, then
merges pass-1 and pass-2 entities into `master_extracted_entities.json`.

Example:

    uv run scripts/llm_masked_pass.py \
      --llm-pass1 output/gliner/20260602T192415/llm_pass1_entities.json \
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


SYSTEM_PROMPT = """You are an expert neuroscience annotator. You are reviewing a text chunk where previously identified neuroscience entities have been blindly masked out with asterisks *. Your tasks are:

Ignore Masked Regions: Do not attempt to extract or label text hidden beneath the * blocks.

Deep-Pass Discovery: Analyze the remaining unmasked text. Search for newly exposed or subtle neuroscience entities (brain regions, molecular pathways, cell types, or techniques) that might have been overlooked during earlier extraction passes.

Label New Discoveries: If you find a new entity, output it along with its appropriate neuroscience label.

Return the newly discovered entities as a structured JSON list containing the extracted string and the assigned label.

Additional requirements:
- Return only valid JSON.
- Do not include text that contains asterisks.
- Do not infer or recover entities hidden under asterisks.
- Use concise neuroscience labels in PascalCase or UPPER_SNAKE_CASE.
- Target biomedical/neuroscientific concepts that can be mapped to ontology identifiers (IRIs) and labels.
- Useful for diseases, genes, proteins, chemicals, anatomical structures, etc.
- Prefer exact source-text entity strings; downstream concept mapping will pass each term with its source sentence as context.
"""


class NewEntity(BaseModel):
    entity: str = Field(description="Exact unmasked string from the masked text chunk.")
    label: str = Field(description="Appropriate neuroscience label.")

    @field_validator("entity")
    @classmethod
    def clean_entity(cls, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(value)).strip()
        if "*" in cleaned:
            raise ValueError("entity must not contain masked asterisks")
        return cleaned

    @field_validator("label")
    @classmethod
    def clean_label(cls, value: str) -> str:
        cleaned = str(value).strip()
        cleaned = re.sub(r"\s+", "_", cleaned)
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "", cleaned)
        return cleaned


class LLMDiscoveryResponse(BaseModel):
    entities: list[NewEntity] = Field(default_factory=list)


class MasterEntity(BaseModel):
    entity: str
    label: str
    chunk: str
    start: int | None = Field(description="Start offset relative to the original raw chunk body when available.")
    end: int | None = Field(description="End offset relative to the original raw chunk body when available.")
    global_start: int | None = Field(default=None, description="Start offset relative to the parsed full document when chunk metadata is available.")
    global_end: int | None = Field(default=None, description="End offset relative to the parsed full document when chunk metadata is available.")
    source_pass: str
    context: str | None = None
    source_chunk_path: str | None = None
    masked: bool = False


class MaskRecord(BaseModel):
    entity: str
    label: str
    start: int
    end: int
    pass1_start: int | None = None
    pass1_end: int | None = None


class UnmatchedPass1Entity(BaseModel):
    entity: str
    label: str
    pass1_start: int | None = None
    pass1_end: int | None = None
    reason: str


class ChunkMaskedPassResult(BaseModel):
    chunk: str
    source_chunk_path: str
    masked_text_path: str | None = None
    mask_count: int
    unmatched_pass1_count: int
    second_pass_count: int
    masks: list[MaskRecord]
    unmatched_pass1_entities: list[UnmatchedPass1Entity]
    second_pass_entities: list[MasterEntity]


class MasterOutput(BaseModel):
    created_at: str
    model: str
    llm_pass1_path: str
    output_path: str
    total_master_entities: int
    pass1_entity_count: int
    second_pass_entity_count: int
    chunks: list[ChunkMaskedPassResult]
    master_entities: list[MasterEntity]


@dataclass
class MaskingResult:
    masked_text: str
    masks: list[MaskRecord]
    unmatched: list[UnmatchedPass1Entity]
    pass1_master_entities: list[MasterEntity]


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
    parser = argparse.ArgumentParser(description="Run a blind masked LLM recall pass and merge extracted entities.")
    parser.add_argument("--llm-pass1", required=True, type=Path, help="Path to llm_pass1_entities.json.")
    parser.add_argument("--model", default=os.getenv("LITELLM_MODEL", "gpt-4o"), help="LiteLLM model name.")
    parser.add_argument("--output", type=Path, help="Output path. Defaults to <llm-pass1-dir>/master_extracted_entities.json.")
    parser.add_argument("--artifacts-dir", type=Path, help="Optional directory for masked text artifacts. Defaults to <llm-pass1-dir>/llm_masked_pass_artifacts.")
    parser.add_argument("--chunk-index", type=int, action="append", help="Only process the given chunk index; repeatable.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM sampling temperature.")
    parser.add_argument("--dry-run", action="store_true", help="Create masked artifacts and merge pass-1 entities without calling the LLM.")
    return parser.parse_args()


def strip_chunk_header(text: str) -> str:
    if "\n---\n" in text:
        return text.split("\n---\n", 1)[1]
    return text


def read_chunk_body(path: Path) -> str:
    return strip_chunk_header(path.read_text(encoding="utf-8", errors="replace"))


_CHUNK_CHAR_START_CACHE: dict[str, int | None] = {}


def chunk_char_start(source_chunk_path: str | None) -> int | None:
    """Read char_start from a chunk header so overlap duplicates can be deduped globally."""
    if not source_chunk_path:
        return None
    if source_chunk_path in _CHUNK_CHAR_START_CACHE:
        return _CHUNK_CHAR_START_CACHE[source_chunk_path]
    value: int | None = None
    try:
        first_line = Path(source_chunk_path).read_text(encoding="utf-8", errors="replace").splitlines()[0]
        header = json.loads(first_line)
        raw = header.get("char_start")
        if isinstance(raw, int):
            value = raw
    except Exception:
        value = None
    _CHUNK_CHAR_START_CACHE[source_chunk_path] = value
    return value


def global_span(source_chunk_path: str | None, start: int | None, end: int | None) -> tuple[int | None, int | None]:
    base = chunk_char_start(source_chunk_path)
    if base is None or start is None or end is None:
        return None, None
    return base + start, base + end


def chunk_index(chunk_name: str) -> int | None:
    match = re.search(r"chunk_(\d+)$", chunk_name)
    return int(match.group(1)) if match else None


def context_for(text: str, start: int | None, end: int | None, window: int = 100) -> str | None:
    if start is None or end is None or start < 0 or end > len(text) or start >= end:
        return None
    left = max(0, start - window)
    right = min(len(text), end + window)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def occupied_overlap(occupied: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(start < occ_end and end > occ_start for occ_start, occ_end in occupied)


def find_best_unoccupied_exact(text: str, entity: str, preferred_start: int | None, occupied: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not entity:
        return None
    candidates: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(entity, cursor)
        if start == -1:
            break
        end = start + len(entity)
        if not occupied_overlap(occupied, start, end):
            candidates.append((start, end))
        cursor = start + 1

    if not candidates:
        return None
    if preferred_start is None:
        return candidates[0]
    return min(candidates, key=lambda span: abs(span[0] - preferred_start))


def mask_pass1_entities(raw_text: str, chunk_payload: dict[str, Any]) -> MaskingResult:
    chars = list(raw_text)
    occupied: list[tuple[int, int]] = []
    masks: list[MaskRecord] = []
    unmatched: list[UnmatchedPass1Entity] = []
    pass1_master: list[MasterEntity] = []
    chunk_name = str(chunk_payload.get("chunk", ""))
    source_chunk_path = str(chunk_payload.get("source_chunk_path", ""))

    entities = chunk_payload.get("entities", [])
    if not isinstance(entities, list):
        entities = []

    # Longest strings first avoids masking a short substring before a longer entity.
    sorted_entities = sorted(
        [e for e in entities if isinstance(e, dict)],
        key=lambda e: (-(len(str(e.get("entity", "")))), int(e.get("start") or 0)),
    )

    for ent in sorted_entities:
        entity = str(ent.get("entity", "")).strip()
        label = str(ent.get("label", "")).strip()
        pass1_start = ent.get("start") if isinstance(ent.get("start"), int) else None
        pass1_end = ent.get("end") if isinstance(ent.get("end"), int) else None
        if not entity or not label:
            unmatched.append(
                UnmatchedPass1Entity(entity=entity, label=label, pass1_start=pass1_start, pass1_end=pass1_end, reason="empty entity or label")
            )
            continue

        span = find_best_unoccupied_exact(raw_text, entity, pass1_start, occupied)
        if span is None:
            unmatched.append(
                UnmatchedPass1Entity(entity=entity, label=label, pass1_start=pass1_start, pass1_end=pass1_end, reason="exact string not found unoccupied in raw chunk")
            )
            pass1_master.append(
                MasterEntity(
                    entity=entity,
                    label=label,
                    chunk=chunk_name,
                    start=None,
                    end=None,
                    source_pass="llm_pass1_unmapped",
                    context=str(ent.get("context")) if ent.get("context") else None,
                    source_chunk_path=source_chunk_path,
                    masked=False,
                )
            )
            continue

        start, end = span
        for i in range(start, end):
            chars[i] = "*"
        occupied.append((start, end))
        occupied.sort()
        masks.append(MaskRecord(entity=entity, label=label, start=start, end=end, pass1_start=pass1_start, pass1_end=pass1_end))
        global_start, global_end = global_span(source_chunk_path, start, end)
        pass1_master.append(
            MasterEntity(
                entity=raw_text[start:end],
                label=label,
                chunk=chunk_name,
                start=start,
                end=end,
                global_start=global_start,
                global_end=global_end,
                source_pass="llm_pass1",
                context=context_for(raw_text, start, end),
                source_chunk_path=source_chunk_path,
                masked=True,
            )
        )

    masks.sort(key=lambda m: (m.start, m.end))
    pass1_master.sort(key=lambda e: (e.start is None, e.start or 10**12, e.end or 10**12, e.entity))
    return MaskingResult(masked_text="".join(chars), masks=masks, unmatched=unmatched, pass1_master_entities=pass1_master)


def extract_json_object(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(raw[start : end + 1])
        else:
            # Accept a top-level JSON list despite the object schema request.
            list_start = raw.find("[")
            list_end = raw.rfind("]")
            if list_start >= 0 and list_end > list_start:
                parsed = {"entities": json.loads(raw[list_start : list_end + 1])}
            else:
                raise
    if isinstance(parsed, list):
        return {"entities": parsed}
    if isinstance(parsed, dict):
        return parsed
    raise ValueError("LLM JSON response must be an object or list")


def discover_with_llm(model: str, masked_text: str, temperature: float) -> list[NewEntity]:
    schema = LLMDiscoveryResponse.model_json_schema()
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Return JSON matching this schema exactly:\n"
                    f"{json.dumps(schema)}\n\n"
                    "MASKED_TEXT_CHUNK:\n"
                    f"{masked_text}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    raw = response.choices[0].message.content or ""
    try:
        parsed = LLMDiscoveryResponse.model_validate(extract_json_object(raw))
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise RuntimeError(f"LLM did not return valid masked-pass JSON: {exc}\nRaw response:\n{raw}") from exc
    return parsed.entities


def find_second_pass_entities(masked_text: str, raw_text: str, chunk_name: str, source_chunk_path: str, discoveries: list[NewEntity]) -> list[MasterEntity]:
    occupied: list[tuple[int, int]] = []
    results: list[MasterEntity] = []
    for discovery in discoveries:
        entity = discovery.entity
        if not entity:
            continue
        span = find_best_unoccupied_exact(masked_text, entity, None, occupied)
        if span is None:
            # The LLM may normalize whitespace. Keep it for audit, but mark as unmapped.
            results.append(
                MasterEntity(
                    entity=entity,
                    label=discovery.label,
                    chunk=chunk_name,
                    start=None,
                    end=None,
                    source_pass="llm_masked_pass_unmapped",
                    context=None,
                    source_chunk_path=source_chunk_path,
                    masked=False,
                )
            )
            continue
        start, end = span
        # Because masking preserves string length, masked offsets equal raw chunk offsets.
        occupied.append((start, end))
        occupied.sort()
        global_start, global_end = global_span(source_chunk_path, start, end)
        results.append(
            MasterEntity(
                entity=raw_text[start:end],
                label=discovery.label,
                chunk=chunk_name,
                start=start,
                end=end,
                global_start=global_start,
                global_end=global_end,
                source_pass="llm_masked_pass",
                context=context_for(raw_text, start, end),
                source_chunk_path=source_chunk_path,
                masked=False,
            )
        )
    return results


def spans_overlap(a: MasterEntity, b: MasterEntity) -> bool:
    if a.chunk != b.chunk or a.start is None or a.end is None or b.start is None or b.end is None:
        return False
    return a.start < b.end and a.end > b.start


def global_spans_overlap(a: MasterEntity, b: MasterEntity) -> bool:
    if a.global_start is None or a.global_end is None or b.global_start is None or b.global_end is None:
        return False
    return a.global_start < b.global_end and a.global_end > b.global_start


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def deduplicate_entities(entities: list[MasterEntity]) -> list[MasterEntity]:
    """Deduplicate exact same spans and overlapping alternatives, not repeated mentions.

    Repeated mentions of the same entity string in different locations are kept.
    Unmapped entities are deduplicated conservatively by chunk/text/label because
    they have no reliable span.
    """
    deduped: list[MasterEntity] = []
    for ent in sorted(entities, key=lambda e: (e.global_start is None, e.global_start or 10**12, e.chunk, e.start is None, e.start or 10**12, -(len(e.entity)), e.label, e.source_pass)):
        duplicate = False
        for existing in deduped:
            both_global_mapped = ent.global_start is not None and ent.global_end is not None and existing.global_start is not None and existing.global_end is not None
            if both_global_mapped:
                same_global_span = ent.global_start == existing.global_start and ent.global_end == existing.global_end
                compatible_text = normalized_text(ent.entity) == normalized_text(existing.entity)
                compatible_label = ent.label.casefold() == existing.label.casefold()
                if same_global_span or (global_spans_overlap(ent, existing) and compatible_text and compatible_label):
                    duplicate = True
                    break

            if ent.chunk != existing.chunk:
                continue

            both_mapped = ent.start is not None and ent.end is not None and existing.start is not None and existing.end is not None
            if both_mapped:
                same_span = ent.start == existing.start and ent.end == existing.end
                if same_span or spans_overlap(ent, existing):
                    duplicate = True
                    break
            else:
                ent_key = (ent.chunk, normalized_text(ent.entity), ent.label.casefold())
                existing_key = (existing.chunk, normalized_text(existing.entity), existing.label.casefold())
                if ent_key == existing_key:
                    duplicate = True
                    break

        if not duplicate:
            deduped.append(ent)
    return deduped


def write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def main() -> int:
    load_dotenv()
    args = parse_args()

    pass1_path = args.llm_pass1.expanduser().resolve()
    if not pass1_path.is_file():
        raise FileNotFoundError(f"llm_pass1_entities.json not found: {pass1_path}")

    output_path = args.output.expanduser().resolve() if args.output else pass1_path.parent / "master_extracted_entities.json"
    artifacts_dir = args.artifacts_dir.expanduser().resolve() if args.artifacts_dir else pass1_path.parent / "llm_masked_pass_artifacts"

    pass1 = json.loads(pass1_path.read_text(encoding="utf-8"))
    chunks = pass1.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError("llm_pass1_entities.json must contain a top-level 'chunks' array")

    selected = set(args.chunk_index or [])
    chunk_results: list[ChunkMaskedPassResult] = []
    all_master_entities: list[MasterEntity] = []
    second_pass_total = 0
    pass1_total = 0

    for chunk_payload in chunks:
        if not isinstance(chunk_payload, dict):
            continue
        chunk_name = str(chunk_payload.get("chunk", ""))
        idx = chunk_index(chunk_name)
        if selected and idx not in selected:
            continue

        source_chunk_path = Path(str(chunk_payload.get("source_chunk_path", ""))).expanduser()
        if not source_chunk_path.is_file():
            print(f"⚠ skipping {chunk_name}: source chunk missing: {source_chunk_path}", file=sys.stderr)
            continue

        raw_text = read_chunk_body(source_chunk_path)
        masking = mask_pass1_entities(raw_text, chunk_payload)
        masked_path = write_text(artifacts_dir / "masked_text" / f"{chunk_name}.txt", masking.masked_text)

        print(f"Masked {chunk_name}: {len(masking.masks)} entities, {len(masking.unmatched)} unmatched", file=sys.stderr)

        if args.dry_run:
            discoveries: list[NewEntity] = []
        else:
            discoveries = discover_with_llm(args.model, masking.masked_text, args.temperature)

        second_pass_entities = find_second_pass_entities(
            masking.masked_text,
            raw_text,
            chunk_name,
            str(source_chunk_path),
            discoveries,
        )
        second_pass_total += len(second_pass_entities)
        pass1_total += len(masking.pass1_master_entities)
        all_master_entities.extend(masking.pass1_master_entities)
        all_master_entities.extend(second_pass_entities)

        chunk_results.append(
            ChunkMaskedPassResult(
                chunk=chunk_name,
                source_chunk_path=str(source_chunk_path),
                masked_text_path=masked_path,
                mask_count=len(masking.masks),
                unmatched_pass1_count=len(masking.unmatched),
                second_pass_count=len(second_pass_entities),
                masks=masking.masks,
                unmatched_pass1_entities=masking.unmatched,
                second_pass_entities=second_pass_entities,
            )
        )

    master_entities = deduplicate_entities(all_master_entities)
    result = MasterOutput(
        created_at=datetime.now().isoformat(timespec="seconds"),
        model=args.model,
        llm_pass1_path=str(pass1_path),
        output_path=str(output_path),
        total_master_entities=len(master_entities),
        pass1_entity_count=pass1_total,
        second_pass_entity_count=second_pass_total,
        chunks=chunk_results,
        master_entities=master_entities,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(master_entities)} master entities to {output_path}")
    print(f"Wrote masked-pass artifacts to {artifacts_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
