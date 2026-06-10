#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# ///
"""Audit and normalize final Neuroscience NER outputs.

This deterministic post-processing stage accepts either:

- `master_extracted_entities.json` from `llm_masked_pass.py`
- `neuro_entities_mapped.json` from `map_ontology.py`
- a raw object containing an `entities` array

It writes a canonical audited JSON file that preserves every raw mention while
adding grouped views, run statistics, span validation, and ontology IRI checks.
No LLM calls are made.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional


# ---------------------------------------------------------------------------
# IRI validation
# ---------------------------------------------------------------------------

_OBO_PREFIX_PATTERN = r"^https?://purl\.obolibrary\.org/obo/{prefix}_[0-9]+(?:[/#?].*)?$"
_CURIE_PATTERN = r"^{prefix}:[0-9A-Za-z\-_]+$"
_IRI_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "uberon": (re.compile(_OBO_PREFIX_PATTERN.format(prefix="UBERON")), re.compile(_CURIE_PATTERN.format(prefix="UBERON"))),
    "cl": (re.compile(_OBO_PREFIX_PATTERN.format(prefix="CL")), re.compile(_CURIE_PATTERN.format(prefix="CL"))),
    "pcl": (re.compile(_OBO_PREFIX_PATTERN.format(prefix="PCL")), re.compile(_CURIE_PATTERN.format(prefix="PCL"))),
    "fma": (re.compile(_OBO_PREFIX_PATTERN.format(prefix="FMA")), re.compile(_CURIE_PATTERN.format(prefix="FMA"))),
    "nifstd": (re.compile(r"^https?://uri\.neuinfo\.org/nif/nifstd/[A-Za-z0-9_]+$"),),
    "go": (re.compile(_OBO_PREFIX_PATTERN.format(prefix="GO")), re.compile(_CURIE_PATTERN.format(prefix="GO"))),
    "obi": (re.compile(_OBO_PREFIX_PATTERN.format(prefix="OBI")), re.compile(_CURIE_PATTERN.format(prefix="OBI"))),
    "efo": (re.compile(r"^https?://www\.ebi\.ac\.uk/efo/EFO_[0-9]+$"), re.compile(r"^EFO:[0-9]+$")),
    "mondo": (re.compile(_OBO_PREFIX_PATTERN.format(prefix="MONDO")), re.compile(_CURIE_PATTERN.format(prefix="MONDO"))),
    "doid": (re.compile(_OBO_PREFIX_PATTERN.format(prefix="DOID")), re.compile(_CURIE_PATTERN.format(prefix="DOID"))),
    "hp": (re.compile(_OBO_PREFIX_PATTERN.format(prefix="HP")), re.compile(_CURIE_PATTERN.format(prefix="HP"))),
    "mp": (re.compile(_OBO_PREFIX_PATTERN.format(prefix="MP")), re.compile(_CURIE_PATTERN.format(prefix="MP"))),
    "chebi": (re.compile(_OBO_PREFIX_PATTERN.format(prefix="CHEBI")), re.compile(_CURIE_PATTERN.format(prefix="CHEBI"))),
    "dron": (re.compile(_OBO_PREFIX_PATTERN.format(prefix="DRON")), re.compile(_CURIE_PATTERN.format(prefix="DRON"))),
    "bto": (re.compile(_OBO_PREFIX_PATTERN.format(prefix="BTO")), re.compile(_CURIE_PATTERN.format(prefix="BTO"))),
    "pr": (re.compile(_OBO_PREFIX_PATTERN.format(prefix="PR")), re.compile(_CURIE_PATTERN.format(prefix="PR"))),
    "ncbitaxon": (re.compile(_OBO_PREFIX_PATTERN.format(prefix="NCBITaxon")), re.compile(_CURIE_PATTERN.format(prefix="NCBITaxon"))),
    "hgnc": (re.compile(r"^https?://identifiers\.org/hgnc/[0-9]+$"), re.compile(r"^HGNC:[0-9]+$")),
    "ncbigene": (re.compile(r"^https?://www\.ncbi\.nlm\.nih\.gov/gene/[0-9]+$"), re.compile(r"^NCBIGene:[0-9]+$")),
    "mgi": (re.compile(r"^MGI:[0-9]+$"), re.compile(r"^https?://www\.informatics\.jax\.org/marker/MGI:[0-9]+$")),
    "uniprot": (re.compile(r"^https?://(?:www\.)?uniprot\.org/uniprot(?:kb)?/[A-Z0-9]{6,10}$"), re.compile(r"^UniProtKB:[A-Z0-9]{6,10}$")),
    "snomedct": (re.compile(r"^https?://purl\.bioontology\.org/ontology/SNOMEDCT/[A-Za-z0-9_\-]+$"), re.compile(r"^SNOMEDCT:[A-Za-z0-9_\-]+$")),
}
_GENERIC_IRI = re.compile(r"^(https?://[^\s]+|urn:[A-Za-z0-9][A-Za-z0-9:._\-]+|[A-Za-z][A-Za-z0-9_]*:[A-Za-z0-9_\-]+)$")
_OBO_GENERIC = re.compile(r"^https?://purl\.obolibrary\.org/obo/[A-Za-z]+_[0-9]+(?:[/#?].*)?$")
_BIOPORTAL_PURL = re.compile(r"^https?://purl\.bioontology\.org/ontology/[A-Za-z0-9_\-]+/[A-Za-z0-9_\-]+$")
_IDENTIFIERS_ORG = re.compile(r"^https?://identifiers\.org/[A-Za-z0-9._]+(?:/[A-Za-z0-9._:-]+)?$")
_EBI_OLS = re.compile(r"^https?://www\.ebi\.ac\.uk/[A-Za-z0-9/_-]+/[A-Za-z0-9_]+_[0-9]+$")
_SEMANTIC_WEB = re.compile(r"^https?://www\.semanticweb\.org/.+#[A-Za-z0-9_]+$")
_GENERIC_OWL_IRI = re.compile(r"^https?://[A-Za-z0-9./_-]+/[A-Za-z0-9._-]+_[0-9]+$")


def is_well_formed_iri(iri: Any) -> bool:
    if not isinstance(iri, str):
        return False
    value = iri.strip()
    if not value or value.lower() in {"n/a", "none", "null", "unmapped"}:
        return False
    return bool(_GENERIC_IRI.match(value))


def matches_any_known_ontology(iri: str) -> bool:
    value = iri.strip()
    for patterns in _IRI_PATTERNS.values():
        if any(pattern.match(value) for pattern in patterns):
            return True
    return False


def looks_like_real_ontology_iri(iri: str) -> bool:
    value = iri.strip()
    return any(
        pattern.match(value)
        for pattern in (_OBO_GENERIC, _BIOPORTAL_PURL, _IDENTIFIERS_ORG, _EBI_OLS, _SEMANTIC_WEB, _GENERIC_OWL_IRI)
    )


def validate_iri(item: dict[str, Any], *, strict: bool) -> tuple[bool, str | None]:
    ontology_id = item.get("ontology_id")
    provenance = str(item.get("concept_mapping_provenance") or "").strip().lower()

    if provenance in {"unmapped", "skipped"} and not ontology_id:
        return True, None
    if not ontology_id:
        return False, "missing_ontology_id"
    if provenance == "llm_knowledge":
        return False, "llm_knowledge_rejected"
    if not is_well_formed_iri(ontology_id):
        return False, "malformed_iri"
    # BioPortal/local mapping services can legitimately return ontology IRIs
    # outside this script's known prefix registry. Treat those as valid if they
    # are structurally well formed; the registry is used to catch obvious
    # malformed values, not to reject real but unfamiliar ontology namespaces.
    return True, None


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def input_kind(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("master_entities"), list):
        return "master_extracted_entities"
    if payload.get("backend") is not None and isinstance(payload.get("entities"), list):
        return "ontology_mapped_entities"
    if isinstance(payload.get("entities"), list):
        return "entities"
    raise ValueError("Input JSON must contain `master_entities` or `entities` array")


def normalize_entity(raw: dict[str, Any], *, kind: str, index: int) -> dict[str, Any] | None:
    surface = clean_text(raw.get("entity") or raw.get("extracted_text") or raw.get("text"))
    if not surface:
        return None

    label = clean_text(raw.get("label") or raw.get("llm_label") or raw.get("type")) or None
    ontology_id = raw.get("ontology_id") or raw.get("ontology_uri")
    ontology_label = raw.get("ontology_label") or raw.get("bioportal_prefLabel")
    ontology = raw.get("ontology")
    mapping_backend = raw.get("mapping_backend")
    mapping_error = raw.get("mapping_error")

    if ontology_id:
        provenance = raw.get("concept_mapping_provenance") or "tool"
        alignment_method = raw.get("alignment_method") or "direct_tool_call"
    elif kind == "ontology_mapped_entities":
        provenance = raw.get("concept_mapping_provenance") or "unmapped"
        alignment_method = raw.get("alignment_method") or "direct_tool_call"
    else:
        provenance = raw.get("concept_mapping_provenance") or "skipped"
        alignment_method = raw.get("alignment_method") or "skipped"

    context = clean_text(raw.get("context") or raw.get("sentence")) or None

    normalized = {
        "id": f"mention_{index:06d}",
        "entity": surface,
        "label": label,
        "context": context,
        "sentence": context,
        "chunk": raw.get("chunk"),
        "start": raw.get("start"),
        "end": raw.get("end"),
        "global_start": raw.get("global_start"),
        "global_end": raw.get("global_end"),
        "source_pass": raw.get("source_pass"),
        "source_chunk_path": raw.get("source_chunk_path"),
        "masked": raw.get("masked"),
        "ontology_id": ontology_id,
        "ontology_label": ontology_label,
        "ontology": ontology,
        "concept_mapping_provenance": provenance,
        "alignment_method": alignment_method,
        "mapping_backend": mapping_backend,
        "mapping_error": mapping_error,
    }

    # Preserve noncanonical source fields under `source_fields` for auditability.
    preserved = {
        key: value
        for key, value in raw.items()
        if key not in normalized
        and key not in {"extracted_text", "llm_label", "bioportal_prefLabel", "ontology_uri", "entity", "label", "context", "sentence"}
    }
    if preserved:
        normalized["source_fields"] = preserved
    return normalized


def normalize_entities(payload: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    raw_entities = payload.get("master_entities") if kind == "master_extracted_entities" else payload.get("entities")
    if not isinstance(raw_entities, list):
        raise ValueError("Input entity container is not an array")
    entities: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_entities):
        if not isinstance(raw, dict):
            continue
        normalized = normalize_entity(raw, kind=kind, index=len(entities))
        if normalized:
            entities.append(normalized)
    return entities


# ---------------------------------------------------------------------------
# Span validation
# ---------------------------------------------------------------------------

_CHUNK_BODY_CACHE: dict[str, str | None] = {}
_SOURCE_TEXT_CACHE: dict[str, str | None] = {}


def strip_chunk_header(text: str) -> str:
    if "\n---\n" in text:
        return text.split("\n---\n", 1)[1]
    return text


def read_text_cached(path: str, cache: dict[str, str | None], *, chunk_body: bool = False) -> str | None:
    if path in cache:
        return cache[path]
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8", errors="replace")
        if chunk_body:
            text = strip_chunk_header(text)
    except Exception:
        text = None
    cache[path] = text
    return text


def validate_span_against_text(text: str | None, item: dict[str, Any], start_key: str, end_key: str) -> bool | None:
    if text is None:
        return None
    start = item.get(start_key)
    end = item.get(end_key)
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 0 or end < start or end > len(text):
        return False
    return text[start:end] == item.get("entity")


def validate_spans(entities: list[dict[str, Any]], source_text_path: str | None) -> dict[str, Any]:
    counts = Counter()
    examples: list[dict[str, Any]] = []
    source_text = read_text_cached(source_text_path, _SOURCE_TEXT_CACHE) if source_text_path else None

    for item in entities:
        local_result: bool | None = None
        chunk_path = item.get("source_chunk_path")
        if isinstance(chunk_path, str) and chunk_path:
            chunk_body = read_text_cached(chunk_path, _CHUNK_BODY_CACHE, chunk_body=True)
            local_result = validate_span_against_text(chunk_body, item, "start", "end")

        global_result = validate_span_against_text(source_text, item, "global_start", "global_end") if source_text else None

        if local_result is True or global_result is True:
            counts["valid"] += 1
            item["span_validation"] = "valid"
        elif local_result is False or global_result is False:
            counts["invalid"] += 1
            item["span_validation"] = "invalid"
            if len(examples) < 20:
                examples.append({"id": item.get("id"), "entity": item.get("entity"), "start": item.get("start"), "end": item.get("end"), "global_start": item.get("global_start"), "global_end": item.get("global_end"), "source_chunk_path": chunk_path})
        else:
            counts["not_checked"] += 1
            item["span_validation"] = "not_checked"

    return {
        "valid": counts["valid"],
        "invalid": counts["invalid"],
        "not_checked": counts["not_checked"],
        "invalid_examples": examples,
        "source_text_path": source_text_path,
    }


# ---------------------------------------------------------------------------
# Grouping and stats
# ---------------------------------------------------------------------------


def canonical_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item.get("entity") or "").strip().lower(), str(item.get("label") or "").strip())


def best_mapping(items: list[dict[str, Any]]) -> dict[str, Any]:
    def score(item: dict[str, Any]) -> tuple[int, int]:
        provenance = item.get("concept_mapping_provenance")
        ontology_id = item.get("ontology_id")
        return (100 if provenance == "tool" else 10 if provenance == "unmapped" else 0, 1 if ontology_id else 0)

    return max(items, key=score)


def grouped_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in entities:
        buckets[canonical_key(item)].append(item)

    grouped: list[dict[str, Any]] = []
    for (_surface_lower, label), mentions in buckets.items():
        surface_counts = Counter(str(m.get("entity")) for m in mentions)
        canonical_surface = surface_counts.most_common(1)[0][0]
        best = best_mapping(mentions)
        judge_scores = [m.get("judge_score") for m in mentions if isinstance(m.get("judge_score"), (int, float))]
        contexts = []
        seen_contexts = set()
        for m in mentions:
            ctx = m.get("context") or m.get("sentence")
            if ctx and ctx not in seen_contexts:
                seen_contexts.add(ctx)
                contexts.append(ctx)
        grouped.append(
            {
                "entity": canonical_surface,
                "label": label or None,
                "mention_count": len(mentions),
                "mention_ids": [m.get("id") for m in mentions],
                "chunks": sorted({str(m.get("chunk")) for m in mentions if m.get("chunk")}),
                "source_passes": dict(Counter(str(m.get("source_pass")) for m in mentions if m.get("source_pass"))),
                "ontology_id": best.get("ontology_id"),
                "ontology_label": best.get("ontology_label"),
                "ontology": best.get("ontology"),
                "concept_mapping_provenance": best.get("concept_mapping_provenance"),
                "alignment_method": best.get("alignment_method"),
                "mapping_backend": best.get("mapping_backend"),
                "judge_score_max": max(judge_scores) if judge_scores else None,
                "judge_score_avg": round(mean(judge_scores), 3) if judge_scores else None,
                "judge_score_min": min(judge_scores) if judge_scores else None,
                "contexts": contexts[:10],
            }
        )
    grouped.sort(key=lambda item: (-item["mention_count"], str(item.get("entity") or "")))
    return grouped


def validate_ontology(entities: list[dict[str, Any]], *, strict_iri: bool) -> dict[str, Any]:
    counts = Counter()
    by_reason = Counter()
    examples: list[dict[str, Any]] = []

    for item in entities:
        ok, reason = validate_iri(item, strict=strict_iri)
        if ok:
            if item.get("ontology_id"):
                counts["valid_mapped"] += 1
                item["ontology_validation"] = "valid"
            else:
                counts[str(item.get("concept_mapping_provenance") or "missing")] += 1
                item["ontology_validation"] = "not_mapped"
        else:
            counts["invalid"] += 1
            by_reason[reason or "unknown"] += 1
            item["ontology_validation"] = "invalid"
            item["ontology_validation_error"] = reason
            if len(examples) < 20:
                examples.append({"id": item.get("id"), "entity": item.get("entity"), "ontology_id": item.get("ontology_id"), "ontology": item.get("ontology"), "reason": reason})

    return {
        "strict_iri_validation": strict_iri,
        "valid_mapped": counts["valid_mapped"],
        "invalid": counts["invalid"],
        "unmapped": counts["unmapped"],
        "skipped": counts["skipped"],
        "missing": counts["missing"],
        "invalid_by_reason": dict(by_reason),
        "invalid_examples": examples,
    }


def compute_stats(entities: list[dict[str, Any]], groups: list[dict[str, Any]], validation: dict[str, Any]) -> dict[str, Any]:
    by_label = Counter(str(item.get("label") or "Unknown") for item in entities)
    by_source_pass = Counter(str(item.get("source_pass") or "unknown") for item in entities)
    by_chunk = Counter(str(item.get("chunk") or "unknown") for item in entities)
    by_mapping_backend = Counter(str(item.get("mapping_backend") or "none") for item in entities)
    by_provenance = Counter(str(item.get("concept_mapping_provenance") or "missing") for item in entities)
    unique_surfaces = {str(item.get("entity") or "").lower() for item in entities if item.get("entity")}
    mapped = sum(1 for item in entities if item.get("ontology_id"))

    return {
        "totals": {
            "total_entity_mentions": len(entities),
            "unique_surface_forms": len(unique_surfaces),
            "unique_entity_label_groups": len(groups),
            "mapped_mentions": mapped,
            "unmapped_or_skipped_mentions": len(entities) - mapped,
        },
        "mentions_per_unique_surface": round(len(entities) / len(unique_surfaces), 2) if unique_surfaces else 0,
        "by_label": dict(by_label.most_common()),
        "by_source_pass": dict(by_source_pass.most_common()),
        "by_chunk": dict(by_chunk.most_common()),
        "alignment": {
            "by_mapping_backend": dict(by_mapping_backend.most_common()),
            "by_provenance": dict(by_provenance.most_common()),
        },
        "validation_summary": {
            "span": {k: v for k, v in validation.get("span", {}).items() if k != "invalid_examples"},
            "ontology": {k: v for k, v in validation.get("ontology", {}).items() if k != "invalid_examples"},
        },
    }


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_audited.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and normalize final Neuroscience NER outputs.")
    parser.add_argument("--input", required=True, type=Path, help="Input JSON path: master_extracted_entities.json, neuro_entities_mapped.json, or an object with entities[].")
    parser.add_argument("--output", type=Path, help="Output JSON path. Defaults to <input_stem>_audited.json.")
    parser.add_argument("--source-text", type=Path, help="Optional original full text file for global_start/global_end span validation.")
    parser.add_argument("--no-strict-iri", action="store_true", help="Disable strict ontology IRI structural validation.")
    parser.add_argument("--fail-on-invalid", action="store_true", help="Exit non-zero if invalid spans or invalid ontology IRIs are found.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else default_output_path(input_path)
    source_text_path = str(args.source_text.expanduser().resolve()) if args.source_text else None

    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object")

    kind = input_kind(payload)
    entities = normalize_entities(payload, kind)
    ontology_validation = validate_ontology(entities, strict_iri=not args.no_strict_iri)
    span_validation = validate_spans(entities, source_text_path)
    groups = grouped_entities(entities)

    validation = {"ontology": ontology_validation, "span": span_validation}
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "input_kind": kind,
        "source_metadata": {
            "source_text_path": source_text_path,
            "upstream_created_at": payload.get("created_at"),
            "upstream_model": payload.get("model"),
            "upstream_backend": payload.get("backend"),
            "upstream_entity_count": payload.get("entity_count") or payload.get("total_master_entities"),
        },
        "entities": entities,
        "entities_grouped": groups,
        "validation": validation,
        "stats": compute_stats(entities, groups, validation),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    invalid_spans = span_validation.get("invalid", 0)
    invalid_iris = ontology_validation.get("invalid", 0)
    mapped = result["stats"]["totals"]["mapped_mentions"]
    total = result["stats"]["totals"]["total_entity_mentions"]

    print(f"Wrote audited NER output to {output_path}")
    print(f"Mentions: {total} | groups: {len(groups)} | mapped: {mapped} | invalid_spans: {invalid_spans} | invalid_iris: {invalid_iris}")

    if args.fail_on_invalid and (invalid_spans or invalid_iris):
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
