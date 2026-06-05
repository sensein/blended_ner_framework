#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pydantic>=2.7.0",
#   "python-dotenv>=1.0.1",
#   "requests>=2.32.0",
#   "tenacity>=8.2.3",
# ]
# ///
"""Map extracted NER entities to ontology identifiers.

This script is a direct, deterministic migration of the useful logic from the
CrewAI ConceptMappingTool and ConceptMappingLocalTool implementations, without
CrewAI Agent/Task/Crew overhead.

It supports two mapping backends:

1. local:    POST <LOCAL_CONCEPT_MAPPING_URL>/map/batch
2. bioportal: BioPortal Search API at http://data.bioontology.org/search

Input is the `master_extracted_entities.json` file produced by the masked-pass
pipeline. Default JSON output is `neuro_entities_mapped.json`; optional CSV
output can be requested with `--csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MapOntology")

MAX_QUERY_LENGTH = 500
MAX_CONTEXT_LEN = 200
DEFAULT_BIOPORTAL_ONTOLOGIES = "UBERON,NIFSTD,FMA,GO,SNOMEDCT"
BIOPORTAL_SEARCH_URL = "http://data.bioontology.org/search"

_CONCEPT_MAPPING_CACHE: dict[str, dict[str, Any]] = {}
_CONCEPT_MAPPING_CACHE_LOCK = threading.Lock()


class RetryableBioPortalError(RuntimeError):
    """Raised for retryable BioPortal status codes such as 429 and 5xx."""


class EnrichedEntity(BaseModel):
    extracted_text: str
    llm_label: str | None = None
    bioportal_prefLabel: str | None = None
    ontology_uri: str | None = None
    ontology: str | None = None
    mapping_backend: str
    mapping_error: str | None = None
    chunk: str | None = None
    start: int | None = None
    end: int | None = None
    global_start: int | None = None
    global_end: int | None = None
    source_pass: str | None = None
    context: str | None = None
    source_chunk_path: str | None = None

    model_config = {"extra": "allow"}


class NeuroEntitiesMappedOutput(BaseModel):
    created_at: str
    input_path: str
    backend: str
    ontologies: str | None = None
    max_results: int
    entity_count: int
    mapped_count: int
    entities: list[EnrichedEntity]


def load_environment() -> None:
    """Load repo-root .env with python-dotenv. Never hardcode API keys."""
    load_dotenv(dotenv_path=Path(".env"), override=False)


def _sanitize_text(text: Optional[str]) -> str:
    """Sanitize text as in the CrewAI concept mapping tools."""
    if text is None:
        return ""
    s = str(text)
    s = s.strip().replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = "".join(c for c in s if c.isprintable() or c.isspace())
    s = s.strip()
    if len(s) > MAX_QUERY_LENGTH:
        s = s[:MAX_QUERY_LENGTH].rstrip()
    return s


def _normalize_max_results(value: Any, default: int = 1) -> int:
    if value is None:
        value = os.getenv("MAX_CONCEPT_MAPPING_RESULTS", str(default))
    if isinstance(value, list):
        value = value[0] if value else default
    try:
        return max(1, min(int(value), 20))
    except (TypeError, ValueError):
        return default


def _concept_mapping_cache_max_size() -> int:
    try:
        return max(100, int(os.getenv("CONCEPT_MAPPING_CACHE_SIZE", "2000")))
    except (TypeError, ValueError):
        return 2000


def _local_base_url() -> str:
    return os.getenv("LOCAL_CONCEPT_MAPPING_URL", "http://localhost:8000").rstrip("/")


def _local_api_key() -> Optional[str]:
    v = os.getenv("LOCAL_CONCEPT_MAPPING_API_KEY", "").strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
    return v if v else None


def _local_model() -> Optional[str]:
    v = os.getenv("LOCAL_CONCEPT_MAPPING_MODEL", "").strip() or os.getenv("OPENROUTER_MODEL", "").strip()
    return v if v else None


def _local_timeout() -> float:
    try:
        return max(1.0, float(os.getenv("LOCAL_CONCEPT_MAPPING_TIMEOUT", "30")))
    except (TypeError, ValueError):
        return 30.0


def parse_master_entities(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entities = payload.get("master_entities") or payload.get("entities")
    if not isinstance(entities, list):
        raise ValueError("Input JSON must contain 'master_entities' or 'entities' array")
    valid: list[dict[str, Any]] = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        text = _sanitize_text(ent.get("entity") or ent.get("extracted_text"))
        if text:
            copied = dict(ent)
            copied["entity"] = text
            valid.append(copied)
    return valid


def context_for_entity(entity: dict[str, Any]) -> str | None:
    ctx = entity.get("context")
    if ctx:
        return _sanitize_text(str(ctx))[:MAX_CONTEXT_LEN]
    return None


def build_term_objects(entities: list[dict[str, Any]]) -> list[dict[str, str | None]]:
    term_objects: list[dict[str, str | None]] = []
    for ent in entities:
        text = _sanitize_text(ent.get("entity"))
        if text:
            term_objects.append({"text": text, "context": context_for_entity(ent)})
    return term_objects


def cache_get(cache_key: str) -> dict[str, Any] | None:
    with _CONCEPT_MAPPING_CACHE_LOCK:
        value = _CONCEPT_MAPPING_CACHE.get(cache_key)
        return dict(value) if value else None


def cache_put(cache_key: str, mapping: dict[str, Any]) -> None:
    with _CONCEPT_MAPPING_CACHE_LOCK:
        if len(_CONCEPT_MAPPING_CACHE) >= _concept_mapping_cache_max_size():
            _CONCEPT_MAPPING_CACHE.pop(next(iter(_CONCEPT_MAPPING_CACHE)), None)
        _CONCEPT_MAPPING_CACHE[cache_key] = dict(mapping)


class LocalOntologyMapper:
    """Direct migration of ConceptMappingLocalTool's efficient /map/batch logic."""

    def __init__(self) -> None:
        self.session = requests.Session()

    def _post_batch(self, term_objects: list[dict[str, str | None]], max_results: int) -> Optional[dict[str, Any]]:
        url = f"{_local_base_url()}/map/batch"
        text_payload = []
        for obj in term_objects:
            item: dict[str, str] = {"text": str(obj["text"])}
            ctx = obj.get("context") or ""
            if ctx:
                item["context"] = ctx[:MAX_CONTEXT_LEN]
            text_payload.append(item)

        payload: dict[str, Any] = {"text": text_payload, "max_results": max_results}
        api_key = _local_api_key()
        if api_key:
            payload["openrouter_api_key"] = api_key
        model = _local_model()
        if model:
            payload["openrouter_model"] = model

        logger.info("Sending local ontology batch to %s | %d term(s) | max_results=%d", url, len(term_objects), max_results)
        try:
            resp = self.session.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=_local_timeout(),
            )
        except Exception as exc:
            logger.error("Local concept mapping request to %s failed: %s", url, exc)
            return None

        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception as exc:
                logger.error("Failed to parse JSON from %s: %s", url, exc)
                return None

        logger.warning("Local concept mapping service returned HTTP %s | response: %s", resp.status_code, resp.text[:500])
        return None

    @staticmethod
    def _result_items_to_mapping(items: list[Any], max_results: int) -> dict[str, Any]:
        valid = [it for it in items if isinstance(it, dict) and it.get("ontology_id") and it.get("ontology_label")][:max_results]
        if not valid:
            return {"error": "No mapping found", "ontology_id": None, "ontology_label": None, "ontology": None}
        item = valid[0]
        return {
            "ontology_id": item["ontology_id"],
            "ontology_label": item["ontology_label"],
            "ontology": item.get("ontology", ""),
        }

    def map_terms(self, term_objects: list[dict[str, str | None]], max_results: int) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        uncached: list[dict[str, str | None]] = []

        for obj in term_objects:
            term = str(obj["text"])
            cache_key = f"local|{term}|{max_results}"
            cached = cache_get(cache_key)
            if cached is not None:
                results[term] = cached
            else:
                uncached.append(obj)

        seen_texts: set[str] = set()
        deduped_uncached: list[dict[str, str | None]] = []
        for obj in uncached:
            term = str(obj["text"])
            if term not in seen_texts:
                seen_texts.add(term)
                deduped_uncached.append(obj)
        uncached = deduped_uncached

        if not uncached:
            return results

        try:
            batch_size = max(1, int(os.getenv("LOCAL_CONCEPT_MAPPING_BATCH_SIZE", "4000")))
        except (TypeError, ValueError):
            batch_size = 4000
        try:
            max_workers = max(1, int(os.getenv("LOCAL_CONCEPT_MAPPING_WORKERS", "4")))
        except (TypeError, ValueError):
            max_workers = 4

        n_batches = math.ceil(len(uncached) / batch_size)
        sub_batches = [uncached[i * batch_size : (i + 1) * batch_size] for i in range(n_batches)]

        def _fetch_sub_batch(batch: list[dict[str, str | None]]) -> dict[str, Any]:
            raw = self._post_batch(batch, max_results)
            if raw and isinstance(raw.get("results"), dict):
                return raw["results"]
            return {}

        per_term: dict[str, Any] = {}
        if len(sub_batches) == 1:
            per_term = _fetch_sub_batch(sub_batches[0])
        else:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(sub_batches))) as executor:
                futures = {executor.submit(_fetch_sub_batch, sb): sb for sb in sub_batches}
                for future in as_completed(futures):
                    try:
                        per_term.update(future.result())
                    except Exception as exc:
                        logger.warning("Sub-batch request failed: %s", exc)

        for obj in uncached:
            term = str(obj["text"])
            items = per_term.get(term)
            if isinstance(items, list):
                mapping = self._result_items_to_mapping(items, max_results)
            else:
                mapping = {"error": f"No mapping returned for: {term}", "ontology_id": None, "ontology_label": None, "ontology": None}
            results[term] = mapping
            cache_put(f"local|{term}|{max_results}", mapping)
        return results


class BioPortalOntologyMapper:
    """BioPortal mapper with exact-match first, fuzzy fallback, retries, and ontology filtering."""

    def __init__(self, api_key: Optional[str] = None, search_url: str = BIOPORTAL_SEARCH_URL) -> None:
        self.api_key = api_key or os.getenv("BIOPORTAL_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("BIOPORTAL_API_KEY not set; place it in .env or your shell environment")
        self.search_url = search_url
        self.session = requests.Session()

    @retry(
        retry=retry_if_exception_type(RetryableBioPortalError),
        wait=wait_exponential(multiplier=2, min=2, max=8),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _search_request(self, params: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"apikey token={self.api_key}"}
        response = self.session.get(self.search_url, params=params, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429 or response.status_code >= 500:
            raise RetryableBioPortalError(f"BioPortal retryable HTTP {response.status_code}: {response.text[:300]}")
        logger.warning("BioPortal non-retryable HTTP %s for q=%r: %s", response.status_code, params.get("q"), response.text[:300])
        return {}

    def search(self, term: str, ontologies: str, max_results: int) -> dict[str, Any]:
        query = _sanitize_text(term)
        if not query:
            return {"error": "No valid text to map", "ontology_id": None, "ontology_label": None, "ontology": None}

        cache_key = f"bioportal|{query}|{ontologies}|{max_results}"
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        base_params: dict[str, Any] = {
            "q": query,
            "ontologies": ontologies,
            "pagesize": max_results,
            "also_search_obsolete": "false",
        }

        # First exact match, then fuzzy fallback.
        for exact in (True, False):
            params = dict(base_params)
            if exact:
                params["exact_match"] = "true"
            try:
                result = self._search_request(params)
            except RetryableBioPortalError as exc:
                mapping = {"error": str(exc), "ontology_id": None, "ontology_label": None, "ontology": None}
                cache_put(cache_key, mapping)
                return mapping

            collection = result.get("collection") if isinstance(result, dict) else None
            if isinstance(collection, list) and collection:
                item = next((x for x in collection if isinstance(x, dict) and (x.get("@id") or x.get("iri")) and x.get("prefLabel")), None)
                if item:
                    links = item.get("links")
                    ontology = None
                    if isinstance(links, dict) and isinstance(links.get("ontology"), str):
                        ontology = links["ontology"].rstrip("/").split("/")[-1]
                    mapping = {
                        "ontology_id": item.get("@id") or item.get("iri"),
                        "ontology_label": item.get("prefLabel"),
                        "ontology": ontology,
                        "exact_match": exact,
                    }
                    cache_put(cache_key, mapping)
                    return mapping

        mapping = {"error": f"No BioPortal match found for: {term}", "ontology_id": None, "ontology_label": None, "ontology": None}
        cache_put(cache_key, mapping)
        return mapping

    def map_terms(self, term_objects: list[dict[str, str | None]], max_results: int, ontologies: str) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        for obj in term_objects:
            term = str(obj["text"])
            if term in seen:
                continue
            seen.add(term)
            results[term] = self.search(term, ontologies, max_results)
        return results


def mapping_succeeded(mapping: dict[str, Any]) -> bool:
    return bool(mapping.get("ontology_id") and mapping.get("ontology_label") and not mapping.get("error"))


def annotate_backend(mappings: dict[str, dict[str, Any]], backend: str) -> dict[str, dict[str, Any]]:
    annotated: dict[str, dict[str, Any]] = {}
    for term, mapping in mappings.items():
        copied = dict(mapping)
        copied["mapping_backend"] = backend
        annotated[term] = copied
    return annotated


def map_with_backend(term_objects: list[dict[str, str | None]], backend: str, max_results: int, ontologies: str) -> tuple[str, dict[str, dict[str, Any]]]:
    if backend == "local":
        return "local", annotate_backend(LocalOntologyMapper().map_terms(term_objects, max_results), "local")

    if backend == "bioportal":
        return "bioportal", annotate_backend(BioPortalOntologyMapper().map_terms(term_objects, max_results, ontologies), "bioportal")

    if backend != "auto":
        raise ValueError(f"Unsupported backend: {backend}")

    # Auto mode is per-term: use local results where they succeed, and fall back
    # to BioPortal only for terms that local did not map.
    local_results = annotate_backend(LocalOntologyMapper().map_terms(term_objects, max_results), "local")
    unique_terms: dict[str, dict[str, str | None]] = {str(obj["text"]): obj for obj in term_objects}
    fallback_terms = [obj for term, obj in unique_terms.items() if not mapping_succeeded(local_results.get(term, {}))]

    if fallback_terms:
        logger.warning("Local ontology mapping failed for %d/%d unique term(s); trying BioPortal per-term fallback", len(fallback_terms), len(unique_terms))
        try:
            bioportal_results = annotate_backend(BioPortalOntologyMapper().map_terms(fallback_terms, max_results, ontologies), "bioportal")
        except Exception as exc:
            logger.warning("BioPortal fallback unavailable; leaving %d local failures unmapped: %s", len(fallback_terms), exc)
            bioportal_results = {}
        for term, bioportal_mapping in bioportal_results.items():
            if mapping_succeeded(bioportal_mapping):
                local_results[term] = bioportal_mapping

    return "auto", local_results


def enriched_entity(entity: dict[str, Any], mapping: dict[str, Any], backend: str) -> EnrichedEntity:
    entity_backend = mapping.get("mapping_backend") or backend
    return EnrichedEntity(
        extracted_text=_sanitize_text(entity.get("entity")),
        llm_label=entity.get("label"),
        bioportal_prefLabel=mapping.get("ontology_label"),
        ontology_uri=mapping.get("ontology_id"),
        ontology=mapping.get("ontology"),
        mapping_backend=entity_backend,
        mapping_error=mapping.get("error"),
        chunk=entity.get("chunk"),
        start=entity.get("start"),
        end=entity.get("end"),
        global_start=entity.get("global_start"),
        global_end=entity.get("global_end"),
        source_pass=entity.get("source_pass"),
        context=entity.get("context"),
        source_chunk_path=entity.get("source_chunk_path"),
    )


def write_csv(path: Path, entities: list[EnrichedEntity]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "extracted_text",
        "llm_label",
        "bioportal_prefLabel",
        "ontology_uri",
        "ontology",
        "mapping_backend",
        "mapping_error",
        "chunk",
        "start",
        "end",
        "global_start",
        "global_end",
        "source_pass",
        "context",
        "source_chunk_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entity in entities:
            writer.writerow({field: getattr(entity, field) for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map master extracted neuroscience entities to ontology identifiers.")
    parser.add_argument("--input", required=True, type=Path, help="Path to master_extracted_entities.json.")
    parser.add_argument("--output", type=Path, help="Output JSON path. Defaults to <input parent>/neuro_entities_mapped.json.")
    parser.add_argument("--csv", nargs="?", const="AUTO", default=None, help="Optionally write a CSV. Pass a path or use flag alone for <output>.csv.")
    parser.add_argument("--backend", choices=["auto", "local", "bioportal"], default="auto", help="Mapping backend.")
    parser.add_argument("--max-results", type=int, default=None, help="Maximum ontology results per term. Defaults to MAX_CONCEPT_MAPPING_RESULTS or 1.")
    parser.add_argument(
        "--ontologies",
        default=DEFAULT_BIOPORTAL_ONTOLOGIES,
        help="Comma-separated BioPortal ontology acronyms. Defaults to UBERON,NIFSTD,FMA,GO,SNOMEDCT. Ignored by local backend.",
    )
    return parser.parse_args()


def main() -> int:
    load_environment()
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")
    output_path = args.output.expanduser().resolve() if args.output else input_path.parent / "neuro_entities_mapped.json"
    max_results = _normalize_max_results(args.max_results)
    ontologies = _sanitize_text(args.ontologies or DEFAULT_BIOPORTAL_ONTOLOGIES).replace(" ", "")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    entities = parse_master_entities(payload)
    term_objects = build_term_objects(entities)
    if not term_objects:
        raise ValueError("No valid entities to map")

    backend_used, mappings = map_with_backend(term_objects, args.backend, max_results, ontologies)
    enriched = [enriched_entity(ent, mappings.get(_sanitize_text(ent.get("entity")), {}), backend_used) for ent in entities]
    mapped_count = sum(1 for ent in enriched if ent.ontology_uri)

    result = NeuroEntitiesMappedOutput(
        created_at=datetime.now().isoformat(timespec="seconds"),
        input_path=str(input_path),
        backend=backend_used,
        ontologies=ontologies if backend_used in {"bioportal", "auto"} else None,
        max_results=max_results,
        entity_count=len(enriched),
        mapped_count=mapped_count,
        entities=enriched,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path: Path | None = None
    if args.csv is not None:
        csv_path = output_path.with_suffix(".csv") if args.csv == "AUTO" else Path(args.csv).expanduser().resolve()
        write_csv(csv_path, enriched)

    print(f"Wrote {mapped_count}/{len(enriched)} ontology-mapped entities to {output_path} using backend={backend_used}")
    if csv_path:
        print(f"Wrote CSV view to {csv_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
