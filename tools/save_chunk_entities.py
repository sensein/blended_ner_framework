"""
save_chunk_entities.py
======================
CLI helper that the NER agent calls after processing each chunk.

Reads a JSON array of entity mentions from stdin and writes it to
``output/<paper_name>/<run_id>/chunk_NNN.json``, wrapped with run metadata.
Creates the output directory if it does not exist.

This tool exists for three reasons:

1. **Filesystem isolation.** Each run lands in its own ``run_id`` directory
   so retries, A/B experiments, and parallel agent runs never overwrite
   each other or contaminate each other's results.

2. **Schema validation.** The agent's free-form JSON is checked against
   the expected shape before being persisted. Malformed output fails loudly
   instead of poisoning the downstream merge step.

3. **Mkdir-on-write.** The output directory is created on first call.
   Agents that hit a missing-directory error tend to "explore" to fix it,
   which is exactly the read-around behaviour we are trying to prevent.

Usage
-----
    echo '[{"entity":"S1","label":"BrainRegion","context":"..."}]' | \\
        uv run tools/save_chunk_entities.py \\
            --paper-name smith_2024 \\
            --run-id 20260528T143215_a3f1 \\
            --chunk-index 0

Output layout
-------------
    output/
    └── <paper_name>/
        └── <run_id>/
            ├── chunk_000.json
            ├── chunk_001.json
            └── ...

Per-chunk file format
---------------------
    {
      "paper_name": "smith_2024",
      "run_id":     "20260528T143215_a3f1",
      "chunk_index": 0,
      "entity_count": 14,
      "entities": [
        {"entity": "S1", "label": "BrainRegion", "context": "..."},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List

# Sensible bound on context strings. Anything past this is almost certainly
# a multi-paragraph dump rather than a sentence, and inflates response size
# unnecessarily. We warn (not reject) so the agent can self-correct.
MAX_CONTEXT_CHARS = 1_000

# Required fields on every entity record. Order is irrelevant; presence is.
REQUIRED_ENTITY_FIELDS = ("entity", "label", "context")

# Root of the output tree, relative to wherever the tool is invoked. The
# orchestrator can override with --output-root if a different layout is
# needed (e.g. for testing).
DEFAULT_OUTPUT_ROOT = Path("output")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_entities(payload: Any) -> List[dict]:
    """
    Verify the stdin payload is a list of well-formed entity dicts.

    Returns the validated list on success. Raises ``ValueError`` with a
    specific, actionable message on any structural problem so the agent
    gets clear feedback rather than a stack trace.
    """
    if not isinstance(payload, list):
        raise ValueError(
            f"Expected a JSON array at the top level, got {type(payload).__name__}. "
            f"Wrap your entities in `[...]`."
        )

    validated: List[dict] = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(
                f"Entity #{i} is {type(item).__name__}, expected an object "
                f"with keys {REQUIRED_ENTITY_FIELDS}."
            )

        missing = [f for f in REQUIRED_ENTITY_FIELDS if f not in item]
        if missing:
            raise ValueError(
                f"Entity #{i} is missing required field(s): {missing}. "
                f"Got keys: {sorted(item.keys())}."
            )

        for field in REQUIRED_ENTITY_FIELDS:
            value = item[field]
            if not isinstance(value, str):
                raise ValueError(
                    f"Entity #{i} field '{field}' is {type(value).__name__}, "
                    f"expected a string."
                )
            if not value.strip():
                raise ValueError(
                    f"Entity #{i} field '{field}' is empty or whitespace-only."
                )

        # Soft-trim absurdly long contexts; warn so the agent learns the bound.
        ctx = item["context"]
        if len(ctx) > MAX_CONTEXT_CHARS:
            print(
                f"WARNING: entity #{i} context is {len(ctx)} chars "
                f"(max suggested: {MAX_CONTEXT_CHARS}). Trimming.",
                file=sys.stderr,
            )
            item = {**item, "context": ctx[:MAX_CONTEXT_CHARS] + "…"}

        # Drop any keys the agent invented beyond the required set.
        # Keeps the output schema stable for the downstream merge step.
        validated.append({f: item[f] for f in REQUIRED_ENTITY_FIELDS})

    return validated


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def _validate_path_segment(name: str, label: str) -> str:
    """
    Reject path segments that contain separators, traversal, or whitespace.

    Prevents an agent (or a buggy orchestrator) from writing outside the
    intended output tree via paper_name="../../etc" or similar tricks.
    """
    if not name or not name.strip():
        raise ValueError(f"--{label} cannot be empty.")
    if any(c in name for c in ("/", "\\", "..", "\n", "\r", "\t")):
        raise ValueError(
            f"--{label} must not contain path separators or whitespace. "
            f"Got: {name!r}"
        )
    return name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="save_chunk_entities",
        description=(
            "Persist NER agent output for one chunk to "
            "output/<paper_name>/<run_id>/chunk_NNN.json. Reads entity "
            "JSON from stdin."
        ),
    )
    p.add_argument(
        "--paper-name",
        required=True,
        help="Stable identifier for the source paper (e.g. PDF stem).",
    )
    p.add_argument(
        "--run-id",
        required=True,
        help=(
            "Unique identifier for this pipeline run. Provided by the "
            "orchestrator — the agent must use it verbatim, never invent one."
        ),
    )
    p.add_argument(
        "--chunk-index",
        type=int,
        required=True,
        help="Zero-based chunk index this output corresponds to.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root output directory (default: ./{DEFAULT_OUTPUT_ROOT}).",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # ---- Validate path-bearing arguments before touching the filesystem ----
    try:
        paper_name = _validate_path_segment(args.paper_name, "paper-name")
        run_id = _validate_path_segment(args.run_id, "run-id")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.chunk_index < 0:
        print(
            f"ERROR: --chunk-index must be >= 0, got {args.chunk_index}",
            file=sys.stderr,
        )
        return 2

    # ---- Parse stdin ----
    stdin_text = sys.stdin.read()
    if not stdin_text.strip():
        print(
            "ERROR: no JSON received on stdin. "
            "Pipe the entity array into this tool.",
            file=sys.stderr,
        )
        return 2

    try:
        payload = json.loads(stdin_text)
    except json.JSONDecodeError as exc:
        print(
            f"ERROR: stdin is not valid JSON: {exc}.\n"
            f"Re-emit the entity array; do not read other files to recover.",
            file=sys.stderr,
        )
        return 2

    # ---- Validate entity schema ----
    try:
        entities = _validate_entities(payload)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # ---- Build the output document ----
    out_dir = args.output_root / paper_name / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"chunk_{args.chunk_index:03d}.json"

    document = {
        "paper_name": paper_name,
        "run_id": run_id,
        "chunk_index": args.chunk_index,
        "entity_count": len(entities),
        "entities": entities,
    }

    # ---- Atomic write: stage to .tmp, then rename ----
    # Prevents a partial file being left behind if the process is interrupted
    # mid-write — important because the merge script globs this directory.
    tmp_path = out_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(document, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    tmp_path.replace(out_path)

    print(
        f"✓ chunk_{args.chunk_index:03d}: {len(entities)} entities → {out_path}",
        file=sys.stderr,
    )
    # Also print the path on stdout so the agent (or a shell pipeline) can
    # capture it cleanly.
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())