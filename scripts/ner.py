#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gliner>=0.2.16",
# ]
# ///
"""Local GLiNER runner for chunked NER inputs.

Expected by scripts/hybrid_ner_orchestrator.py. It accepts a file or folder and a
comma-separated label list, runs GLiNER locally, and writes JSON outputs.

Example:

    uv run scripts/ner.py \
      --input data/papers/multiscale_spatial_transcriptomic/20260602T143152/chunks \
      --labels BRAIN_REGION,CELL_TYPE,GENE,TECHNIQUE
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_GLINER_MODEL = "urchade/gliner_medium-v2.1"
TEXT_SUFFIXES = {".txt", ".md", ".xml", ".tei", ".json", ".jsonl", ".csv", ".tsv"}

logger = logging.getLogger(__name__)


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


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local GLiNER NER over a text file or chunk folder.")
    parser.add_argument("--input", required=True, help="Input text file, chunk_NNN.txt file, or directory of chunks/text files.")
    parser.add_argument(
        "--labels",
        required=True,
        help="Comma-separated labels or a JSON array of labels, e.g. BRAIN_REGION,CELL_TYPE or '[\"BRAIN_REGION\"]'.",
    )
    parser.add_argument("--model", default=DEFAULT_GLINER_MODEL, help="GLiNER Hugging Face model ID.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum GLiNER confidence threshold.")
    parser.add_argument("--output-dir", help="Directory for JSON results. Defaults to output/gliner/<timestamp>.")
    parser.add_argument("--recursive", action="store_true", help="Recursively process text files under --input directories.")
    parser.add_argument("--max-chars", type=int, default=0, help="Optional max chars per file; 0 means no truncation.")
    parser.add_argument(
        "--device",
        default=os.getenv("GLINER_DEVICE", "auto"),
        help="PyTorch device for GLiNER inference: auto, cuda:0, mps, cpu, etc. Defaults to GLINER_DEVICE or auto.",
    )
    parser.add_argument(
        "--fp16",
        choices=["auto", "on", "off"],
        default=os.getenv("GLINER_FP16", "auto"),
        help="Cast GLiNER to FP16 on CUDA/ROCm devices. auto enables FP16 only when device.type == 'cuda'.",
    )
    parser.add_argument(
        "--mps-empty-cache-every",
        type=int,
        default=env_int("GLINER_MPS_EMPTY_CACHE_EVERY", 50),
        help="On Apple Silicon MPS, call torch.mps.empty_cache() every N files to reduce allocator fragmentation. 0 disables.",
    )
    return parser.parse_args()


def get_optimal_device(requested: str = "auto"):
    """Dynamically select the optimal PyTorch accelerator for the current environment."""
    import torch

    requested = (requested or "auto").strip()
    if requested.lower() != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {requested!r}, but CUDA/ROCm is not available")
        if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError(f"Requested device {requested!r}, but Apple Silicon MPS is not available")
        logger.info("Using explicitly requested PyTorch device: %s", device)
        return device

    # NVIDIA CUDA or AMD ROCm. PyTorch built with ROCm also reports this via
    # torch.cuda.is_available(), so cuda:0 is the portable accelerator string.
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "unknown GPU"
        logger.info("Hardware accelerator found: CUDA/ROCm (%s)", device_name)
        return torch.device("cuda:0")

    # Apple Silicon Metal Performance Shaders.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Hardware accelerator found: Apple Silicon (MPS)")
        return torch.device("mps")

    logger.warning("No hardware accelerator found. Falling back to CPU. Inference will be severely bottlenecked.")
    return torch.device("cpu")


def should_use_fp16(setting: str, device: Any) -> bool:
    """Enable FP16 only where it is expected to be stable and beneficial."""
    setting = (setting or "auto").lower()
    if setting == "off":
        return False
    if setting == "auto":
        return getattr(device, "type", None) == "cuda"
    if getattr(device, "type", None) != "cuda":
        logger.warning("--fp16=on was requested, but device is %s; skipping FP16 to avoid MPS/CPU instability", device)
        return False
    return True


def maybe_empty_mps_cache(device: Any, every: int, completed_files: int) -> None:
    """Periodically clear the MPS allocator cache to reduce long-loop fragmentation."""
    if getattr(device, "type", None) != "mps" or every <= 0 or completed_files <= 0 or completed_files % every != 0:
        return
    try:
        import torch

        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
            logger.info("Emptied Apple MPS cache after %d file(s)", completed_files)
    except Exception as exc:
        logger.debug("Could not empty MPS cache: %s", exc)


def parse_labels(raw: str) -> list[str]:
    raw = raw.strip()
    if not raw:
        raise ValueError("--labels cannot be empty")

    if raw.startswith("["):
        labels = json.loads(raw)
        if not isinstance(labels, list):
            raise ValueError("JSON --labels must be an array")
    else:
        labels = raw.split(",")

    seen: set[str] = set()
    normalized: list[str] = []
    for label in labels:
        cleaned = str(label).strip()
        if not cleaned:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)

    if not normalized:
        raise ValueError("No valid labels parsed from --labels")
    return normalized


def discover_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    chunk_files = sorted(input_path.rglob("chunk_*.txt") if recursive else input_path.glob("chunk_*.txt"))
    if chunk_files:
        return chunk_files

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    files = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No chunk/text files found under {input_path}")
    return files


def read_text_body(path: Path, max_chars: int = 0) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Chunk files are self-describing: JSON header, separator, body.
    if "\n---\n" in text and path.name.startswith("chunk_"):
        text = text.split("\n---\n", 1)[1]
    text = text.strip()
    return text[:max_chars] if max_chars and max_chars > 0 else text


def safe_output_name(path: Path) -> str:
    stem = path.stem or "document"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem) + ".json"


def context_for(text: str, start: int, end: int, window: int = 100) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def normalize_entity(entity: dict[str, Any], text: str, source_file: Path) -> dict[str, Any]:
    start = int(entity.get("start", entity.get("start_pos", -1)))
    end = int(entity.get("end", entity.get("end_pos", -1)))
    surface = entity.get("text") or entity.get("entity") or (text[start:end] if 0 <= start < end <= len(text) else "")
    label = entity.get("label") or entity.get("entity_type")
    score = entity.get("score")

    return {
        "entity": surface,
        "label": label,
        "score": score,
        "start": start,
        "end": end,
        "context": context_for(text, start, end) if 0 <= start < end <= len(text) else "",
        "source_file": str(source_file),
    }


def write_manifest(output_dir: Path, args: argparse.Namespace, labels: list[str], files: Iterable[Path], device: Any, fp16_enabled: bool) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(Path(args.input).resolve()),
        "model": args.model,
        "threshold": args.threshold,
        "requested_device": args.device,
        "device": str(device),
        "fp16_enabled": fp16_enabled,
        "mps_empty_cache_every": args.mps_empty_cache_every,
        "labels": labels,
        "files": [str(p) for p in files],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s - %(levelname)s - %(message)s")
    load_dotenv()
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    labels = parse_labels(args.labels)
    files = discover_files(input_path, args.recursive)

    output_dir = Path(args.output_dir) if args.output_dir else Path("output") / "gliner" / datetime.now().strftime("%Y%m%dT%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_optimal_device(args.device)
    fp16_enabled = should_use_fp16(args.fp16, device)

    print(f"Loading GLiNER model: {args.model}")
    print(f"PyTorch device: {device}")
    print(f"FP16 enabled: {fp16_enabled}")
    from gliner import GLiNER  # Imported lazily so --help and argument errors stay lightweight.

    model = GLiNER.from_pretrained(args.model).to(device)
    if fp16_enabled:
        model = model.half()
        logger.info("Cast GLiNER model to FP16 for CUDA/ROCm inference")

    print(f"Input: {input_path}")
    print(f"Files: {len(files)}")
    print(f"Labels: {', '.join(labels)}")
    print(f"Threshold: {args.threshold}")
    print(f"Output directory: {output_dir}")

    total_entities = 0
    for file_number, file_path in enumerate(files, start=1):
        text = read_text_body(file_path, args.max_chars)
        if not text:
            entities: list[dict[str, Any]] = []
        else:
            raw_entities = model.predict_entities(text, labels, threshold=args.threshold)
            entities = [normalize_entity(ent, text, file_path) for ent in raw_entities]

        total_entities += len(entities)
        out_path = output_dir / safe_output_name(file_path)
        out_path.write_text(json.dumps(entities, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{file_path} -> {out_path} ({len(entities)} entities)")
        maybe_empty_mps_cache(device, args.mps_empty_cache_every, file_number)

    write_manifest(output_dir, args, labels, files, device, fp16_enabled)
    print(f"Done. Wrote {total_entities} entities across {len(files)} file(s).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
