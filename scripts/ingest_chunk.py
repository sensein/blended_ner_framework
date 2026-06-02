#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "grobid-client-python>=0.0.9",
#   "pymupdf4llm>=0.0.27",
#   "requests>=2.32.0",
#   "transformers>=4.44.0",
#   "sentencepiece>=0.2.0",
#   "tiktoken>=0.7.0",
# ]
# ///
"""
PDF ingestion and model-aware semantic chunking.

Run with uv inline dependency management:

    uv run scripts/ingest_chunk.py /path/to/paper.pdf \
      --model-id bert-base-uncased \
      --out-dir data/chunks

The input path may be a single PDF or a directory. Directories are searched
recursively for *.pdf files. The parser tries a live local Grobid server first
(default: http://localhost:8070/api/isalive) and falls back to pymupdf4llm on
any ping, timeout, import, or parse failure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from xml.etree import ElementTree as ET

import requests
from transformers import AutoTokenizer, PreTrainedTokenizerBase


TEI_NS = "http://www.tei-c.org/ns/1.0"
CHUNK_SEPARATOR = "---"
VERY_LARGE_MODEL_MAX_LENGTH = 1_000_000


@dataclass
class ParsedDocument:
    text: str
    source: str


@dataclass
class Chunk:
    text: str
    token_count: int
    char_start: int
    char_end: int
    chunk_index: int = 0


class GrobidUnavailable(RuntimeError):
    """Raised when Grobid cannot be reached or cannot parse a PDF."""


def grobid_is_alive(grobid_url: str, timeout: float) -> bool:
    """Return True if Grobid responds positively at /api/isalive."""
    try:
        response = requests.get(
            f"{grobid_url.rstrip('/')}/api/isalive",
            timeout=timeout,
        )
        return response.ok and response.text.strip().lower() in {"true", "1", "alive", ""}
    except requests.RequestException:
        return False


def parse_pdf_with_grobid(pdf_path: Path, grobid_url: str, timeout: float) -> ParsedDocument:
    """
    Parse a PDF through Grobid using the grobid-client-python package when
    available, with a compatible direct HTTP request as a robust fallback.
    """
    if not grobid_is_alive(grobid_url, timeout=min(timeout, 5.0)):
        raise GrobidUnavailable(f"Grobid is not alive at {grobid_url}/api/isalive")

    # Prefer grobid-client-python logic if import/API is available. The package
    # writes TEI XML to disk, so use a temporary output directory and read it.
    try:
        from grobid_client.grobid_client import GrobidClient  # type: ignore

        with tempfile.TemporaryDirectory(prefix="grobid_in_") as in_tmp, tempfile.TemporaryDirectory(prefix="grobid_out_") as out_tmp:
            isolated_pdf = Path(in_tmp) / pdf_path.name
            isolated_pdf.write_bytes(pdf_path.read_bytes())

            client = GrobidClient(config_path=None)
            if hasattr(client, "config"):
                client.config["grobid_server"] = grobid_url.rstrip("/")
                client.config["batch_size"] = 1
                client.config["sleep_time"] = 0
                client.config["timeout"] = timeout
            client.process(
                "processFulltextDocument",
                str(isolated_pdf.parent),
                output=str(out_tmp),
                n=1,
                generateIDs=True,
                consolidate_header=True,
                consolidate_citations=False,
                tei_coordinates=False,
                force=True,
                verbose=False,
            )
            tei_path = Path(out_tmp) / f"{pdf_path.stem}.tei.xml"
            if tei_path.exists():
                return ParsedDocument(text=tei_to_text(tei_path.read_text(encoding="utf-8")), source="grobid")
    except Exception as exc:
        print(f"⚠ grobid-client-python path failed for {pdf_path.name}: {exc}", file=sys.stderr)

    # Direct Grobid HTTP call mirrors grobid-client-python's fulltext endpoint
    # and keeps ingestion working across package API changes.
    try:
        with pdf_path.open("rb") as handle:
            response = requests.post(
                f"{grobid_url.rstrip('/')}/api/processFulltextDocument",
                files={"input": (pdf_path.name, handle, "application/pdf")},
                data={"consolidateHeader": "1", "consolidateCitations": "0"},
                timeout=timeout,
            )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise GrobidUnavailable(f"Grobid parse failed for {pdf_path}: {exc}") from exc

    text = tei_to_text(response.text)
    if not text.strip():
        raise GrobidUnavailable(f"Grobid returned empty text for {pdf_path}")
    return ParsedDocument(text=text, source="grobid")


def tei_to_text(tei_xml: str) -> str:
    """Extract readable paragraph text from Grobid TEI XML."""
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError as exc:
        raise GrobidUnavailable(f"Invalid TEI XML returned by Grobid: {exc}") from exc

    paragraphs: list[str] = []
    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1]
        if tag in {"title", "head", "p", "note"}:
            text = normalize_text("".join(elem.itertext()))
            if text:
                paragraphs.append(text)
    return "\n\n".join(paragraphs).strip()


def parse_pdf_with_pymupdf4llm(pdf_path: Path) -> ParsedDocument:
    """Parse a PDF with pymupdf4llm and return markdown-like text."""
    try:
        import pymupdf4llm  # type: ignore

        text = pymupdf4llm.to_markdown(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"pymupdf4llm failed for {pdf_path}: {exc}") from exc

    text = normalize_markdown(text)
    if not text:
        raise RuntimeError(f"pymupdf4llm returned empty text for {pdf_path}")
    return ParsedDocument(text=text, source="pymupdf4llm")


def parse_pdf(pdf_path: Path, grobid_url: str, grobid_timeout: float) -> ParsedDocument:
    """Try Grobid first, then gracefully fall back to pymupdf4llm."""
    try:
        print(f"→ Trying Grobid for {pdf_path.name}", file=sys.stderr)
        return parse_pdf_with_grobid(pdf_path, grobid_url, grobid_timeout)
    except Exception as exc:
        print(f"⚠ Grobid unavailable/failed for {pdf_path.name}: {exc}", file=sys.stderr)
        print(f"→ Falling back to pymupdf4llm for {pdf_path.name}", file=sys.stderr)
        return parse_pdf_with_pymupdf4llm(pdf_path)


def normalize_text(text: str) -> str:
    """Normalize whitespace inside a paragraph without destroying words."""
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def normalize_markdown(text: str) -> str:
    """Keep paragraph boundaries while removing excessive blank lines/spaces."""
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    """Load the model-specific Hugging Face tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load tokenizer for {model_id!r}. If this is a gated model, "
            "run `huggingface-cli login` or set HF_TOKEN. Original error: "
            f"{exc}"
        ) from exc
    return tokenizer


def tokenizer_limit(tokenizer: PreTrainedTokenizerBase, requested_max_tokens: int | None) -> int:
    """Choose a practical chunk token ceiling from CLI or tokenizer metadata."""
    if requested_max_tokens is not None:
        if requested_max_tokens < 32:
            raise ValueError("--max-tokens must be at least 32")
        return requested_max_tokens

    model_max = int(getattr(tokenizer, "model_max_length", 0) or 0)
    if model_max <= 0 or model_max >= VERY_LARGE_MODEL_MAX_LENGTH:
        # Many HF tokenizers use a huge sentinel value when the true limit is
        # unknown. Use a conservative default suitable for common LLM contexts.
        return 4096
    # Leave room for prompts, instructions, and generated output downstream.
    return max(32, int(model_max * 0.85))


def count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    """Count tokens without adding BOS/EOS or other special tokens."""
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def paragraph_spans(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield non-empty paragraph spans from text."""
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.DOTALL):
        para = match.group(0).strip()
        if para:
            leading = len(match.group(0)) - len(match.group(0).lstrip())
            yield match.start() + leading, match.start() + leading + len(para), para


def sentence_spans(text: str, base_offset: int = 0) -> Iterator[tuple[int, int, str]]:
    """
    Yield sentence-like spans. This avoids splitting common abbreviations as
    much as possible without requiring a heavyweight NLP model.
    """
    boundary = re.compile(r"(?<=[.!?])(?:[\"')\]]+)?\s+(?=[A-Z0-9#*\-])|(?<=\n)\s*(?=\S)")
    start = 0
    for match in boundary.finditer(text):
        end = match.end()
        piece = text[start:end].strip()
        if piece:
            local_start = start + len(text[start:end]) - len(text[start:end].lstrip())
            yield base_offset + local_start, base_offset + local_start + len(piece), piece
        start = end
    tail = text[start:].strip()
    if tail:
        local_start = start + len(text[start:]) - len(text[start:].lstrip())
        yield base_offset + local_start, base_offset + local_start + len(tail), tail


def split_oversized_text(
    text: str,
    base_offset: int,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
) -> Iterator[tuple[int, int, str]]:
    """
    Last-resort splitter for a sentence/paragraph exceeding max_tokens.
    Prefers commas/semicolons/colons, then whitespace. Only splits inside a
    sentence when the sentence alone is too large for the selected model.
    """
    if count_tokens(tokenizer, text) <= max_tokens:
        yield base_offset, base_offset + len(text), text
        return

    pieces = re.split(r"(?<=[,;:])\s+", text)
    if len(pieces) == 1:
        pieces = re.split(r"(\s+)", text)
        pieces = [p for p in pieces if p and not p.isspace()]

    cursor = 0
    buffer: list[str] = []
    buffer_start = 0

    def flush() -> Iterator[tuple[int, int, str]]:
        nonlocal buffer
        if buffer:
            joined = " ".join(buffer).strip()
            if joined:
                yield base_offset + buffer_start, base_offset + buffer_start + len(joined), joined
            buffer = []

    for piece in pieces:
        idx = text.find(piece, cursor)
        if idx < 0:
            idx = cursor
        cursor = idx + len(piece)
        candidate = (" ".join(buffer + [piece])).strip() if buffer else piece.strip()
        if buffer and count_tokens(tokenizer, candidate) > max_tokens:
            yield from flush()
            buffer_start = idx
            buffer = [piece.strip()]
        elif count_tokens(tokenizer, piece.strip()) > max_tokens:
            # Truly pathological token (or no whitespace). Use tokenizer offset
            # mapping to cut by exact token ranges.
            enc = tokenizer(piece, add_special_tokens=False, return_offsets_mapping=True)
            offsets = enc["offset_mapping"]
            for i in range(0, len(offsets), max_tokens):
                span_offsets = offsets[i : i + max_tokens]
                if not span_offsets:
                    continue
                start_char = span_offsets[0][0]
                end_char = span_offsets[-1][1]
                sub = piece[start_char:end_char].strip()
                if sub:
                    yield base_offset + idx + start_char, base_offset + idx + end_char, sub
            buffer = []
        else:
            if not buffer:
                buffer_start = idx
            buffer.append(piece.strip())

    yield from flush()


def semantic_units(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
) -> Iterator[tuple[int, int, str]]:
    """Yield paragraph/sentence units small enough for token packing."""
    for p_start, _p_end, paragraph in paragraph_spans(text):
        if count_tokens(tokenizer, paragraph) <= max_tokens:
            yield p_start, p_start + len(paragraph), paragraph
            continue
        for s_start, _s_end, sentence in sentence_spans(paragraph, base_offset=p_start):
            if count_tokens(tokenizer, sentence) <= max_tokens:
                yield s_start, s_start + len(sentence), sentence
            else:
                yield from split_oversized_text(sentence, s_start, tokenizer, max_tokens)


def chunk_text_semantically(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
) -> list[Chunk]:
    """
    Greedily pack paragraph/sentence units without exceeding max_tokens.
    Boundaries are semantic: paragraphs first, sentences second, and only
    pathological over-limit sentences are split further.

    Chunk text is always copied from the original parsed document span so the
    recorded char_start/char_end offsets remain valid for downstream mapping.
    """
    chunks: list[Chunk] = []
    chunk_start: int | None = None
    chunk_end: int | None = None

    def emit(start: int, end: int) -> None:
        body = text[start:end].strip()
        if not body:
            return
        chunks.append(
            Chunk(
                text=body,
                token_count=count_tokens(tokenizer, body),
                char_start=start,
                char_end=end,
                chunk_index=len(chunks),
            )
        )

    for unit_start, unit_end, _unit_text in semantic_units(text, tokenizer, max_tokens):
        if chunk_start is None:
            chunk_start, chunk_end = unit_start, unit_end
            continue

        assert chunk_end is not None
        candidate = text[chunk_start:unit_end].strip()
        if count_tokens(tokenizer, candidate) > max_tokens:
            emit(chunk_start, chunk_end)
            chunk_start, chunk_end = unit_start, unit_end
        else:
            chunk_end = unit_end

    if chunk_start is not None and chunk_end is not None:
        emit(chunk_start, chunk_end)
    return chunks


def discover_pdfs(input_path: Path) -> list[Path]:
    """Return PDFs from a file or recursively from a directory."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        return [input_path]
    if input_path.is_dir():
        pdfs = sorted(p for p in input_path.rglob("*.pdf") if p.is_file())
        if not pdfs:
            raise ValueError(f"No PDF files found under directory: {input_path}")
        return pdfs
    raise ValueError(f"Input path does not exist: {input_path}")


def output_dir_for(pdf_path: Path, input_root: Path, out_dir: Path) -> Path:
    """Keep directory inputs collision-free while using PDF stem for files."""
    if input_root.is_dir():
        try:
            rel_parent = pdf_path.parent.relative_to(input_root)
        except ValueError:
            rel_parent = Path()
        return out_dir / rel_parent / pdf_path.stem / "chunks"
    return out_dir


def write_chunks(
    chunks: Sequence[Chunk],
    parsed: ParsedDocument,
    pdf_path: Path,
    model_id: str,
    max_tokens: int,
    out_dir: Path,
) -> list[Path]:
    """Write chunk_NNN.txt files and a manifest.json for one PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for chunk in chunks:
        header = {
            "chunk_index": chunk.chunk_index,
            "total_chunks": len(chunks),
            "token_count": chunk.token_count,
            "max_tokens": max_tokens,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "chars": len(chunk.text),
            "doc_chars": len(parsed.text),
            "source": parsed.source,
            "doc_id": pdf_path.stem,
            "model_id": model_id,
        }
        path = out_dir / f"chunk_{chunk.chunk_index:03d}.txt"
        path.write_text(
            json.dumps(header, ensure_ascii=False) + f"\n{CHUNK_SEPARATOR}\n" + chunk.text,
            encoding="utf-8",
        )
        paths.append(path.resolve())

    manifest = {
        "pdf": str(pdf_path.resolve()),
        "doc_id": pdf_path.stem,
        "parser_source": parsed.source,
        "model_id": model_id,
        "max_tokens": max_tokens,
        "doc_chars": len(parsed.text),
        "total_chunks": len(chunks),
        "chunks": [str(p.name) for p in paths],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return paths


def process_pdf(
    pdf_path: Path,
    input_root: Path,
    out_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    model_id: str,
    max_tokens: int,
    grobid_url: str,
    grobid_timeout: float,
) -> list[Path]:
    parsed = parse_pdf(pdf_path, grobid_url=grobid_url, grobid_timeout=grobid_timeout)
    chunks = chunk_text_semantically(parsed.text, tokenizer=tokenizer, max_tokens=max_tokens)
    if not chunks:
        raise RuntimeError(f"No chunks produced for {pdf_path}")
    pdf_out_dir = output_dir_for(pdf_path, input_root, out_dir)
    paths = write_chunks(chunks, parsed, pdf_path, model_id, max_tokens, pdf_out_dir)
    print(f"✓ {pdf_path.name}: wrote {len(paths)} chunks to {pdf_out_dir}", file=sys.stderr)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs with Grobid→pymupdf4llm fallback and model-specific token chunking.",
    )
    parser.add_argument("input", type=Path, help="PDF file or directory containing PDFs.")
    parser.add_argument(
        "--model-id",
        required=True,
        help="Hugging Face model identifier used to load AutoTokenizer (for example, bert-base-uncased or any accessible LLM tokenizer).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("chunks"),
        help="Output directory. For a single PDF, chunks are written directly here; for a directory, per-PDF subdirectories are created.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional per-chunk token ceiling. Defaults to 85%% of tokenizer.model_max_length, or 4096 when unknown.",
    )
    parser.add_argument(
        "--grobid-url",
        default="http://localhost:8070",
        help="Base URL for the local Grobid server.",
    )
    parser.add_argument(
        "--grobid-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for Grobid parse requests.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        pdfs = discover_pdfs(args.input)
        tokenizer = load_tokenizer(args.model_id)
        max_tokens = tokenizer_limit(tokenizer, args.max_tokens)
        print(
            f"→ Loaded tokenizer for {args.model_id}; chunk limit={max_tokens} tokens; PDFs={len(pdfs)}",
            file=sys.stderr,
        )

        all_paths: list[Path] = []
        for pdf in pdfs:
            all_paths.extend(
                process_pdf(
                    pdf_path=pdf,
                    input_root=args.input,
                    out_dir=args.out_dir,
                    tokenizer=tokenizer,
                    model_id=args.model_id,
                    max_tokens=max_tokens,
                    grobid_url=args.grobid_url,
                    grobid_timeout=args.grobid_timeout,
                )
            )
        for path in all_paths:
            print(path)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
