"""
ner_chunker.py
==============
End-to-end PDF → parsed text → chunked-text pipeline for an NER agent running
on pi.dev (or any environment with a strict per-call payload limit).

Pipeline
--------
    paper.pdf
        │
        ├─► Grobid (/api/processFulltextDocument) → TEI XML
        │      └─► structural chunker (paragraph packer)
        │
        └─► PyMuPDF4LLM (markdown/text extraction)
               └─► sliding-window chunker

    chunk_000.txt, chunk_001.txt, ...

Each chunk file embeds its own metadata header (JSON) followed by a `---`
separator and the chunk body. The downstream NER agent reads one chunk at a
time, runs the model on the body, and maps local span offsets to global
document offsets using the header's `global_offset` field — no separate
manifest required.

Chunk file format
-----------------
    {"chunk_index": 0, "global_offset": 0, "total_chunks": 4, "chars": 620,
     "doc_chars": 2457, "source": "structural", "doc_id": "paper"}
    ---
    <plain-text body, exactly `chars` characters long>

CLI usage
---------
    # Default run (auto): try Grobid first, then fall back to PyMuPDF4LLM
    python ner_chunker.py paper.pdf --out-dir ./chunks/

    # Force Grobid
    python ner_chunker.py paper.pdf \\
        --parser grobid \\
        --grobid-url http://grobid.local:8070 \\
        --max-chars 40000 \\
        --out-dir ./out/

    # Force PyMuPDF4LLM
    python ner_chunker.py paper.pdf --parser pymupdf4llm --out-dir ./out/

Library usage
-------------
    from ner_chunker import chunk_tei_xml, OffsetTracker, read_chunk_file

    header, body = read_chunk_file("chunks/chunk_000.txt")
    local_spans  = run_ner(body)                      # your model call
    global_spans = [
        {"start": s["start"] + header["global_offset"],
         "end":   s["end"]   + header["global_offset"],
         "label": s["label"]}
        for s in local_spans
    ]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple
from xml.etree import ElementTree as ET


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    """
    A single payload-safe slice of the document.

    Attributes
    ----------
    text:
        The plain-text content of this chunk.
    global_offset:
        Character index in the *reconstructed plain-text document* at which
        this chunk starts. Used to project local NER spans to global spans.
    chunk_index:
        Zero-based position in the ordered list of all chunks.
    source:
        ``"structural"`` when produced by TeiXmlChunker, ``"sliding_window"``
        when produced by SlidingWindowChunker.
    """

    text: str
    global_offset: int
    chunk_index: int
    source: str = "structural"

    @property
    def byte_size(self) -> int:
        """UTF-8 byte length of the chunk text."""
        return len(self.text.encode("utf-8"))


@dataclass
class NerSpan:
    """
    A single named-entity prediction in *global* document coordinates.
    """

    start: int
    end: int
    label: str
    text: str = ""
    chunk_index: int = -1

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "NerSpan") -> bool:
        """Return True when two spans share at least one character position."""
        return self.start < other.end and other.start < self.end


# ---------------------------------------------------------------------------
# Grobid client
# ---------------------------------------------------------------------------

class GrobidClient:
    """
    Thin wrapper around the Grobid HTTP API for full-text PDF parsing.

    Grobid is expected to be running as a service (Docker image
    ``lfoppiano/grobid`` on port 8070 by default). This client POSTs a PDF
    to the ``processFulltextDocument`` endpoint and returns the resulting
    TEI XML string.

    Parameters
    ----------
    base_url:
        Root URL of the Grobid service, e.g. ``http://localhost:8070``.
    timeout:
        HTTP timeout in seconds. Large papers (>30 pages) can take 60+ seconds
        to parse so the default is set generously.
    consolidate_citations:
        Grobid flag — set to 0 (off), 1 (consolidate via CrossRef), or 2
        (consolidate against local biblio-glutton service).
    """

    FULLTEXT_ENDPOINT = "/api/processFulltextDocument"
    ISALIVE_ENDPOINT = "/api/isalive"

    def __init__(
        self,
        base_url: str = "http://localhost:8070",
        timeout: float = 300.0,
        consolidate_citations: int = 0,
    ) -> None:
        try:
            import requests as requests_mod
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The `requests` package is required for GrobidClient. "
                "Install with: uv add requests"
            ) from exc

        self._requests: Any = requests_mod
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.consolidate_citations = consolidate_citations

    def is_alive(self) -> bool:
        """Return True if the Grobid server responds at /api/isalive."""
        req = self._requests
        try:
            r = req.get(self.base_url + self.ISALIVE_ENDPOINT, timeout=5)
            return r.status_code == 200
        except req.RequestException:
            return False

    def process_fulltext(self, pdf_path: Path) -> str:
        """
        POST *pdf_path* to Grobid's full-text endpoint and return TEI XML.

        Parameters
        ----------
        pdf_path:
            Path to the source PDF on disk.

        Returns
        -------
        str
            UTF-8 TEI XML string as returned by Grobid.

        Raises
        ------
        FileNotFoundError
            If *pdf_path* does not exist.
        RuntimeError
            If Grobid returns a non-2xx status, or the request fails.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        url = self.base_url + self.FULLTEXT_ENDPOINT
        req = self._requests
        try:
            with pdf_path.open("rb") as fh:
                resp = req.post(
                    url,
                    files={"input": (pdf_path.name, fh, "application/pdf")},
                    data={"consolidateCitations": str(self.consolidate_citations)},
                    timeout=self.timeout,
                )
        except req.RequestException as exc:
            raise RuntimeError(
                f"Grobid request to {url} failed: {exc}. "
                f"Is the Grobid server running and reachable?"
            ) from exc

        if not resp.ok:
            raise RuntimeError(
                f"Grobid returned HTTP {resp.status_code} for {pdf_path.name}: "
                f"{resp.text[:200]}"
            )
        return resp.text


# ---------------------------------------------------------------------------
# 1. Structural Chunker — paragraph-level greedy packing
# ---------------------------------------------------------------------------

_TEI_NS = "http://www.tei-c.org/ns/1.0"
_P_TAG = f"{{{_TEI_NS}}}p"
_P_TAG_BARE = "p"


def _iter_paragraph_texts(xml_source: str) -> Iterator[str]:
    """
    Yield whitespace-normalised plain-text content of every <p> element in
    the TEI tree. Inline markup (<ref>, <hi>, etc.) is collapsed via itertext().
    """
    try:
        root = ET.fromstring(xml_source)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse TEI XML: {exc}") from exc

    for elem in root.iter():
        if elem.tag in (_P_TAG, _P_TAG_BARE):
            raw = "".join(elem.itertext())
            normalised = re.sub(r"[ \t]+", " ", raw).strip()
            if normalised:
                yield normalised


class TeiXmlChunker:
    """
    Greedy paragraph-packing chunker for Grobid TEI XML.

    See module docstring for the full algorithm description. Single
    paragraphs that exceed ``max_chars`` are routed through
    :class:`SlidingWindowChunker` automatically.
    """

    def __init__(
        self,
        max_chars: int = 45_000,
        paragraph_sep: str = "\n\n",
    ) -> None:
        self.max_chars = max_chars
        self.paragraph_sep = paragraph_sep

    def chunk(self, xml_source: str) -> Tuple[List[TextChunk], str]:
        """
        Parse *xml_source* and return ``(chunks, plain_text)``.

        *plain_text* is the full reconstructed document obtained by joining
        every chunk with ``paragraph_sep``; it is the ground truth against
        which all global offsets resolve.
        """
        paragraphs = list(_iter_paragraph_texts(xml_source))
        return self._pack(paragraphs)

    def _pack(self, paragraphs: List[str]) -> Tuple[List[TextChunk], str]:
        chunks: List[TextChunk] = []
        buffer: List[str] = []
        buffer_chars = 0
        global_offset = 0
        sep = self.paragraph_sep
        sep_len = len(sep)

        def _flush() -> None:
            nonlocal global_offset, buffer, buffer_chars
            if not buffer:
                return
            text = sep.join(buffer)
            chunks.append(
                TextChunk(
                    text=text,
                    global_offset=global_offset,
                    chunk_index=len(chunks),
                    source="structural",
                )
            )
            global_offset += len(text) + sep_len
            buffer = []
            buffer_chars = 0

        for para in paragraphs:
            addition = (sep_len if buffer else 0) + len(para)

            if buffer and buffer_chars + addition > self.max_chars:
                _flush()

            if len(para) > self.max_chars:
                # Oversized paragraph → hand off to sliding window.
                _flush()
                fallback_chunks, _ = SlidingWindowChunker(
                    max_chars=self.max_chars
                ).chunk_text(para, global_offset=global_offset)
                for fc in fallback_chunks:
                    fc.chunk_index = len(chunks)
                    chunks.append(fc)
                if fallback_chunks:
                    last = fallback_chunks[-1]
                    global_offset = last.global_offset + len(last.text) + sep_len
                continue

            buffer.append(para)
            buffer_chars += addition

        _flush()
        plain_text = sep.join(c.text for c in chunks)
        return chunks, plain_text


# ---------------------------------------------------------------------------
# 2. Sliding Window Chunker — fallback for plain text or giant paragraphs
# ---------------------------------------------------------------------------

class SlidingWindowChunker:
    """
    Overlapping sliding-window chunker for plain text.

    Used when the input is bare text (no TEI markup) or when a single
    paragraph exceeds the per-chunk ceiling. Cuts are snapped backwards to
    the nearest sentence boundary within the overlap zone so entities are
    never split mid-token.
    """

    _SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=\n)\s*")

    def __init__(
        self,
        max_chars: int = 40_000,
        stride: int = 35_000,
        snap_window: Optional[int] = None,
    ) -> None:
        if stride >= max_chars:
            raise ValueError("stride must be less than max_chars")
        self.max_chars = max_chars
        self.stride = stride
        self.snap_window = (
            snap_window if snap_window is not None else (max_chars - stride)
        )

    def chunk_text(
        self, text: str, global_offset: int = 0
    ) -> Tuple[List[TextChunk], str]:
        """
        Slice *text* into overlapping chunks.

        *global_offset* is added to every chunk's offset so that a single
        oversized paragraph hosted inside a larger structural chunk pipeline
        retains consistent global coordinates.
        """
        chunks: List[TextChunk] = []
        doc_len = len(text)
        window_start = 0

        while window_start < doc_len:
            raw_end = min(window_start + self.max_chars, doc_len)
            snapped_end = self._snap_to_boundary(text, raw_end, window_start)
            chunk_text = text[window_start:snapped_end]
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    global_offset=global_offset + window_start,
                    chunk_index=len(chunks),
                    source="sliding_window",
                )
            )
            if snapped_end >= doc_len:
                break
            next_start = window_start + self.stride
            next_start = self._snap_to_boundary(text, next_start, window_start)
            window_start = max(next_start, window_start + 1)

        return chunks, text

    def _snap_to_boundary(
        self, text: str, cut: int, window_start: int
    ) -> int:
        """Snap *cut* back to the nearest sentence boundary, if one exists."""
        if cut >= len(text):
            return len(text)
        search_start = max(cut - self.snap_window, window_start)
        segment = text[search_start:cut]
        matches = list(self._SENTENCE_END.finditer(segment))
        if matches:
            return search_start + matches[-1].end()
        return cut


# ---------------------------------------------------------------------------
# 3. Global Offset Tracker & Deduplication
# ---------------------------------------------------------------------------

class OffsetTracker:
    """
    Maintains global state for a chunked document and projects chunk-local
    NER spans to global coordinates.
    """

    def __init__(self, plain_text: str) -> None:
        self.plain_text = plain_text
        self._chunks: List[TextChunk] = []

    def register_chunks(self, chunks: List[TextChunk]) -> None:
        """Store the ordered chunk list for offset resolution."""
        self._chunks = list(chunks)

    def map_to_global(
        self, local_spans: List[dict], chunk_index: int
    ) -> List[NerSpan]:
        """
        Convert local chunk spans to global :class:`NerSpan` objects.

        Each input dict must have ``start``, ``end``, ``label`` keys.
        Raises ``ValueError`` for spans outside the chunk's extent.
        """
        if not self._chunks:
            raise RuntimeError("No chunks registered. Call register_chunks() first.")
        chunk = self._chunks[chunk_index]
        base = chunk.global_offset
        chunk_len = len(chunk.text)

        out: List[NerSpan] = []
        for sp in local_spans:
            lo, hi, label = sp["start"], sp["end"], sp["label"]
            if not (0 <= lo < hi <= chunk_len):
                raise ValueError(
                    f"Span ({lo}, {hi}) outside chunk {chunk_index} (len {chunk_len})"
                )
            g_start, g_end = base + lo, base + hi
            out.append(
                NerSpan(
                    start=g_start,
                    end=g_end,
                    label=label,
                    text=self.plain_text[g_start:g_end],
                    chunk_index=chunk_index,
                )
            )
        return out

    @staticmethod
    def deduplicate(spans: List[NerSpan]) -> List[NerSpan]:
        """
        Remove overlapping entity spans, keeping the longest at each location.

        Sliding-window chunks produce duplicates in the overlap region; this
        collapses them. For equal-length overlaps the earlier chunk wins.
        """
        if not spans:
            return []
        sorted_spans = sorted(spans, key=lambda s: (s.start, -s.length))
        accepted: List[NerSpan] = []
        for cand in sorted_spans:
            overlapping = [kept for kept in accepted if cand.overlaps(kept)]
            if any(kept.length >= cand.length for kept in overlapping):
                continue
            accepted = [kept for kept in accepted if not cand.overlaps(kept)]
            accepted.append(cand)
        return sorted(accepted, key=lambda s: s.start)

    def resolve_surface(self, span: NerSpan) -> str:
        """Return the surface text for a globally-mapped span."""
        return self.plain_text[span.start:span.end]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def chunk_tei_xml(
    xml_source: str, max_chars: int = 45_000
) -> Tuple[List[TextChunk], OffsetTracker]:
    """One-shot: TEI XML → chunks + pre-registered :class:`OffsetTracker`."""
    chunker = TeiXmlChunker(max_chars=max_chars)
    chunks, plain_text = chunker.chunk(xml_source)
    tracker = OffsetTracker(plain_text)
    tracker.register_chunks(chunks)
    return chunks, tracker


# ---------------------------------------------------------------------------
# Chunk file I/O — self-describing format (JSON header + `---` + body)
# ---------------------------------------------------------------------------

CHUNK_SEPARATOR = "---"


def write_chunk_file(
    chunk: TextChunk,
    total_chunks: int,
    doc_chars: int,
    doc_id: str,
    out_dir: Path,
) -> Path:
    """
    Serialise a chunk to ``<out_dir>/chunk_NNN.txt`` with a JSON header.

    Layout::

        {"chunk_index": 0, "global_offset": 0, "total_chunks": 4,
         "chars": 620, "doc_chars": 2457, "source": "structural",
         "doc_id": "paper"}
        ---
        <body text>

    The header is one line of compact JSON. The body starts on line 3 and
    contains exactly ``chunk.text`` — no trailing newline is added, so
    ``len(body)`` equals the ``chars`` field.

    Parameters
    ----------
    chunk:
        The :class:`TextChunk` to serialise.
    total_chunks:
        Total number of chunks for this document (embedded in every file so
        the agent never needs a separate manifest).
    doc_chars:
        Length of the reconstructed plain-text document.
    doc_id:
        Stable identifier for the source document (typically the PDF stem).
    out_dir:
        Output directory; created if it does not exist.

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "chunk_index": chunk.chunk_index,
        "global_offset": chunk.global_offset,
        "total_chunks": total_chunks,
        "chars": len(chunk.text),
        "doc_chars": doc_chars,
        "source": chunk.source,
        "doc_id": doc_id,
    }
    path = out_dir / f"chunk_{chunk.chunk_index:03d}.txt"
    # Write header as compact JSON on a single line, then separator, then body.
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False))
        fh.write("\n")
        fh.write(CHUNK_SEPARATOR)
        fh.write("\n")
        fh.write(chunk.text)
    return path.resolve()


def read_chunk_file(path: Path) -> Tuple[dict, str]:
    """
    Parse a chunk file written by :func:`write_chunk_file`.

    Returns
    -------
    (header, body)
        ``header`` is the parsed JSON dict; ``body`` is the chunk text
        exactly as written, with no surrounding whitespace stripped.

    Raises
    ------
    ValueError
        If the file does not match the expected header/separator/body layout.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        header_line = fh.readline()
        sep_line = fh.readline()
        body = fh.read()
    if not header_line or sep_line.rstrip("\n") != CHUNK_SEPARATOR:
        raise ValueError(
            f"{path} does not look like a chunk file "
            f"(expected JSON header line followed by `{CHUNK_SEPARATOR}`)"
        )
    try:
        header = json.loads(header_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON header in {path}: {exc}") from exc
    return header, body


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_PARSER_CHOICES = ("auto", "grobid", "pymupdf4llm")


def _chunk_pdf_with_grobid(
    pdf_path: Path,
    grobid_url: str,
    max_chars: int,
) -> Tuple[List[TextChunk], OffsetTracker]:
    """Parse a PDF with Grobid and return chunks + offset tracker."""
    client = GrobidClient(base_url=grobid_url)
    if not client.is_alive():
        raise RuntimeError(
            f"Grobid service at {grobid_url} is not responding. "
            f"Start it (e.g. `docker run -p 8070:8070 lfoppiano/grobid:0.8.1`) "
            f"or use `--parser pymupdf4llm`."
        )

    print(f"→ Sending {pdf_path.name} to Grobid at {grobid_url} ...", file=sys.stderr)
    tei_xml = client.process_fulltext(pdf_path)
    print(f"  Received {len(tei_xml):,} chars of TEI XML.", file=sys.stderr)
    return chunk_tei_xml(tei_xml, max_chars=max_chars)


def _chunk_pdf_with_pymupdf4llm(
    pdf_path: Path,
    max_chars: int,
) -> Tuple[List[TextChunk], OffsetTracker]:
    """Parse a PDF with PyMuPDF4LLM and return chunks + offset tracker."""
    try:
        import pymupdf4llm  # pyright: ignore[reportMissingImports]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyMuPDF4LLM parser requested but the package failed to import. "
            "Install with: `uv add pymupdf4llm` and activate the project virtualenv."
        ) from exc

    print(f"→ Parsing {pdf_path.name} with PyMuPDF4LLM ...", file=sys.stderr)
    try:
        extracted = pymupdf4llm.to_markdown(str(pdf_path))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"PyMuPDF4LLM failed to parse {pdf_path.name}: {exc}") from exc

    plain_text = re.sub(r"\n{3,}", "\n\n", extracted).strip()
    if not plain_text:
        raise RuntimeError(f"PyMuPDF4LLM returned empty text for {pdf_path.name}")

    stride = max(1, int(max_chars * 0.85))
    if stride >= max_chars:
        stride = max_chars - 1
    chunks, reconstructed = SlidingWindowChunker(
        max_chars=max_chars,
        stride=stride,
    ).chunk_text(plain_text)
    for chunk in chunks:
        chunk.source = "pymupdf4llm"

    tracker = OffsetTracker(reconstructed)
    tracker.register_chunks(chunks)
    return chunks, tracker


def _process_pdf(
    pdf_path: Path,
    out_dir: Path,
    grobid_url: str,
    max_chars: int,
    parser_name: str,
) -> List[Path]:
    """Full pipeline: PDF → parsed text → chunks on disk."""
    if parser_name not in _PARSER_CHOICES:
        raise ValueError(f"Unsupported parser: {parser_name!r}")

    if parser_name == "grobid":
        chunks, tracker = _chunk_pdf_with_grobid(
            pdf_path=pdf_path,
            grobid_url=grobid_url,
            max_chars=max_chars,
        )
    elif parser_name == "pymupdf4llm":
        chunks, tracker = _chunk_pdf_with_pymupdf4llm(
            pdf_path=pdf_path,
            max_chars=max_chars,
        )
    else:
        try:
            chunks, tracker = _chunk_pdf_with_grobid(
                pdf_path=pdf_path,
                grobid_url=grobid_url,
                max_chars=max_chars,
            )
        except RuntimeError as grobid_exc:
            print(
                f"⚠ Grobid parse failed: {grobid_exc}\n"
                f"  Falling back to PyMuPDF4LLM.",
                file=sys.stderr,
            )
            chunks, tracker = _chunk_pdf_with_pymupdf4llm(
                pdf_path=pdf_path,
                max_chars=max_chars,
            )

    doc_chars = len(tracker.plain_text)
    doc_id = pdf_path.stem

    paths: List[Path] = []
    for chunk in chunks:
        p = write_chunk_file(
            chunk=chunk,
            total_chunks=len(chunks),
            doc_chars=doc_chars,
            doc_id=doc_id,
            out_dir=out_dir,
        )
        paths.append(p)
    return paths


def cli_main(argv: Optional[List[str]] = None) -> int:
    """Argparse entry point. Returns process exit code."""
    parser = argparse.ArgumentParser(
        prog="ner_chunker",
        description=(
            "Parse a PDF with Grobid or PyMuPDF4LLM and emit payload-safe text "
            "chunks with embedded global-offset metadata, ready for an NER agent."
        ),
    )
    parser.add_argument(
        "pdf",
        type=Path,
        help="Path to the input PDF.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write chunk files into. "
            "Defaults to <pdf_stem>.chunks/ next to the input."
        ),
    )
    parser.add_argument(
        "--parser",
        choices=_PARSER_CHOICES,
        default="auto",
        help=(
            "PDF parser to use: auto (try Grobid, then fallback), grobid, "
            "or pymupdf4llm."
        ),
    )
    parser.add_argument(
        "--grobid-url",
        default="http://localhost:8070",
        help="Base URL of the Grobid service (default: http://localhost:8070).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=45_000,
        help="Maximum characters per chunk (default: 45000, ~45kb for ASCII).",
    )

    args = parser.parse_args(argv)

    if not args.pdf.is_file():
        parser.error(f"PDF not found: {args.pdf}")
    if args.max_chars < 200:
        parser.error("--max-chars must be >= 200")

    out_dir = args.out_dir or args.pdf.with_suffix("").parent / f"{args.pdf.stem}.chunks"

    try:
        paths = _process_pdf(
            pdf_path=args.pdf,
            out_dir=out_dir,
            grobid_url=args.grobid_url,
            max_chars=args.max_chars,
            parser_name=args.parser,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"✓ Wrote {len(paths)} chunks to {out_dir}", file=sys.stderr)
    for p in paths:
        print(p)  # stdout: one path per line for easy piping
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())