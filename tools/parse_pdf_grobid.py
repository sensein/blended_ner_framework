import argparse
import json
import sys
import xml.etree.ElementTree as ET

import requests


def _normalized_text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    text = " ".join("".join(elem.itertext()).split())
    return text.strip()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _find_first(root: ET.Element, name: str, *, within: ET.Element | None = None) -> ET.Element | None:
    container = within if within is not None else root
    for elem in container.iter():
        if _local_name(elem.tag) == name:
            return elem
    return None


def _find_all(root: ET.Element, name: str, *, within: ET.Element | None = None) -> list[ET.Element]:
    container = within if within is not None else root
    return [elem for elem in container.iter() if _local_name(elem.tag) == name]


def process_pdf(pdf_path: str):
    url = "http://localhost:8070/api/processFulltextDocument"

    try:
        with open(pdf_path, "rb") as file:
            files = {"input": (pdf_path, file, "application/pdf")}
            response = requests.post(url, files=files)

        response.raise_for_status()

        # Parse TEI XML without external XML parser dependencies (e.g., lxml).
        root = ET.fromstring(response.content)

        # 1) Title
        title_stmt = _find_first(root, "titleStmt")
        title_tag = _find_first(root, "title", within=title_stmt) if title_stmt is not None else _find_first(root, "title")
        title = _normalized_text(title_tag) or "Unknown Title"

        # 2) Authors
        authors = []
        for author in _find_all(root, "author"):
            pers_name = _find_first(root, "persName", within=author)
            if pers_name is None:
                continue

            forenames = [_normalized_text(f) for f in _find_all(root, "forename", within=pers_name)]
            surname = _normalized_text(_find_first(root, "surname", within=pers_name))
            full_name = " ".join([*filter(None, forenames), surname]).strip()
            if full_name:
                authors.append(full_name)

        # 3) Keywords
        keywords = []
        for term in _find_all(root, "term"):
            keyword = _normalized_text(term)
            if keyword:
                keywords.append(keyword)

        # 4) Abstract
        abstract_tag = _find_first(root, "abstract")
        abstract = _normalized_text(abstract_tag) or "No abstract found."

        # 5) Body
        body_tag = _find_first(root, "body")
        if body_tag is not None:
            paragraphs = []
            for p in _find_all(root, "p", within=body_tag):
                p_text = _normalized_text(p)
                if p_text:
                    paragraphs.append(p_text)
            body = "\n\n".join(paragraphs) if paragraphs else _normalized_text(body_tag)
        else:
            body = "No body text found."

        output = {
            "document_metadata": {
                "title": title,
                "authors": authors,
                "keywords": keywords,
            },
            "content": {
                "abstract": abstract,
                "body": body,
            },
        }

        print(json.dumps(output, indent=2))

    except requests.exceptions.ConnectionError:
        print(
            "ERROR: Could not connect to Grobid. Ensure the Docker container is running on port 8070.",
            file=sys.stderr,
        )
        sys.exit(1)
    except FileNotFoundError:
        print(f"ERROR: Could not find PDF file at {pdf_path}", file=sys.stderr)
        sys.exit(1)
    except ET.ParseError as e:
        print(f"ERROR: GROBID returned malformed XML: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR processing PDF: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PDF using local Grobid service for LLM ingestion.")
    parser.add_argument("pdf_path", help="Path to the target PDF file")
    args = parser.parse_args()

    process_pdf(args.pdf_path)
