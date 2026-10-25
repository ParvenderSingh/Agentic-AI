"""Create contextual JSONL chunks from Markdown headings.

Run from the repository root (or any other working directory):
    .venv\\Scripts\\python.exe Ingestion\\chunk_markdown_sections.py

Each Markdown section becomes one JSON Lines record in ``Source/chunks``.
The ``text`` field includes the document and heading path so it can be sent
directly to an embedding model for contextual retrieval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIRECTORY = REPOSITORY_ROOT / "Source" / "markdown"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "Source" / "chunks"
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
PAGE_PATTERN = re.compile(r"^Page\s+(\d+)$", re.IGNORECASE)
NUMBERED_SUBSECTION_PATTERN = re.compile(r"^\s*\d+\.\s+(.+?)\s*$")
UPPERCASE_SUBSECTION_PATTERN = re.compile(r"^[A-Z][A-Z0-9 &'’(),./-]{3,}\s*$")


def split_subsections(content: str) -> list[tuple[str | None, str]]:
    """Split a section at numbered or all-uppercase subsection titles."""
    subsections: list[tuple[str | None, str]] = []
    title: str | None = None
    lines: list[str] = []

    def save_subsection() -> None:
        subsection_content = "\n".join(lines).strip()
        if subsection_content:
            subsections.append((title, subsection_content))

    for line in content.splitlines():
        numbered_match = NUMBERED_SUBSECTION_PATTERN.match(line)
        is_uppercase_heading = UPPERCASE_SUBSECTION_PATTERN.match(line) and line == line.upper()
        if numbered_match or is_uppercase_heading:
            save_subsection()
            title = line.strip()
            lines = []
        else:
            lines.append(line)

    save_subsection()
    return subsections


def section_chunks(markdown_path: Path) -> list[dict[str, object]]:
    """Return one contextual chunk for each non-empty Markdown section."""
    heading_stack: list[str] = []
    current_heading: str | None = None
    current_level = 0
    content_lines: list[str] = []
    chunks: list[dict[str, object]] = []

    def save_section() -> None:
        if current_heading is None:
            return

        content = "\n".join(content_lines).strip()
        if not content:
            return

        section_path = [*heading_stack, current_heading]
        page_number = next(
            (
                int(match.group(1))
                for heading in reversed(section_path)
                if (match := PAGE_PATTERN.match(heading))
            ),
            None,
        )
        document_title = markdown_path.stem

        for subsection_title, subsection_content in split_subsections(content):
            contextual_path = (
                [*section_path, subsection_title] if subsection_title else section_path
            )
            context = f"Document: {document_title}\nSection: {' > '.join(contextual_path)}"
            chunks.append(
                {
                    "chunk_id": f"{markdown_path.stem}-section-{len(chunks) + 1}",
                    "source_file": markdown_path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "section_path": contextual_path,
                    "page_number": page_number,
                    "content": subsection_content,
                    "text": f"{context}\n\n{subsection_content}",
                }
            )

    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        match = HEADING_PATTERN.match(line)
        if not match:
            content_lines.append(line)
            continue

        save_section()
        level, title = len(match.group(1)), match.group(2)
        if current_heading is not None:
            heading_stack = heading_stack[: max(current_level - 1, 0)]
            heading_stack.append(current_heading)
        heading_stack = heading_stack[: level - 1]
        current_heading = title
        current_level = level
        content_lines = []

    save_section()
    return chunks


def chunk_markdown_files() -> int:
    """Write one JSONL file per Markdown input file."""
    markdown_paths = sorted(INPUT_DIRECTORY.glob("*.md")) if INPUT_DIRECTORY.exists() else []
    if not markdown_paths:
        print(f"No Markdown files found in: {INPUT_DIRECTORY}")
        return 0

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for markdown_path in markdown_paths:
        chunks = section_chunks(markdown_path)
        output_path = OUTPUT_DIRECTORY / f"{markdown_path.stem}.jsonl"
        with output_path.open("w", encoding="utf-8") as output_file:
            for chunk in chunks:
                output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"Saved {len(chunks)} section chunk(s): {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(chunk_markdown_files())
