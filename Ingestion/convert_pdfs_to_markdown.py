"""Convert PDFs in ``Source/pdf`` into Markdown files with pypdf.

Setup:
    .venv\\Scripts\\python.exe -m pip install -r requirements.txt

Run from the repository root (or any other working directory):
    .venv\\Scripts\\python.exe Ingestion\\convert_pdfs_to_markdown.py

Every ``.pdf`` file in ``Source/pdf`` is converted to a same-named ``.md``
file in ``Source/markdown``. Existing generated Markdown files are replaced.
The converter extracts the text layer from each page; scanned PDFs require OCR
and are outside this lightweight learning PoC.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIRECTORY = REPOSITORY_ROOT / "Source" / "pdf"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "Source" / "markdown"


def convert_pdfs() -> int:
    """Convert all PDFs in the configured input directory.

    Returns a process exit code: zero when every conversion succeeds, or one
    when at least one PDF could not be converted.
    """
    pdf_paths = (
        sorted(
            path
            for path in INPUT_DIRECTORY.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
        if INPUT_DIRECTORY.exists()
        else []
    )

    if not pdf_paths:
        print(f"No PDF files found in: {INPUT_DIRECTORY}")
        return 0

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        print(
            "pypdf is not installed. Run: "
            ".venv\\Scripts\\python.exe -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    failures = 0

    for pdf_path in pdf_paths:
        output_path = OUTPUT_DIRECTORY / f"{pdf_path.stem}.md"
        print(f"Converting: {pdf_path.name}")

        try:
            reader = PdfReader(pdf_path)
            pages = [page.extract_text() or "" for page in reader.pages]
            markdown = f"# {pdf_path.stem}\n\n"
            markdown += "\n\n".join(
                f"## Page {number}\n\n{text.strip()}"
                for number, text in enumerate(pages, start=1)
            )
            output_path.write_text(markdown, encoding="utf-8")
        except Exception as error:  # Continue processing independent PDFs.
            failures += 1
            print(f"  Failed: {error}", file=sys.stderr)
            continue

        print(f"  Saved: {output_path}")

    if failures:
        print(f"Completed with {failures} failed conversion(s).", file=sys.stderr)
        return 1

    print(f"Completed {len(pdf_paths)} conversion(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(convert_pdfs())
