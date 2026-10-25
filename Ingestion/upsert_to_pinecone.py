"""Upsert local Nomic vectors to Pinecone.

Add the following to the repository-root ``.env`` file:
    PINECONE_API_KEY=...
    PINECONE_INDEX_HOST=your-index-host.pinecone.io
    PINECONE_NAMESPACE=rag-demo

Optional:
    PINECONE_VECTORS_PATH=Source/vectors/RAG demo.jsonl
    PINECONE_BATCH_SIZE=50

Run:
    .venv\\Scripts\\python.exe Ingestion\\upsert_to_pinecone.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VECTOR_DIRECTORY = REPOSITORY_ROOT / "Source" / "vectors"
REQUIRED_ENVIRONMENT_VARIABLES = (
    "PINECONE_API_KEY",
    "PINECONE_INDEX_HOST",
    "PINECONE_NAMESPACE",
)


def vector_paths() -> list[Path]:
    """Return configured vector JSONL files, defaulting to every vector file."""
    configured_path = os.getenv("PINECONE_VECTORS_PATH")
    if configured_path:
        path = Path(configured_path)
        return [path if path.is_absolute() else REPOSITORY_ROOT / path]
    return sorted(DEFAULT_VECTOR_DIRECTORY.glob("*.jsonl"))


def pinecone_record(chunk: dict[str, object]) -> dict[str, object]:
    """Map one local vector record into Pinecone's ID, values, metadata shape."""
    return {
        "id": str(chunk["chunk_id"]),
        "values": chunk["embedding"],
        "metadata": {
            "source_file": chunk["source_file"],
            "section_path": " > ".join(chunk["section_path"]),
            "page_number": chunk["page_number"],
            "content": chunk["content"],
            "text": chunk["text"],
            "embedding_model": chunk["embedding_model"],
        },
    }


def main() -> int:
    """Load contextual vector JSONL data and upsert it into Pinecone."""
    load_dotenv(REPOSITORY_ROOT / ".env")
    missing = [name for name in REQUIRED_ENVIRONMENT_VARIABLES if not os.getenv(name)]
    if missing:
        print(f"Missing required .env values: {', '.join(missing)}", file=sys.stderr)
        return 1

    paths = vector_paths()
    if not paths:
        print(f"No vector JSONL files found in: {DEFAULT_VECTOR_DIRECTORY}", file=sys.stderr)
        return 1
    absent_paths = [path for path in paths if not path.is_file()]
    if absent_paths:
        print(f"Vector file not found: {absent_paths[0]}", file=sys.stderr)
        return 1

    index_host = os.environ["PINECONE_INDEX_HOST"].removeprefix("https://")
    namespace = os.environ["PINECONE_NAMESPACE"]
    batch_size = int(os.getenv("PINECONE_BATCH_SIZE", "50"))
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(host=index_host)

    total_upserted = 0
    for path in paths:
        chunks = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records = [pinecone_record(chunk) for chunk in chunks]
        if not records:
            continue
        dimensions = len(records[0]["values"])
        if dimensions != 768 or any(len(record["values"]) != dimensions for record in records):
            raise ValueError("Expected consistent 768-dimensional Nomic embeddings.")

        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            try:
                index.upsert(vectors=batch, namespace=namespace)
            except Exception as error:
                if "document schema" in str(error).lower():
                    print(
                        "The configured Pinecone index uses an integrated document schema. "
                        "Create a standard dense vector index (dimension 768, metric cosine) "
                        "without an embedding model, then update PINECONE_INDEX_HOST in .env.",
                        file=sys.stderr,
                    )
                    return 1
                raise
            total_upserted += len(batch)
            print(f"Upserted {total_upserted} vector(s)...")

    print(f"Completed: upserted {total_upserted} vector(s) to namespace '{namespace}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
