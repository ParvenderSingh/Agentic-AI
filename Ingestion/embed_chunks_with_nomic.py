"""Embed contextual JSONL chunks locally with Ollama's Nomic model.

Prerequisites:
    ollama serve
    ollama pull nomic-embed-text

Run:
    .venv\\Scripts\\python.exe Ingestion\\embed_chunks_with_nomic.py

The script writes one JSONL record per source chunk to ``Source/vectors``.
Each record keeps the source metadata and adds the Nomic embedding vector.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIRECTORY = REPOSITORY_ROOT / "Source" / "chunks"
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "Source" / "vectors"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
BATCH_SIZE = 16


def embeddings_for(texts: list[str]) -> list[list[float]]:
    """Request a batch of document embeddings from the local Ollama service."""
    request_body = json.dumps(
        {"model": MODEL, "input": texts, "truncate": False}
    ).encode("utf-8")
    request = Request(
        OLLAMA_EMBED_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["embeddings"]


def embed_chunk_files() -> int:
    """Create embedding-enriched JSONL files for every chunk file."""
    chunk_paths = sorted(INPUT_DIRECTORY.glob("*.jsonl")) if INPUT_DIRECTORY.exists() else []
    if not chunk_paths:
        print(f"No chunk JSONL files found in: {INPUT_DIRECTORY}")
        return 0

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for chunk_path in chunk_paths:
        chunks = [
            json.loads(line)
            for line in chunk_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        embedded_chunks: list[dict[str, object]] = []

        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            document_texts = [f"search_document: {chunk['text']}" for chunk in batch]
            embeddings = embeddings_for(document_texts)
            if len(embeddings) != len(batch):
                raise RuntimeError("Ollama returned an unexpected number of embeddings.")

            for chunk, embedding in zip(batch, embeddings, strict=True):
                embedded_chunks.append(
                    {
                        **chunk,
                        "embedding_model": MODEL,
                        "embedding": embedding,
                    }
                )

        output_path = OUTPUT_DIRECTORY / chunk_path.name
        with output_path.open("w", encoding="utf-8") as output_file:
            for chunk in embedded_chunks:
                output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        dimensions = len(embedded_chunks[0]["embedding"]) if embedded_chunks else 0
        print(f"Saved {len(embedded_chunks)} vectors ({dimensions} dimensions): {output_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(embed_chunk_files())
    except URLError:
        print(
            "Cannot reach Ollama. Start it with `ollama serve` and install the model "
            "with `ollama pull nomic-embed-text`.",
            file=sys.stderr,
        )
        raise SystemExit(1)
