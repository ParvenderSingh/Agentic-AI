"""Ask questions over the Pinecone-backed RAG PoC using local Ollama.

Prerequisites:
    - ``.env`` contains the Pinecone variables used by ``upsert_to_pinecone.py``.
    - Ollama is running with ``nomic-embed-text`` and ``gpt-oss:20b`` installed.

Examples:
    .venv\\Scripts\\python.exe Ingestion\\ask_rag.py --question "What is the Rotunda?"
    .venv\\Scripts\\python.exe Ingestion\\ask_rag.py --interactive

Optional .env settings:
    RAG_TOP_K=3
    OLLAMA_BASE_URL=http://127.0.0.1:11434
    OLLAMA_EMBED_MODEL=nomic-embed-text
    OLLAMA_CHAT_MODEL=gpt-oss:20b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, build_opener, ProxyHandler

from dotenv import load_dotenv
from pinecone import Pinecone


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_ENVIRONMENT_VARIABLES = (
    "PINECONE_API_KEY",
    "PINECONE_INDEX_HOST",
    "PINECONE_NAMESPACE",
)
HTTP_OPENER = build_opener(ProxyHandler({}))


def post_ollama(base_url: str, endpoint: str, payload: dict[str, object]) -> dict[str, object]:
    """Send a local-only JSON request to Ollama and return its JSON response."""
    request = Request(
        f"{base_url.rstrip('/')}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with HTTP_OPENER.open(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def question_embedding(base_url: str, model: str, question: str) -> list[float]:
    """Create a Nomic retrieval-query vector compatible with stored chunks."""
    response = post_ollama(
        base_url,
        "/api/embed",
        {"model": model, "input": [f"search_query: {question}"], "truncate": False},
    )
    return response["embeddings"][0]


def context_from_matches(matches: list[object]) -> str:
    """Format retrieved chunks into a citation-labelled LLM context block."""
    sections = []
    for match in matches:
        metadata = match.metadata or {}
        sections.append(
            f"[Source: {match.id} | score: {match.score:.4f}]\n"
            f"{metadata.get('text', metadata.get('content', ''))}"
        )
    return "\n\n---\n\n".join(sections)


def answer_question(
    index: object,
    namespace: str,
    question: str,
    top_k: int,
    base_url: str,
    embed_model: str,
    chat_model: str,
) -> None:
    """Retrieve supporting chunks, generate a grounded answer, and report metrics."""
    started_at = time.perf_counter()
    embedding_started_at = time.perf_counter()
    query_vector = question_embedding(base_url, embed_model, question)
    embedding_ms = (time.perf_counter() - embedding_started_at) * 1000

    retrieval_started_at = time.perf_counter()
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )
    retrieval_ms = (time.perf_counter() - retrieval_started_at) * 1000
    matches = result.matches
    context = context_from_matches(matches)

    system_prompt = (
        "You are a retrieval-augmented assistant. Answer only from the supplied "
        "context. Cite every factual statement with its source chunk ID in square "
        "brackets, for example [RAG demo-section-6]. If the context does not contain "
        "the answer, say that clearly instead of guessing."
    )
    generation_started_at = time.perf_counter()
    response = post_ollama(
        base_url,
        "/api/chat",
        {
            "model": chat_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContext:\n{context}",
                },
            ],
            "options": {"temperature": 0, "num_predict": 400},
        },
    )
    generation_ms = (time.perf_counter() - generation_started_at) * 1000
    answer = response["message"]["content"].strip()

    print("\nAnswer\n------")
    print(answer)
    print("\nRetrieved chunks\n----------------")
    for match in matches:
        metadata = match.metadata or {}
        print(f"{match.id} | score={match.score:.4f} | {metadata.get('section_path', '')}")
    print("\nMetrics\n-------")
    print(f"Embedding: {embedding_ms:.0f} ms")
    print(f"Retrieval: {retrieval_ms:.0f} ms")
    print(f"Generation: {generation_ms:.0f} ms")
    print(f"Total: {(time.perf_counter() - started_at) * 1000:.0f} ms")


def main() -> int:
    """Configure the agent and run either one question or an interactive session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="One question to answer.")
    parser.add_argument("--interactive", action="store_true", help="Ask questions until you exit.")
    parser.add_argument("--top-k", type=int, help="Number of Pinecone chunks to retrieve.")
    args = parser.parse_args()
    if not args.question and not args.interactive:
        parser.error("provide --question or --interactive")

    load_dotenv(REPOSITORY_ROOT / ".env")
    missing = [name for name in REQUIRED_ENVIRONMENT_VARIABLES if not os.getenv(name)]
    if missing:
        print(f"Missing required .env values: {', '.join(missing)}", file=sys.stderr)
        return 1

    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "gpt-oss:20b")
    top_k = args.top_k or int(os.getenv("RAG_TOP_K", "3"))
    if top_k < 1:
        parser.error("--top-k must be at least 1")

    index_host = os.environ["PINECONE_INDEX_HOST"].removeprefix("https://")
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(host=index_host)
    namespace = os.environ["PINECONE_NAMESPACE"]

    questions = [args.question] if args.question else None
    while True:
        question = questions.pop(0) if questions else input("\nQuestion (or 'exit'): ").strip()
        if question.lower() in {"exit", "quit"}:
            return 0
        if question:
            answer_question(index, namespace, question, top_k, base_url, embed_model, chat_model)
        if questions is not None and not questions:
            return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except URLError:
        print("Cannot reach Ollama. Start it with `ollama serve`.", file=sys.stderr)
        raise SystemExit(1)
