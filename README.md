# RAG Proof of Concept

This repository is a learning-oriented Retrieval-Augmented Generation (RAG) pipeline. It accepts PDF documents, extracts their text, creates contextual chunks, generates local Nomic embeddings, stores them in Pinecone, and answers user questions with a local `gpt-oss:20b` model through Ollama.

The current example document is `Source/pdf/RAG demo.pdf`. Its pipeline output contains 25 contextual chunks and 25 corresponding 768-dimensional vectors.

## Architecture

```text
PDF
 │
 ├─ convert_pdfs_to_markdown.py ──> Source/markdown/*.md
 │
 ├─ chunk_markdown_sections.py ──> Source/chunks/*.jsonl
 │
 ├─ embed_chunks_with_nomic.py ──> Source/vectors/*.jsonl
 │                                      │
 │                                      └─ nomic-embed-text through Ollama
 │
 ├─ upsert_to_pinecone.py ────────> Pinecone namespace
 │
 └─ ask_rag.py
       ├─ embeds a user question with Nomic
       ├─ retrieves the most similar Pinecone chunks
       └─ asks gpt-oss:20b to answer only from that context
```

## Requirements

- Windows with Python 3.15 or another supported Python version.
- [Ollama](https://ollama.com/) running locally.
- A Pinecone account and a **standard raw-vector dense index**.

Install the two Ollama models once:

```powershell
ollama pull nomic-embed-text
ollama pull gpt-oss:20b
ollama serve
```

Create and activate a virtual environment, then install project dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Pinecone setup

Create a Pinecone index intended for precomputed vectors. Do **not** select an integrated-embedding, document-schema, or assistant index.

| Setting | Required value | Why |
| --- | --- | --- |
| Vector type | Dense / bring your own vectors | This PoC generates vectors locally. |
| Dimension | `768` | `nomic-embed-text` produces 768-dimensional embeddings. |
| Metric | `cosine` | Correct similarity metric for the generated vectors. |
| Embedding model | None | Pinecone must accept the raw Nomic vector values. |
| Namespace | `rag-demo` or another chosen name | Separates this document collection from others. |

The current implementation uses the Pinecone index host, which is available on the index details page in the Pinecone console.

## Environment configuration

Create a repository-root `.env` file. It is ignored by Git and must never be committed.

```dotenv
# Required for Pinecone upserts and retrieval
PINECONE_API_KEY=replace_with_your_key
PINECONE_INDEX_HOST=your-index-host.pinecone.io
PINECONE_NAMESPACE=rag-demo

# Optional: defaults are shown below
PINECONE_VECTORS_PATH=Source/vectors/RAG demo.jsonl
PINECONE_BATCH_SIZE=50
RAG_TOP_K=3
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_CHAT_MODEL=gpt-oss:20b
```

`PINECONE_API_KEY` is the only secret in this file. `PINECONE_INDEX_HOST` should identify the raw-vector index created above, not an index name and not an integrated-embedding index.

## Run the ingestion pipeline

Run all commands from the repository root. Each stage processes every matching input file in its source folder and safely replaces its generated output when run again.

### 1. Convert PDFs to Markdown

```powershell
.venv\Scripts\python.exe Ingestion\convert_pdfs_to_markdown.py
```

Input: `Source/pdf/*.pdf`  
Output: `Source/markdown/<document name>.md`

`convert_pdfs_to_markdown.py` uses `pypdf.PdfReader` to extract a PDF's embedded text layer. It creates a document title (`# <document name>`) and an `## Page N` heading for every page. This is intentionally lightweight and does not perform OCR; scanned PDFs without an embedded text layer require an OCR-capable extraction step.

### 2. Create contextual chunks

```powershell
.venv\Scripts\python.exe Ingestion\chunk_markdown_sections.py
```

Input: `Source/markdown/*.md`  
Output: `Source/chunks/<document name>.jsonl`

`chunk_markdown_sections.py` chunks on document structure rather than an arbitrary character count:

- Markdown headings maintain the parent document/page hierarchy.
- Numbered lines such as `4. THE ROTUNDA` become sub-sections.
- All-uppercase topic lines such as `THE SOUTH LAWN` also become sub-sections.
- Each chunk stores a contextual `text` field that prepends the document and full section path before the source content.

This context is important: a chunk containing only body text such as “It was designed by Thomas Jefferson” becomes much more useful when paired with `Document: RAG demo` and `Section: RAG demo > Page 3 > 4. THE ROTUNDA`.

Each JSONL line has this shape:

```json
{
  "chunk_id": "RAG demo-section-6",
  "source_file": "Source/markdown/RAG demo.md",
  "section_path": ["RAG demo", "Page 3", "4. THE ROTUNDA"],
  "page_number": 3,
  "content": "The Rotunda was designed by Thomas Jefferson...",
  "text": "Document: RAG demo\nSection: RAG demo > Page 3 > 4. THE ROTUNDA\n\nThe Rotunda was designed..."
}
```

### 3. Create local Nomic vectors

```powershell
.venv\Scripts\python.exe Ingestion\embed_chunks_with_nomic.py
```

Input: `Source/chunks/*.jsonl`  
Output: `Source/vectors/<document name>.jsonl`

`embed_chunks_with_nomic.py` sends batches of 16 contextual chunk texts to Ollama's `/api/embed` endpoint using `nomic-embed-text`. Each input is prefixed with `search_document:` to identify it as retrieval corpus content. The script retains every chunk field and adds:

```json
{
  "embedding_model": "nomic-embed-text",
  "embedding": [0.0123, -0.0456]
}
```

The real vector has 768 floating-point values. It is stored in JSONL locally so embeddings can be inspected or re-upserted without recomputing them.

### 4. Upsert vectors and metadata to Pinecone

```powershell
.venv\Scripts\python.exe Ingestion\upsert_to_pinecone.py
```

`upsert_to_pinecone.py` reads `.env`, validates every vector has 768 dimensions, and upserts records in batches. Each Pinecone record uses:

- `id`: `chunk_id`, for example `RAG demo-section-6`.
- `values`: the 768-dimensional Nomic embedding.
- `metadata`: `source_file`, `section_path`, `page_number`, `content`, `text`, and `embedding_model`.

Metadata is stored with the vector so the answer agent can retrieve both the numerical match and the exact source text required to answer and cite the question.

## Ask questions with the RAG agent

Ask one question:

```powershell
.venv\Scripts\python.exe Ingestion\ask_rag.py --question "What is the Rotunda?"
```

Or start an interactive session:

```powershell
.venv\Scripts\python.exe Ingestion\ask_rag.py --interactive
```

Optional retrieval override:

```powershell
.venv\Scripts\python.exe Ingestion\ask_rag.py --question "Where is first-year housing?" --top-k 5
```

### What `ask_rag.py` does

1. Prefixes the question with `search_query:` and creates a Nomic query vector.
2. Queries the configured Pinecone namespace using cosine similarity.
3. Selects the top `k` chunks (`3` by default).
4. Labels each supplied chunk with its source chunk ID and similarity score.
5. Sends the selected context to Ollama's `gpt-oss:20b` model.
6. Uses a grounding instruction: answer only from context, cite factual claims with chunk IDs, and state when the answer is unavailable.
7. Prints the answer, retrieved chunk IDs/scores, and embedding, retrieval, generation, and total latency.

Example validation output from this repository:

```text
RAG demo-section-6 | score=0.7758 | RAG demo > Page 3 > 4. THE ROTUNDA
Embedding: 71 ms
Retrieval: 836 ms
Generation: 6316 ms
Total: 7224 ms
```

The answer correctly identified the Rotunda and cited `RAG demo-section-6`. Generation time is the main latency cost; the Nomic embedding and Pinecone retrieval together were under one second in this test.

## How to validate RAG quality

RAG quality has two separate parts:

1. **Retrieval quality:** Does the expected source chunk appear near the top of the retrieved list?
2. **Answer quality:** Is the answer correct, grounded in retrieved text, and properly cited?

Use a small test set containing questions whose expected source section is already known. For each question, record the expected `chunk_id`, run `ask_rag.py`, and check whether that ID is in the top 3 results. Also confirm that the answer contains a valid citation and does not introduce unsupported facts.

Suggested test questions for the current document:

| Question | Expected source section |
| --- | --- |
| What is the Rotunda? | `RAG demo-section-6` |
| What does Peabody Hall contain? | `RAG demo-section-3` |
| Where do first-year students live? | `RAG demo-section-13` |
| What is the purpose of the Honor System? | The `THE HONOR SYSTEM` chunk |
| Does the document mention a swimming facility? | `ATHLETIC FACILITIES` chunk |

The displayed similarity score is useful for inspection but is not an absolute confidence value. Relevance depends on the document, question wording, chunk size, and embedding model.

## Current optimization choices

- **Contextual chunks:** Document/page/subsection labels are embedded with the chunk rather than relying on body text alone.
- **Section-based splitting:** Keeps each campus location/topic together instead of cutting sentences at a fixed character boundary.
- **Local embedding:** Nomic runs in Ollama, so chunk embedding requires no remote embedding API key.
- **Batch embedding:** 16 chunks are embedded per request, reducing local HTTP overhead.
- **Pinecone metadata:** Avoids a second source-file lookup after retrieval.
- **Top 3 retrieval default:** Keeps the `gpt-oss:20b` prompt focused and limits generation latency. Increase it only when the expected answer spans multiple sections.
- **Temperature 0:** Makes answer generation more repeatable for evaluation.
- **400-token generation cap:** Limits overly long answers and keeps latency predictable.

## Troubleshooting

| Problem | Check / fix |
| --- | --- |
| `No PDF files found` | Place PDFs in `Source/pdf`. |
| Markdown lacks text | The source may be a scanned PDF. Use OCR before or instead of the `pypdf` stage. |
| `Cannot reach Ollama` | Start it with `ollama serve`; check `ollama list`. |
| Model is missing | Run `ollama pull nomic-embed-text` and `ollama pull gpt-oss:20b`. |
| Pinecone says “document schema” | Use a separate raw-vector dense index with no integrated embedding model. |
| Pinecone dimension error | Confirm the Pinecone index dimension is `768`. |
| Answers are unfocused | Try `--top-k 2` or improve the Markdown subsection headings. |
| Answers lack evidence | Confirm the retrieved chunk list includes the expected source section before tuning the LLM prompt. |

## Project structure

```text
Ingestion/
  convert_pdfs_to_markdown.py       PDF text extraction
  chunk_markdown_sections.py        Contextual section/subsection chunking
  embed_chunks_with_nomic.py        Local Nomic embedding generation
  upsert_to_pinecone.py             Pinecone vector and metadata upsert
  ask_rag.py                        Question-answering RAG agent
Source/
  pdf/                              Input PDFs
  markdown/                         Generated Markdown
  chunks/                           Generated contextual JSONL chunks
  vectors/                          Generated JSONL chunks plus embeddings
.env                                Local configuration and Pinecone key (ignored)
requirements.txt                    Python dependencies
```
