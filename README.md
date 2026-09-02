# Production-Ready-Arabic-RAG

Production-ready RAG system for Arabic PDF data — ingestion/indexing pipeline.

## Status
✅ Ingestion/indexing pipeline complete: extraction → table-aware chunking → hybrid embedding → Qdrant indexing.
🚧 Retrieval, reranking, and generation are out of scope for this phase — planned separately.

## Pipeline
1. **Extraction** (`src/extraction/`) — LlamaParse, Arabic-aware, Markdown output preserving table structure
2. **Chunking** (`src/chunking/`) — table-aware chunking: splits multiple tables per page, filters repeated boilerplate, preserves role-context and section titles, handles long-form text separately
3. **Embedding** (`src/embedding/`) — dense vectors via Cohere `embed-multilingual-v3.0`, sparse (BM25) vectors via FastEmbed, both computed for hybrid search
4. **Indexing** (`src/indexing/`) — Qdrant Cloud, named dense + sparse vectors per point, deterministic IDs for safe re-indexing

## Setup
1. Copy `.env.example` to `.env` and fill in your API keys (LlamaParse, Cohere, Qdrant)
2. `pip install -r requirements.txt`
3. Place PDFs in `data/raw/`
4. Run in order: `python -m src.extraction.run_extraction`, `python -m src.chunking.run_chunking`, `python -m src.indexing.run_indexing`