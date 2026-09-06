# Advanced Arabic RAG System

A production-oriented Retrieval-Augmented Generation system for Arabic bank procedure manuals — combining hybrid search, adaptive query routing, and multi-metric evaluation, containerized for deployment.

## What it does

Given a question in Arabic (formal or dialectal), the system:
1. **Routes** the query through a LangGraph classifier to the best-fit retrieval strategy
2. **Transforms** the query if needed (rewriting vague phrasing, generating multiple phrasings, decomposing multi-part questions, or generating a hypothetical answer for semantic matching)
3. **Retrieves** relevant chunks via hybrid search (dense + sparse vectors, RRF-fused) over a Qdrant index built from real, table-heavy Arabic PDFs
4. **Reranks** results against the original query for precision
5. **Generates** a grounded, cited answer using Gemini
6. **Evaluates** itself automatically — Context Relevance, Faithfulness, Answer Relevance, and Correctness via Ragas — with full token/cost tracking

All of it is explorable through a Streamlit UI with visible citations and a batch-evaluation mode, and deployable via Docker.

## Architecture

PDF (Arabic, table-heavy)
↓ LlamaParse extraction
Table-aware chunking (multi-table pages, boilerplate filtering, role-context preservation)
↓ Cohere embeddings (dense) + FastEmbed BM25 (sparse)
Qdrant hybrid index
↓
Query → LangGraph Router → [Rewriting | Multi-Query | Decomposition | HyDE | direct]
↓
Hybrid Retrieval → Cohere Reranking → Gemini Generation
↓
Ragas Evaluation (4 metrics + cost) → Results table / Streamlit UI

Every stage lives in its own module (`src/extraction`, `src/chunking`, `src/embedding`, `src/indexing`, `src/routing`, `src/query_transformation`, `src/retrieval`, `src/reranking`, `src/generation`, `src/evaluation`), each independently testable — see `notebooks/` for the verification script used at each stage.

## Engineering challenges worked through

- **Nested and multi-table PDF pages**: the source manuals have multiple independent tables per page, repeated boilerplate headers, and role-label rows acting as sub-context — none of which naive chunking handles. Built a custom chunker that detects table boundaries, filters repeated chrome, and keeps role-context attached to split rows.
- **Rewriting vs. retrieval vocabulary drift**: query rewriting improved clarity on vague/dialectal input but sometimes drifted from the source document's exact terminology, causing a real recall gap on a multi-step procedure — diagnosed via direct comparison of retrieval pools, not assumption.
- **Provider resilience**: built a Gemini-primary/Groq-fallback LLM client to handle free-tier rate limits — including navigating a mid-project model deprecation (`mistral-saba-24b`) and reasoning-token/output-limit mismatches on the replacement model, resolved by verifying live model availability rather than trusting stale documentation.
- **Ragas + non-OpenAI LLMs**: wired Ragas (built OpenAI-first) to run against Gemini and Groq, working around API differences (no multi-completion support on Groq, LLM-mutation incompatibility with LangChain's fallback wrapper).
- **Dockerizing a Windows-developed project**: resolved WSL2/virtualization setup from scratch, and stripped Windows-only dependencies (`pywin32`, `win32_setctime`) that `pip freeze` had captured but that don't exist on Linux.

## Setup

1. Copy `.env.example` to `.env` and fill in your API keys (LlamaParse, Cohere, Qdrant, Groq)
2. `pip install -r requirements.txt`
3. Place source PDFs in `data/raw/`
4. Run the pipeline in order:
    python -m src.extraction.run_extraction
    python -m src.chunking.run_chunking
    python -m src.indexing.run_indexing

5. Launch the UI: `streamlit run ui/app.py`

## Docker

docker build -t arabic-rag-ui .
docker run -p 8501:8501 --env-file .env arabic-rag-ui
Then open `http://localhost:8501`.

## Status
✅ Full pipeline built and verified end-to-end: ingestion, adaptive routing, hybrid retrieval, reranking, generation, evaluation, UI, Docker deployment.
🚧 Evaluation question set and final written report in progress.