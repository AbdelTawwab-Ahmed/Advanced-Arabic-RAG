import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

# --- Models ---
GEMINI_MODEL = "gemini-2.5-flash"
GROQ_FALLBACK_MODEL = "openai/gpt-oss-120b"   # "qwen/qwen3.6-27b"
COHERE_EMBED_MODEL = "embed-multilingual-v3.0"
COHERE_RERANK_MODEL = "rerank-multilingual-v3.0"
SPARSE_MODEL_NAME = "Qdrant/bm25"

# --- Temperatures ---
TEMP_DETERMINISTIC = 0    # routing, decomposition, generation, reranking-sensitive tasks
TEMP_VARIED = 0.3         # multi-query, HyDE — benefit from phrasing diversity

# --- Qdrant / retrieval ---
QDRANT_COLLECTION_NAME = "arabic_rag_chunks"
DENSE_VECTOR_SIZE = 1024
RETRIEVAL_TOP_K_SINGLE = 15
RETRIEVAL_TOP_K_PER_SUBQUERY = 5
RERANK_TOP_N = 5

# --- Chunking ---
MAX_TOKENS_PER_CHUNK = 400
TEXT_TITLE_MAX_WORDS = 12

# --- Batching / rate-limit protection ---
COHERE_EMBED_BATCH_SIZE = 96
COHERE_EMBED_THROTTLE_SECONDS = 0.5
QDRANT_UPSERT_BATCH_SIZE = 50
EVAL_BATCH_DELAY_SECONDS = 2.0

# --- Cost (USD per 1M tokens, Gemini 2.5 Flash) ---
GEMINI_PRICE_PER_1M_INPUT = 0.30
GEMINI_PRICE_PER_1M_OUTPUT = 2.50