"""
Central configuration for the Ask My Docs backend.

All values can be overridden with environment variables so the same code
runs the same way locally, in Docker, and in CI.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent  # backend/app
BACKEND_DIR = BASE_DIR.parent  # backend/
PROJECT_ROOT = BACKEND_DIR.parent  # repo root
FRONTEND_DIR = PROJECT_ROOT / "frontend"

DATA_DIR = BASE_DIR / "data" / "sample_docs"
DB_PATH = Path(os.getenv("APP_DB_PATH", str(BACKEND_DIR / "app_data.db")))

# --- LLM settings -----------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# --- Retrieval settings -------------------------------------------------
# Embedding backend for the "vector" side of hybrid search.
# "sbert"  -> sentence-transformers bi-encoder (needs internet the first
#             time to download the model). Falls back to "tfidf" automatically
#             if the library/model isn't available.
# "tfidf"  -> pure scikit-learn TF-IDF + SVD, fully offline, no downloads.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sbert")
SBERT_MODEL_NAME = os.getenv("SBERT_MODEL_NAME", "all-MiniLM-L6-v2")

# Cross-encoder reranker. Same offline-fallback behaviour as embeddings.
RERANK_BACKEND = os.getenv("RERANK_BACKEND", "cross_encoder")
CROSS_ENCODER_MODEL_NAME = os.getenv(
    "CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# How many candidates each retriever pulls before fusion / reranking.
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10"))
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "10"))
# How many candidates survive hybrid fusion and go into the reranker.
FUSION_TOP_K = int(os.getenv("FUSION_TOP_K", "8"))
# How many reranked chunks are finally handed to the LLM as context.
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "4"))

# Reciprocal Rank Fusion constant (standard default is 60).
RRF_K = int(os.getenv("RRF_K", "60"))

# Chunking
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "120"))

# --- Citation enforcement -------------------------------------------------
# Minimum fraction of answer sentences that must carry a [n] citation
# marker before we accept the answer as-is. Below this we fall back to a
# safe, fully-grounded extractive answer instead of an unsupported one.
MIN_CITED_SENTENCE_RATIO = float(os.getenv("MIN_CITED_SENTENCE_RATIO", "0.6"))

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

# --- Monitoring -----------------------------------------------------------
# Approximate USD price per token, used to estimate LLM spend for the
# askmydocs_llm_cost_usd_total metric (Grafana "cost" panels). These are
# illustrative defaults -- check https://openai.com/api/pricing/ and update
# to current rates for accurate cost tracking in your own deployment.
MODEL_PRICING = {
    "gpt-4o-mini": {"prompt": 0.15 / 1_000_000, "completion": 0.60 / 1_000_000},
    "gpt-4o": {"prompt": 2.50 / 1_000_000, "completion": 10.00 / 1_000_000},
    "gpt-4.1-mini": {"prompt": 0.40 / 1_000_000, "completion": 1.60 / 1_000_000},
}
DEFAULT_PRICING = {"prompt": 0.0, "completion": 0.0}
