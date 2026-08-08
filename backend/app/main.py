"""
FastAPI application: the API layer for Ask My Docs.

Endpoints:
  GET  /health                    - liveness check
  GET  /metrics                    - Prometheus metrics (scraped by Grafana stack)
  GET  /api/documents              - list the patient's documents
  GET  /api/documents/{doc_id}     - full text of one document
  POST /api/ask                    - ask a question, get a cited answer
  POST /api/feedback               - send +1 / -1 feedback on an answer

Also mounts the static frontend (frontend/) at "/", so a single process
serves both the API and the UI -- run with `python main.py` from the repo
root.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import FRONTEND_DIR
from app import db, metrics
from app.ingest import new_id
from app.models import (
    AskRequest,
    AskResponse,
    Citation,
    DocumentDetail,
    DocumentSummary,
    FeedbackRequest,
    FeedbackResponse,
)
from app.rag import RagPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ask_my_docs")

# Built once at import time; shared by every request and by eval scripts.
pipeline = RagPipeline()
db.init_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Loaded %d documents / %d chunks. vector_backend=%s rerank_backend=%s",
        len(pipeline.documents),
        len(pipeline.chunks),
        pipeline.retriever.vector.backend_name,
        pipeline.reranker.backend_name,
    )
    logger.info("Prometheus metrics available at /metrics")
    yield


app = FastAPI(
    title="Ask My Docs API",
    description="Healthcare RAG API: hybrid retrieval + reranking + citation "
    "enforcement over a patient's own medical documents.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents_loaded": len(pipeline.documents),
        "chunks_indexed": len(pipeline.chunks),
        "vector_backend": pipeline.retriever.vector.backend_name,
        "rerank_backend": pipeline.reranker.backend_name,
    }


@app.get("/metrics")
def prometheus_metrics():
    """Scraped by the Prometheus service in docker-compose.yml / monitoring/."""
    return Response(content=metrics.render_metrics(), media_type=metrics.METRICS_CONTENT_TYPE)


@app.get("/api/documents", response_model=list[DocumentSummary])
def list_documents():
    return [
        DocumentSummary(doc_id=d.doc_id, title=d.title, doc_type=d.doc_type, date=d.date)
        for d in pipeline.list_documents()
    ]


@app.get("/api/documents/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: str):
    doc = pipeline.get_document(doc_id)
    if not doc:
        metrics.record_error("get_document")
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentDetail(
        doc_id=doc.doc_id,
        title=doc.title,
        doc_type=doc.doc_type,
        date=doc.date,
        content=doc.content,
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if req.document_id and not pipeline.get_document(req.document_id):
        metrics.record_error("ask")
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        result = pipeline.answer(req.question, document_id=req.document_id)
    except Exception:
        metrics.record_error("ask")
        raise

    conversation_id = req.conversation_id or new_id()

    db.save_conversation(
        conversation_id=conversation_id,
        question=req.question,
        answer=result["answer"],
        document_id=req.document_id,
        grounded=result["grounded"],
        citations=result["citations"],
        retrieval_debug=result["retrieval_debug"],
    )

    return AskResponse(
        conversation_id=conversation_id,
        question=req.question,
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
        disclaimer=result["disclaimer"],
        grounded=result["grounded"],
        retrieval_debug=result["retrieval_debug"],
    )


@app.post("/api/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    db.save_feedback(req.conversation_id, req.feedback, req.comment)
    metrics.record_feedback(req.feedback)
    return FeedbackResponse(
        message=f"Feedback received for conversation {req.conversation_id}: {req.feedback}"
    )


# --- Static frontend ------------------------------------------------------
# Mounted last so it doesn't shadow the /api routes above.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
