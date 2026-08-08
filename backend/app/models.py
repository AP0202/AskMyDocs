"""Pydantic schemas shared across the API layer."""

from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    doc_type: str
    date: Optional[str] = None


class DocumentDetail(DocumentSummary):
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    document_id: Optional[str] = Field(
        default=None,
        description="Restrict retrieval to a single document. Omit to search "
        "across all of the patient's documents.",
    )
    conversation_id: Optional[str] = None


class Citation(BaseModel):
    marker: str  # e.g. "[1]"
    doc_id: str
    doc_title: str
    section: str
    text: str
    score: float


class AskResponse(BaseModel):
    conversation_id: str
    question: str
    answer: str
    citations: List[Citation]
    disclaimer: str
    grounded: bool
    retrieval_debug: Optional[dict] = None


class FeedbackRequest(BaseModel):
    conversation_id: str
    feedback: int = Field(..., ge=-1, le=1, description="-1, 0, or 1")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    message: str
