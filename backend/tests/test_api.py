"""
API-level tests using FastAPI's TestClient. Run with:

    cd backend
    pytest -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["documents_loaded"] > 0
    assert body["chunks_indexed"] > 0


def test_list_documents():
    res = client.get("/api/documents")
    assert res.status_code == 200
    docs = res.json()
    assert len(docs) >= 5
    assert {"doc_id", "title", "doc_type"} <= set(docs[0].keys())


def test_get_document():
    docs = client.get("/api/documents").json()
    doc_id = docs[0]["doc_id"]
    res = client.get(f"/api/documents/{doc_id}")
    assert res.status_code == 200
    assert res.json()["content"]


def test_get_document_404():
    res = client.get("/api/documents/does-not-exist")
    assert res.status_code == 404


def test_upload_documents_and_list():
    before_docs = client.get("/api/documents").json()
    before_count = len(before_docs)
    payload = (
        "TITLE: Uploaded Nutrition Notes\n"
        "DOCUMENT TYPE: Patient Notes\n"
        "SECTION: Summary\n"
        "Patient asked whether reducing sodium can lower blood pressure.\n"
    )

    res = client.post(
        "/api/documents/upload",
        files={"files": ("nutrition_notes.txt", payload, "text/plain")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["uploaded_count"] == 1
    assert body["uploaded"][0]["doc_type"] == "Uploaded Document"

    after_docs = client.get("/api/documents").json()
    assert len(after_docs) == before_count + 1


def test_upload_mixed_files_reports_skips():
    res = client.post(
        "/api/documents/upload",
        files=[
            ("files", ("notes.txt", "SECTION: A\nPatient is stable.\n", "text/plain")),
            ("files", ("broken.pdf", b"not a real pdf", "application/pdf")),
        ],
    )
    assert res.status_code == 200
    body = res.json()
    assert body["uploaded_count"] == 1
    assert len(body["skipped"]) == 1


def test_ask_returns_cited_answer():
    res = client.post(
        "/api/ask",
        json={"question": "Why is my white blood cell count flagged?"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["answer"]
    assert body["conversation_id"]
    assert body["disclaimer"]
    assert len(body["citations"]) > 0
    # every citation should have the fields the frontend depends on
    c = body["citations"][0]
    assert {"marker", "doc_id", "doc_title", "section", "text"} <= set(c.keys())


def test_ask_scoped_to_document():
    docs = client.get("/api/documents").json()
    doc_id = docs[0]["doc_id"]
    res = client.post(
        "/api/ask",
        json={"question": "What does this document say?", "document_id": doc_id},
    )
    assert res.status_code == 200
    for c in res.json()["citations"]:
        assert c["doc_id"] == doc_id


def test_ask_invalid_document_404():
    res = client.post(
        "/api/ask", json={"question": "test", "document_id": "not-a-real-doc"}
    )
    assert res.status_code == 404


def test_feedback():
    ask_res = client.post("/api/ask", json={"question": "What medications should I take?"})
    conv_id = ask_res.json()["conversation_id"]
    res = client.post(
        "/api/feedback", json={"conversation_id": conv_id, "feedback": 1}
    )
    assert res.status_code == 200
    assert conv_id in res.json()["message"]


def test_metrics_endpoint_exposes_prometheus_format():
    # Generate at least one data point first.
    client.post("/api/ask", json={"question": "Why is my white blood cell count flagged?"})

    res = client.get("/metrics")
    assert res.status_code == 200
    assert "text/plain" in res.headers["content-type"]
    body = res.text
    assert "askmydocs_ask_request_duration_seconds" in body
    assert "askmydocs_pipeline_stage_duration_seconds" in body
    assert "askmydocs_requests_total" in body
