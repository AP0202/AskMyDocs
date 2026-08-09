# Ask My Docs

A production-style **Healthcare RAG** application that helps patients understand
their own medical documents — lab reports, discharge summaries, radiology
reports, and medication sheets — in plain language, with every answer traced
back to the exact source passage it came from.

Structurally inspired by [alexeygrigorev/fitness-assistant](https://github.com/alexeygrigorev/fitness-assistant),
rebuilt for the healthcare domain with a hybrid-retrieval + reranking +
citation-enforcement RAG pipeline and a CI-gated evaluation suite.

> **Demo data only.** All documents in `backend/app/data/sample_docs/` are
> synthetic, fictional records ("Jane Sample, Demo Patient"). No real patient
> data is included or required to run this project.

## Project overview

**Use cases:**
1. **Result interpretation** — "Why is my white blood cell count flagged?"
2. **Discharge understanding** — "Why was I admitted, and what do I need to do now?"
3. **Medication guidance** — "What side effects should I watch for?"
4. **Cross-document Q&A** — ask across the whole chart, not just one file.

Every answer:
- is generated (or extracted) **only** from the patient's own retrieved documents,
- carries `[1]`, `[2]` … citation markers linking back to the exact chunk,
- is checked by a **citation-enforcement** pass before being shown — if the
  model's answer isn't adequately cited, the app swaps in a safe, fully
  grounded extractive answer instead of risking an unsupported claim,
- comes with a visible medical disclaimer ("not medical advice, talk to your
  care team").

## Architecture

```
                     ┌─────────────────────────────────────────┐
                     │              FastAPI backend             │
  Browser  ── REST ─▶│  /api/ask  /api/documents  /api/feedback │
 (frontend/)          │                                          │
                     │   ┌──────────────┐   ┌─────────────────┐ │
                     │   │ BM25 (lexical)│   │ Vector (dense)  │ │
                     │   └──────┬───────┘   └────────┬────────┘ │
                     │          └─────────┬───────────┘          │
                     │            Hybrid fusion (RRF)             │
                     │                    │                        │
                     │           Cross-encoder reranker            │
                     │                    │                        │
                     │      LLM generation + citation enforcement  │
                     │                    │                        │
                     │              SQLite conversation log         │
                     └─────────────────────────────────────────┘
```

## Technologies

- **Backend:** Python 3.11, FastAPI, Uvicorn
- **Retrieval:** [`rank-bm25`](https://pypi.org/project/rank-bm25/) (lexical),
  `sentence-transformers` bi-encoder (dense/semantic) with an offline
  TF-IDF + SVD fallback
- **Fusion:** Reciprocal Rank Fusion (RRF)
- **Reranking:** `sentence-transformers` cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with an offline lexical-overlap
  fallback
- **Generation:** OpenAI (`gpt-4o-mini` by default) with a deterministic
  extractive fallback when no API key is set, so the app runs with **zero
  required API keys**
- **Storage:** SQLite (conversation + feedback log)
- **Frontend:** plain HTML/CSS/JavaScript, no build step, no framework
- **Monitoring:** `prometheus-client` instrumentation in the backend +
  Prometheus + Grafana, provisioned automatically via Docker Compose
- **Evaluation / CI:** `pytest` + custom retrieval & RAG evaluation scripts,
  gated in GitHub Actions

## Project structure

```
ask-my-docs/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app + routes, mounts frontend/
│   │   ├── config.py          # all tunables, env-driven
│   │   ├── models.py          # pydantic request/response schemas
│   │   ├── ingest.py          # load + chunk documents
│   │   ├── rag.py             # orchestration + citation enforcement
│   │   ├── llm.py             # OpenAI call + extractive fallback
│   │   ├── db.py              # sqlite conversation/feedback log
│   │   ├── metrics.py         # Prometheus metrics (latency/tokens/cost)
│   │   ├── retrieval/
│   │   │   ├── bm25_search.py     # lexical retriever
│   │   │   ├── vector_search.py   # dense retriever (sbert / tfidf)
│   │   │   ├── hybrid.py          # RRF fusion
│   │   │   └── reranker.py        # cross-encoder / lexical reranker
│   │   └── data/sample_docs/  # synthetic healthcare documents
│   ├── eval/
│   │   ├── ground_truth.json      # labeled Q&A pairs
│   │   ├── evaluate_retrieval.py  # hit-rate / MRR, CI-gated
│   │   └── evaluate_rag.py        # groundedness / citation validity, CI-gated
│   ├── tests/                 # pytest unit + API tests
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── monitoring/
│   ├── prometheus/prometheus.yml          # scrape config (targets the app)
│   └── grafana/
│       ├── provisioning/datasources/      # auto-registers Prometheus in Grafana
│       ├── provisioning/dashboards/       # auto-loads the dashboard below
│       └── dashboards/ask-my-docs.json    # the Grafana dashboard itself
├── scripts/
│   └── generate_demo_traffic.py   # sends sample questions to populate the dashboard
├── .github/workflows/ci.yml   # test + evaluation gate
├── main.py                    # single entrypoint: runs backend + serves frontend
├── requirements.txt
├── Dockerfile
├── docker-compose.yml          # app + Prometheus + Grafana, one command
└── .env.example
```

## Running the application

### Option A — Python virtual environment (recommended for development)

```bash
git clone <this-repo-url> ask-my-docs
cd ask-my-docs

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env             # optional: add OPENAI_API_KEY to enable LLM generation

python main.py
```

Open **http://localhost:8000** in your browser. The same process serves the
`/api/*` REST endpoints and the static frontend.

> First run note: with the default `EMBEDDING_BACKEND=sbert` /
> `RERANK_BACKEND=cross_encoder`, the app downloads two small models
> (~90MB total) from Hugging Face on first startup. No internet? Set
> `EMBEDDING_BACKEND=tfidf` and `RERANK_BACKEND=lexical` in `.env` for a
> fully offline run — the app falls back to these automatically anyway if
> the download fails.

### Option B — Docker Compose (one command, fully offline by default)

```bash
cp .env.example .env             # optional: add OPENAI_API_KEY
docker compose up --build
```

This starts three containers:

| Service      | URL                          | Purpose                              |
|--------------|-------------------------------|---------------------------------------|
| `app`        | http://localhost:8000         | The RAG application (API + frontend) |
| `prometheus` | http://localhost:9090         | Scrapes `/metrics` from `app` every 5s |
| `grafana`    | http://localhost:3000         | Pre-provisioned monitoring dashboard  |

Grafana logs in as `admin` / `admin` (also enabled for anonymous viewing).
Open it, and **Ask My Docs — RAG Monitoring** is already there under
Dashboards — no manual setup needed, it's auto-provisioned from
`monitoring/grafana/`.

The container image defaults to the offline TF-IDF / lexical-overlap
backends so it starts instantly with no network access required; add
`OPENAI_API_KEY` to `.env` to enable real LLM answers (and real token/cost
metrics).

### Option C — Docker (without compose)

```bash
docker build -t ask-my-docs .
docker run --rm -p 8000:8000 --env-file .env ask-my-docs
```

## Using the application

### Web UI
Open http://localhost:8000, pick a document (or leave "All documents"
selected), and ask a question. Click any `[1]` citation in an answer to jump
to its source passage in the right-hand rail; click a document tab's
"View full document" link to read the whole record.

### REST API directly

List documents:
```bash
curl http://localhost:8000/api/documents
```

Ask a question:
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Why is my white blood cell count flagged?"}'
```

Response:
```json
{
  "conversation_id": "a1b2c3d4-...",
  "question": "Why is my white blood cell count flagged?",
  "answer": "Your white blood cell count is elevated at 11.8 x10^3/uL... [1]",
  "citations": [
    {"marker": "[1]", "doc_id": "lab_cbc_001", "doc_title": "Complete Blood Count (CBC) with Differential", "section": "White Blood Cell (WBC) Count", "text": "...", "score": 6.1}
  ],
  "disclaimer": "This information is generated from your own uploaded documents...",
  "grounded": true
}
```

Send feedback:
```bash
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "a1b2c3d4-...", "feedback": 1}'
```

## Evaluation pipeline

Two evaluation scripts double as CI gates — they exit non-zero (failing the
build) if quality regresses below threshold:

```bash
cd backend

# Hit Rate@k and MRR of the hybrid + reranked retriever
python eval/evaluate_retrieval.py --verbose

# End-to-end groundedness, citation validity, and keyword coverage
python eval/evaluate_rag.py --verbose
```

`backend/eval/ground_truth.json` holds a small hand-labeled set of
(question → expected source document) pairs used by both scripts.

### CI gating

`.github/workflows/ci.yml` runs on every push/PR to `main`:
1. install dependencies
2. `pytest` (unit tests for retrieval/citation logic + API integration tests)
3. `evaluate_retrieval.py` — fails the build if Hit Rate or MRR drop below
   threshold
4. `evaluate_rag.py` — fails the build if groundedness, citation validity,
   or keyword coverage drop below threshold

CI runs with `EMBEDDING_BACKEND=tfidf` / `RERANK_BACKEND=lexical` for speed
and determinism (no model downloads); swap those env vars in the workflow
to exercise the full sbert/cross-encoder pipeline if you want CI to test
production-grade retrieval quality too.

## How citation enforcement works

1. The LLM (or the extractive fallback) is prompted to end every factual
   sentence with a `[n]` marker referencing one of the numbered context
   excerpts it was given.
2. `app/rag.py::enforce_citations` parses the answer into sentences and
   checks what fraction carry a valid `[n]` marker (`MIN_CITED_SENTENCE_RATIO`,
   default 60%).
3. If the answer doesn't clear that bar, it's discarded and replaced with a
   deterministic extractive answer built directly from the retrieved
   chunks — so the user always gets something traceable to their own
   documents, never an unsupported claim.
4. `evaluate_rag.py` additionally verifies no citation marker in any answer
   points at a chunk that wasn't actually retrieved (no hallucinated
   sources), across the whole ground-truth set.

## Configuration

All tunables live in `backend/app/config.py` and are overridable via `.env`
(see `.env.example`): embedding/reranker backend selection, retrieval
`top_k`s, RRF constant, chunk size, citation threshold, LLM model/temperature,
host/port.

## Monitoring: Prometheus + Grafana

The backend is instrumented with [`prometheus-client`](https://github.com/prometheus/client_python)
(`backend/app/metrics.py`) and exposes a `GET /metrics` endpoint. A
Prometheus + Grafana stack in `docker-compose.yml` scrapes it and renders a
pre-built dashboard automatically — no manual dashboard setup required.

### What's tracked

| Metric | Type | What it answers |
|---|---|---|
| `askmydocs_ask_request_duration_seconds` | Histogram | **How long it took** — end-to-end `/api/ask` latency (p50/p95/p99) |
| `askmydocs_pipeline_stage_duration_seconds{stage}` | Histogram | Latency breakdown: `hybrid_retrieval`, `rerank`, `llm_generate` |
| `askmydocs_llm_tokens_total{type,model}` | Counter | **How many tokens it used** — prompt vs completion, per model |
| `askmydocs_llm_cost_usd_total{model}` | Counter | **What it cost** — estimated USD spend, from `MODEL_PRICING` in `config.py` |
| `askmydocs_requests_total{grounded,llm_backend}` | Counter | Request volume, and grounded-answer vs extractive-fallback rate |
| `askmydocs_errors_total{endpoint}` | Counter | Failed requests by endpoint |
| `askmydocs_feedback_total{value}` | Counter | Thumbs up / down submitted by users |

Cost is estimated, not billed truth: it's `tokens x price-per-token` using
the table in `backend/app/config.py::MODEL_PRICING`, which you should keep
in sync with [OpenAI's current pricing](https://openai.com/api/pricing/).
In extractive-fallback mode (no `OPENAI_API_KEY`), token/cost metrics stay
at zero but latency is still tracked — you're still measuring real system
performance, just $0 of LLM spend.

### Running it

```bash
cp .env.example .env
docker compose up --build
```

Then open **http://localhost:3000** → Dashboards → **Ask My Docs — RAG
Monitoring**. Ask a few questions at http://localhost:8000 (or run the
traffic generator below) and watch the panels update within ~10 seconds.

To populate the dashboard with sample traffic instead of asking questions
by hand:

```bash
pip install requests   # if not already installed
python scripts/generate_demo_traffic.py --base-url http://localhost:8000 --requests 40
```

### Dashboard panels

1. **Requests per minute** — overall traffic
2. **Answer latency (p50/p95/p99)** — end-to-end response time
3. **Pipeline stage latency (p95)** — where time is spent: retrieval vs rerank vs generation
4. **LLM tokens per minute** — prompt vs completion token throughput
5. **Estimated cumulative LLM cost** — running USD total
6. **Estimated cost rate ($/hour)** — spend velocity
7. **User feedback (cumulative)** — thumbs up vs down
8. **Grounded vs. extractive-fallback answers** — how often citation enforcement had to kick in
9. **Error rate** — failed requests by endpoint

### Running without Docker

If you're running the app directly (`python main.py`, Option A above),
you can still run Prometheus + Grafana standalone and point Prometheus at
your host: edit `monitoring/prometheus/prometheus.yml`'s target from
`app:8000` to `host.docker.internal:8000` (Mac/Windows) or your machine's
LAN IP (Linux), then:

```bash
docker run --rm -p 9090:9090 \
  -v "$(pwd)/monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml" \
  prom/prometheus

docker run --rm -p 3000:3000 \
  -v "$(pwd)/monitoring/grafana/provisioning:/etc/grafana/provisioning" \
  -v "$(pwd)/monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards/json" \
  grafana/grafana
```

## Notes on taking this to real production

This project is a realistic *skeleton* for a healthcare RAG system, not a
compliant clinical product. To productionize it you'd additionally need:
- **PHI handling & HIPAA compliance**: encryption at rest/in transit, BAAs
  with any third-party LLM provider, audit logging, access controls.
- **Real document ingestion**: PDF/FHIR/HL7 parsing, OCR for scans, and a
  de-identification step before any data reaches a third-party API.
- **A vector database** (e.g. pgvector, Qdrant, Pinecone) instead of the
  in-memory index, for scale and persistence.
- **Clinical review** of prompts and disclaimers, and a human-in-the-loop
  escalation path for anything the system is unsure about.
