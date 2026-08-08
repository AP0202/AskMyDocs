FROM python:3.11-slim

WORKDIR /srv/app

# System deps for building some scientific python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/
COPY main.py main.py

RUN mkdir -p /srv/app/data

ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
# Fully offline by default in the container image; override at `docker run`
# time (-e EMBEDDING_BACKEND=sbert -e OPENAI_API_KEY=...) to enable the
# higher-quality models / real LLM generation.
ENV EMBEDDING_BACKEND=tfidf
ENV RERANK_BACKEND=lexical

EXPOSE 8000

CMD ["python", "main.py"]
