/**
 * Ask My Docs — frontend application logic.
 * Vanilla JS, no build step. Talks to the FastAPI backend over same-origin
 * REST endpoints (see backend/app/main.py).
 */

const API = {
  documents: "/api/documents",
  document: (id) => `/api/documents/${encodeURIComponent(id)}`,
  ask: "/api/ask",
  feedback: "/api/feedback",
};

const state = {
  documents: [],
  scopeDocId: "", // "" = all documents
  conversationId: null,
  lastCitations: [],
};

const els = {
  docList: document.getElementById("doc-list"),
  scopeHeading: document.getElementById("scope-heading"),
  thread: document.getElementById("thread"),
  emptyState: document.getElementById("empty-state"),
  suggestions: document.getElementById("suggestions"),
  form: document.getElementById("ask-form"),
  input: document.getElementById("question-input"),
  sendBtn: document.getElementById("send-btn"),
  sourceList: document.getElementById("source-list"),
  modal: document.getElementById("doc-modal"),
  modalTitle: document.getElementById("modal-title"),
  modalType: document.getElementById("modal-type"),
  modalContent: document.getElementById("modal-content"),
  modalClose: document.getElementById("modal-close"),
};

const TAB_COLORS = {
  "Lab Report": "#6fb0e0",
  "Discharge Summary": "#e0a76f",
  "Radiology Report": "#c98fe0",
  "Medication Sheet": "#7fd0a0",
};

const SUGGESTIONS_BY_SCOPE = {
  "": [
    "What results in my recent labs were flagged as abnormal?",
    "Summarize why I was admitted to the hospital.",
    "What should I watch out for with my medication?",
  ],
  lab_cbc_001: [
    "Why is my white blood cell count flagged?",
    "Is my hemoglobin normal?",
  ],
  lab_lipid_002: [
    "What does my LDL cholesterol result mean?",
    "Are my triglycerides high?",
  ],
  discharge_summary_003: [
    "Why was I admitted to the hospital?",
    "What medications do I need to take after discharge?",
  ],
  radiology_report_004: [
    "What did my chest X-ray show?",
  ],
  medication_sheet_005: [
    "What are the serious side effects I should watch for?",
    "How should I take this medication?",
  ],
};

init();

async function init() {
  autoGrowTextarea();
  bindEvents();
  await loadDocuments();
  renderSuggestions();
}

function bindEvents() {
  els.form.addEventListener("submit", onAsk);
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      els.form.requestSubmit();
    }
  });
  els.input.addEventListener("input", autoGrowTextarea);
  els.modalClose.addEventListener("click", closeModal);
  els.modal.addEventListener("click", (e) => {
    if (e.target === els.modal) closeModal();
  });

  document.querySelectorAll(".scope-btn").forEach((btn) => {
    btn.addEventListener("click", () => setScope(""));
  });
}

function autoGrowTextarea() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 140) + "px";
}

async function loadDocuments() {
  try {
    const res = await fetch(API.documents);
    state.documents = await res.json();
    renderDocList();
  } catch (err) {
    els.docList.innerHTML = `<p style="color:#e0a76f;font-size:12px;">Couldn't load documents. Is the backend running?</p>`;
  }
}

function renderDocList() {
  els.docList.innerHTML = "";
  state.documents.forEach((doc) => {
    const tab = document.createElement("div");
    tab.className = "doc-tab" + (state.scopeDocId === doc.doc_id ? " active" : "");
    tab.style.setProperty("--tab-color", TAB_COLORS[doc.doc_type] || "#8fb0a9");
    tab.dataset.docId = doc.doc_id;
    tab.innerHTML = `
      <span class="doc-type">${escapeHtml(doc.doc_type)}</span>
      <span class="doc-title">${escapeHtml(doc.title)}</span>
      ${doc.date ? `<span class="doc-date">${escapeHtml(doc.date)}</span>` : ""}
      <span class="view-link" data-view="${doc.doc_id}">View full document</span>
    `;
    tab.addEventListener("click", (e) => {
      if (e.target.dataset.view) {
        e.stopPropagation();
        openDocModal(doc.doc_id);
      } else {
        setScope(doc.doc_id);
      }
    });
    els.docList.appendChild(tab);
  });
}

function setScope(docId) {
  state.scopeDocId = docId;
  document.querySelector('.scope-btn[data-doc-id=""]').classList.toggle("active", docId === "");
  renderDocList();

  const doc = state.documents.find((d) => d.doc_id === docId);
  els.scopeHeading.textContent = doc
    ? `Asking about: ${doc.title}`
    : "Asking across all documents";

  renderSuggestions();
}

function renderSuggestions() {
  const list = SUGGESTIONS_BY_SCOPE[state.scopeDocId] || SUGGESTIONS_BY_SCOPE[""];
  els.suggestions.innerHTML = "";
  list.forEach((text) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "suggestion-chip";
    chip.textContent = text;
    chip.addEventListener("click", () => {
      els.input.value = text;
      els.form.requestSubmit();
    });
    els.suggestions.appendChild(chip);
  });
}

async function openDocModal(docId) {
  try {
    const res = await fetch(API.document(docId));
    if (!res.ok) throw new Error("not found");
    const doc = await res.json();
    els.modalType.textContent = doc.doc_type;
    els.modalTitle.textContent = doc.title;
    els.modalContent.textContent = doc.content;
    els.modal.classList.remove("hidden");
  } catch (err) {
    console.error(err);
  }
}

function closeModal() {
  els.modal.classList.add("hidden");
}

async function onAsk(e) {
  e.preventDefault();
  const question = els.input.value.trim();
  if (!question) return;

  els.emptyState.style.display = "none";
  addMessage("user", question);
  els.input.value = "";
  autoGrowTextarea();

  const thinkingEl = addThinking();
  setSending(true);

  try {
    const res = await fetch(API.ask, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        document_id: state.scopeDocId || null,
        conversation_id: state.conversationId,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    state.conversationId = data.conversation_id;
    thinkingEl.remove();
    renderAnswer(data);
    renderSources(data.citations);
  } catch (err) {
    thinkingEl.remove();
    addMessage(
      "assistant",
      `Sorry, something went wrong reaching the backend: ${err.message}`
    );
  } finally {
    setSending(false);
  }
}

function setSending(sending) {
  els.sendBtn.disabled = sending;
}

function addMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  els.thread.appendChild(wrap);
  els.thread.scrollTop = els.thread.scrollHeight;
  return wrap;
}

function addThinking() {
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";
  wrap.innerHTML = `<div class="msg-bubble"><div class="thinking"><span></span><span></span><span></span></div></div>`;
  els.thread.appendChild(wrap);
  els.thread.scrollTop = els.thread.scrollHeight;
  return wrap;
}

function renderAnswer(data) {
  const wrap = document.createElement("div");
  wrap.className = "msg assistant";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.innerHTML = linkifyCitations(data.answer);
  wrap.appendChild(bubble);

  const meta = document.createElement("div");
  meta.className = "msg-meta";
  meta.innerHTML = `
    <span class="grounded-pill ${data.grounded ? "yes" : "no"}">
      ${data.grounded ? "grounded & cited" : "fallback: extractive"}
    </span>
    <span>${data.citations.length} source${data.citations.length === 1 ? "" : "s"}</span>
    <div class="feedback-btns">
      <button data-fb="1" title="Helpful">&#9650;</button>
      <button data-fb="-1" title="Not helpful">&#9660;</button>
    </div>
  `;
  meta.querySelectorAll("[data-fb]").forEach((btn) => {
    btn.addEventListener("click", () => sendFeedback(data.conversation_id, btn, meta));
  });
  wrap.appendChild(meta);

  const disclaimer = document.createElement("div");
  disclaimer.style.fontSize = "11px";
  disclaimer.style.color = "var(--ink-soft)";
  disclaimer.style.marginTop = "6px";
  disclaimer.style.maxWidth = "540px";
  disclaimer.textContent = data.disclaimer;
  wrap.appendChild(disclaimer);

  bubble.querySelectorAll(".cite").forEach((el) => {
    el.addEventListener("click", () => highlightSource(el.dataset.marker));
  });

  els.thread.appendChild(wrap);
  els.thread.scrollTop = els.thread.scrollHeight;
}

function linkifyCitations(text) {
  const escaped = escapeHtml(text);
  return escaped.replace(/\[(\d+)\]/g, (m, n) => {
    return `<span class="cite" data-marker="[${n}]">[${n}]</span>`;
  });
}

async function sendFeedback(conversationId, btn, meta) {
  const value = parseInt(btn.dataset.fb, 10);
  meta.querySelectorAll("[data-fb]").forEach((b) => b.classList.remove("selected"));
  btn.classList.add("selected");
  try {
    await fetch(API.feedback, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conversation_id: conversationId, feedback: value }),
    });
  } catch (err) {
    console.error("Feedback failed", err);
  }
}

function renderSources(citations) {
  state.lastCitations = citations;
  els.sourceList.innerHTML = "";
  if (!citations.length) {
    els.sourceList.innerHTML = `<p class="source-empty">No sources were retrieved for this question.</p>`;
    return;
  }
  citations.forEach((c) => {
    const card = document.createElement("div");
    card.className = "source-card";
    card.id = `source-${c.marker.replace(/[^\d]/g, "")}`;
    card.innerHTML = `
      <span class="source-marker">${c.marker}</span>
      <div class="source-doc">${escapeHtml(c.doc_title)}</div>
      <div class="source-section">${escapeHtml(c.section)}</div>
      <div class="source-text">${escapeHtml(c.text)}</div>
      <div class="source-score">relevance score: ${c.score.toFixed(3)}</div>
    `;
    card.addEventListener("click", () => openDocModal(c.doc_id));
    els.sourceList.appendChild(card);
  });
}

function highlightSource(marker) {
  const id = `source-${marker.replace(/[^\d]/g, "")}`;
  const card = document.getElementById(id);
  if (!card) return;
  document.querySelectorAll(".source-card").forEach((c) => c.classList.remove("highlight"));
  card.classList.add("highlight");
  card.scrollIntoView({ behavior: "smooth", block: "center" });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
