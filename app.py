import streamlit as st
import pdfplumber
import json
import re
import time
from typing import List, Literal, Optional

from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(layout="wide", page_title="Scope Translator (OpenAI)")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("CRITICAL: OPENAI_API_KEY missing. Add it to Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# SAFE LIBRARY (Puppet Master)
# -----------------------------
RESPONSE_TEMPLATES = {
    "UNDEFINED_BOUNDARY": {
        "category": "Common Clarification Area",
        "gap": "The document requires matching or connecting but does not currently define the physical boundary.",
        "question": "Is the transition point the immediate area or the nearest corner?"
    },
    "SUBJECTIVE_QUALITY": {
        "category": "Implied Responsibility",
        "gap": "The term implies a quality standard without citing a specific metric.",
        "question": "Does the document reference a measurable standard or tolerance for acceptance?"
    },
    "UNDEFINED_SCOPE": {
        "category": "Common Clarification Area",
        "gap": "The document implies a complete result ('turnkey') but does not list specific inclusions.",
        "question": "Does the document specify boundaries regarding accessories, furniture, or final cleaning?"
    },
    "EXPLICIT_LIABILITY": {
        "category": "Contractual Commitment",
        "gap": "The document assigns specific financial or schedule liability.",
        "question": "Does the contract account for excusable delays or reciprocal conditions?"
    },
    "COORDINATION_GAP": {
        "category": "Implied Responsibility",
        "gap": "The scope involves multiple trades without defining priority.",
        "question": "Does the document assign priority in this zone to prevent schedule stacking?"
    },
    "IF_POSSIBLE": {
        "category": "Common Clarification Area",
        "gap": "The document includes conditional language without defining the decision rule.",
        "question": "Does the document define who decides, by when, and what happens if it is not possible?"
    },
    "AS_NEEDED": {
        "category": "Common Clarification Area",
        "gap": "The document requires work 'as needed' without defining quantity or limit.",
        "question": "Does the document define a not-to-exceed limit or specific criteria for when this work is required?"
    },
    "REQUIRED_UPGRADES": {
        "category": "Common Clarification Area",
        "gap": "The document references required upgrades without listing scope.",
        "question": "Does the document specify the upgrade scope (panel rating, circuits, load calc, etc.)?"
    }
}

ClassificationKey = Literal[
    "UNDEFINED_BOUNDARY",
    "SUBJECTIVE_QUALITY",
    "UNDEFINED_SCOPE",
    "EXPLICIT_LIABILITY",
    "COORDINATION_GAP",
    "IF_POSSIBLE",
    "AS_NEEDED",
    "REQUIRED_UPGRADES",
]

# Pre-scan triggers to avoid pointless API calls
TRIGGERS = {
    "UNDEFINED_BOUNDARY": ["match existing", "tie into", "patch", "patching", "repair"],
    "SUBJECTIVE_QUALITY": ["industry standard", "workmanlike", "satisfaction of"],
    "UNDEFINED_SCOPE": ["turnkey", "complete system", "including but not limited to"],
    "EXPLICIT_LIABILITY": ["liquidated damages", "time is of the essence", "indemnify"],
    "COORDINATION_GAP": ["coordinate with", "verify in field", "by others"],
    "IF_POSSIBLE": ["if possible", "where possible", "if feasible"],
    "AS_NEEDED": ["as needed", "as required", "as necessary"],
    "REQUIRED_UPGRADES": ["required upgrades", "bring to code", "code upgrades", "upgrade", "upgrades"],
}

ALL_TRIGGER_PHRASES = sorted({p for phrases in TRIGGERS.values() for p in phrases}, key=len, reverse=True)

# -----------------------------
# PROMPT
# -----------------------------
SYSTEM_INSTRUCTION = """
ROLE: You are a strict classifier.

TASK:
- Identify ambiguity triggers in the provided text.
- Return ONLY a JSON array.
- Each object must include:
  - trigger_text: the EXACT substring copied from the input text
  - classification: one of the allowed keys

ALLOWED classification KEYS:
UNDEFINED_BOUNDARY
SUBJECTIVE_QUALITY
UNDEFINED_SCOPE
EXPLICIT_LIABILITY
COORDINATION_GAP
IF_POSSIBLE
AS_NEEDED
REQUIRED_UPGRADES

RULES:
- Extract exact text only. No paraphrasing.
- Do not include reasoning.
- Ignore performance verbs like optimize, maximize, ensure.
- Return [] if nothing found.
"""

# -----------------------------
# TEXT UTILITIES
# -----------------------------
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def chunk_text(text: str, max_chars: int = 2400) -> List[str]:
    text = text.strip()
    if not text:
        return []
    paras = re.split(r"\n\s*\n", text)
    chunks = []
    cur = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p).strip() if cur else p
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks

def chunk_has_any_trigger(chunk: str) -> bool:
    nc = normalize_text(chunk)
    for phrase in ALL_TRIGGER_PHRASES:
        if normalize_text(phrase) in nc:
            return True
    return False

# -----------------------------
# PDF EXTRACTION (fixes spacing)
# -----------------------------
def extract_page_text_linegroup(page) -> str:
    """
    Rebuilds readable lines by grouping words with similar 'top' (y) positions.
    This dramatically improves spacing and hit-rate for trigger phrases.
    """
    words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False, use_text_flow=True)
    if not words:
        return ""

    # Sort by y then x
    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))

    lines = []
    current_line = []
    current_top = None

    for w in words:
        top = round(w["top"], 1)
        if current_top is None:
            current_top = top

        # New line if y jumps
        if abs(top - current_top) > 2.0:
            if current_line:
                lines.append(" ".join(current_line).strip())
            current_line = [w["text"]]
            current_top = top
        else:
            current_line.append(w["text"])

    if current_line:
        lines.append(" ".join(current_line).strip())

    # Light cleanup: add space between camelCase collisions sometimes caused by PDFs
    text = "\n".join(lines)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    return text.strip()

# -----------------------------
# OPENAI CALL (with backoff)
# -----------------------------
def call_openai_classifier(chunk: str, model_name: str, max_retries: int = 3) -> List[dict]:
    prompt = f"{SYSTEM_INSTRUCTION}\n\nTEXT TO ANALYZE:\n{chunk}"

    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model_name,
                input=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": f"TEXT TO ANALYZE:\n{chunk}"},
                ],
                text={"format": {"type": "json_object"}},
            )

            # Responses API can return multiple output items. We want the final text blob.
            out_text = ""
            for item in resp.output:
                if item.type == "message":
                    for c in item.content:
                        if c.type == "output_text":
                            out_text += c.text

            out_text = out_text.strip()
            if not out_text:
                return []

            data = json.loads(out_text)
            # We accept either {"items":[...]} or just [...]
            if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
                return data["items"]
            if isinstance(data, list):
                return data
            return []

        except Exception as e:
            msg = str(e)
            # Backoff on 429 / rate limits
            if "429" in msg or "rate" in msg.lower():
                time.sleep(2 ** attempt)
                continue
            return []

    return []

# -----------------------------
# SCAN ENGINE
# -----------------------------
def scan_document(pages_data: List[dict], model_name: str) -> List[dict]:
    findings = []
    seen = set()

    # Count chunks for progress
    all_chunks = []
    for p in pages_data:
        for ch in chunk_text(p["text"]):
            if chunk_has_any_trigger(ch):
                all_chunks.append((p["page"], ch))

    total = len(all_chunks)
    progress = st.progress(0)
    done = 0

    for page_num, chunk in all_chunks:
        items = call_openai_classifier(chunk, model_name=model_name)

        for it in items:
            key = it.get("classification", "")
            quote = it.get("trigger_text", "")

            if key not in RESPONSE_TEMPLATES:
                continue
            if not quote:
                continue

            # Hallucination guard: exact substring (case-insensitive check)
            if normalize_text(quote) not in normalize_text(chunk):
                continue

            uniq = f"{page_num}|{normalize_text(quote)}|{key}"
            if uniq in seen:
                continue
            seen.add(uniq)

            t = RESPONSE_TEMPLATES[key]
            findings.append(
                {
                    "phrase": quote,
                    "category": t["category"],
                    "gap": t["gap"],
                    "question": t["question"],
                    "page": page_num,
                }
            )

        done += 1
        progress.progress(min(done / max(total, 1), 1.0))

    progress.empty()
    # Sort by page then category for stable output
    findings.sort(key=lambda x: (x["page"], x["category"], x["phrase"].lower()))
    return findings

# -----------------------------
# UI
# -----------------------------
st.title("Scope Translator (OpenAI)")
st.markdown("**Ethos:** Move the burden from the person to the document.")
st.divider()

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("1. Source Document")
    uploaded_file = st.file_uploader("Upload Scope PDF", type="pdf")

    pages_data = []
    full_text_display = ""

    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = extract_page_text_linegroup(page)
                if page_text:
                    pages_data.append({"page": i + 1, "text": page_text})
                    full_text_display += f"--- Page {i+1} ---\n{page_text}\n\n"

        st.text_area("Extracted Text Content", full_text_display, height=600)

with col2:
    st.subheader("2. Analysis")

    # Pick a model. Default to a reliable general model.
    model_name = st.selectbox(
        "Model",
        options=[
            "gpt-5.2",
            "gpt-5.2-mini",
            "gpt-4o",
            "gpt-4o-mini",
        ],
        index=0,
    )

    if "findings" not in st.session_state:
        st.session_state.findings = None

    if pages_data:
        if st.button("Run Strict Analysis"):
            with st.spinner("Applying filters..."):
                st.session_state.findings = scan_document(pages_data, model_name=model_name)

        if st.session_state.findings is not None:
            results = st.session_state.findings
            st.info(f"**Scan Complete.** Found {len(results)} items.")

            for item in results:
                with st.container():
                    st.markdown(f"### 🔹 {item['category']}")
                    st.caption(f"**Found:** \"{item['phrase']}\" [Page {item['page']}]")
                    st.markdown(f"**Gap:** {item['gap']}")
                    st.markdown(f"**Clarification:** {item['question']}")
                    st.divider()
    else:
        st.write("Upload a document to begin.")
