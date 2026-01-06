import streamlit as st
import pdfplumber
import json
import re
import time
from typing import List, Literal

from openai import OpenAI
from pydantic import BaseModel

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(layout="wide", page_title="Scope Translator (OpenAI)")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("CRITICAL: OPENAI_API_KEY missing. Add it to Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# SAFE RESPONSE TEMPLATES
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

CLASS_KEYS = Literal[
    "UNDEFINED_BOUNDARY",
    "SUBJECTIVE_QUALITY",
    "UNDEFINED_SCOPE",
    "EXPLICIT_LIABILITY",
    "COORDINATION_GAP",
    "IF_POSSIBLE",
    "AS_NEEDED",
    "REQUIRED_UPGRADES",
]

# -----------------------------
# STRUCTURED OUTPUT SCHEMA (Pydantic)
# -----------------------------
class Finding(BaseModel):
    trigger_text: str
    classification: CLASS_KEYS

class FindingsPayload(BaseModel):
    items: List[Finding]

# -----------------------------
# PROMPT (STRICT, NEUTRAL)
# -----------------------------
SYSTEM_INSTRUCTION = """
ROLE: You are a strict Classifier. You DO NOT write prose. You only output structured data.

TASK: Analyze the text snippet. Identify specific ambiguities using the KEYS below.

KEYS (Select best fit):
1. UNDEFINED_BOUNDARY (Triggers: "match existing", "tie into", "patch", "repair")
2. SUBJECTIVE_QUALITY (Triggers: "industry standard", "workmanlike", "satisfaction of")
3. UNDEFINED_SCOPE (Triggers: "turnkey", "complete system", "including but not limited to")
4. EXPLICIT_LIABILITY (Triggers: "liquidated damages", "time is of the essence", "indemnify")
5. COORDINATION_GAP (Triggers: "coordinate with", "verify in field", "by others")
6. IF_POSSIBLE (Triggers: "if possible", "where possible", "if feasible")
7. AS_NEEDED (Triggers: "as needed", "as required", "as necessary")
8. REQUIRED_UPGRADES (Triggers: "required upgrades", "bring to code", "code upgrades")

CONSTRAINTS:
- Return ONLY items that appear verbatim in the snippet.
- IGNORE performance verbs like "optimize", "maximize", "ensure".
- If nothing found, return an empty list of items.

OUTPUT:
Return a JSON object with key "items" which is a list of {trigger_text, classification}.
"""

# -----------------------------
# UTILITIES
# -----------------------------
TRIGGERS_PRE_SCAN = [
    # boundary / tie-in
    "match existing", "tie into", "patch", "repair",
    # subjective
    "industry standard", "workmanlike", "satisfaction of",
    # scope
    "turnkey", "complete system", "including but not limited to",
    # liability
    "liquidated damages", "time is of the essence", "indemnify",
    # coordination
    "coordinate with", "verify in field", "by others",
    # conditionals
    "if possible", "where possible", "if feasible",
    # as-needed
    "as needed", "as required", "as necessary",
    # upgrades
    "required upgrades", "bring to code", "code upgrades",
]

def normalize_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())

def chunk_text_smart(text: str, max_chars: int = 2200) -> List[str]:
    """
    Works even when pdfplumber text has NO paragraph breaks.
    Strategy:
      1) Try split on double-newlines (real paragraphs)
      2) Else split on sentence-ish boundaries / numbered scope items
    """
    t = text.strip()
    if not t:
        return []

    # If we have paragraph breaks, use them
    if "\n\n" in t:
        parts = t.split("\n\n")
    else:
        # Split on common scope patterns: "1.", "1.1", "2)", ";", ". "
        parts = re.split(r"(?:(?<=\.)\s+|(?<=;)\s+|(?<=\))\s+|(?<=\n)\s+|(?=\b\d{1,2}\.\d{1,2}\b)|(?=\b\d{1,2}\.\b))", t)

    chunks = []
    cur = ""
    for p in parts:
        if not p:
            continue
        if len(cur) + len(p) <= max_chars:
            cur += (p + " ")
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = p + " "
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def chunk_has_triggers(chunk: str) -> bool:
    n = normalize_text(chunk)
    return any(normalize_text(t) in n for t in TRIGGERS_PRE_SCAN)

def call_model_parse(chunk: str, model: str, max_retries: int = 6) -> FindingsPayload:
    """
    Exponential backoff on rate limits / transient errors.
    """
    backoff = 1.5
    for attempt in range(max_retries):
        try:
            resp = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": f"TEXT TO ANALYZE:\n{chunk}"},
                ],
                text_format=FindingsPayload,
            )
            return resp.output_parsed
        except Exception as e:
            msg = str(e).lower()
            # Typical rate limit / overload signals
            if "rate" in msg or "429" in msg or "overloaded" in msg or "temporarily" in msg:
                sleep_s = min(20, backoff * (attempt + 1))
                time.sleep(sleep_s)
                continue
            # Non-retryable: bubble up
            raise
    # If we ran out of retries, return empty
    return FindingsPayload(items=[])

def scan_document(pages_data, model: str):
    findings = []
    seen = set()

    # Progress based on chunks, not pages
    all_chunks = []
    for p in pages_data:
        chunks = chunk_text_smart(p["text"])
        for c in chunks:
            all_chunks.append((p["page"], c))

    total = max(1, len(all_chunks))
    progress = st.progress(0)

    for idx, (page_num, chunk) in enumerate(all_chunks):
        progress.progress((idx + 1) / total)

        if len(chunk.strip()) < 20:
            continue

        # Pre-scan to reduce calls
        if not chunk_has_triggers(chunk):
            continue

        payload = call_model_parse(chunk, model=model)
        for item in payload.items:
            quote = item.trigger_text
            key = item.classification

            if not quote:
                continue

            # Trust anchor: must appear in the chunk verbatim-ish
            nq = normalize_text(quote)
            nc = normalize_text(chunk)
            if nq not in nc:
                continue

            unique_id = f"{page_num}|{key}|{nq}"
            if unique_id in seen:
                continue
            seen.add(unique_id)

            template = RESPONSE_TEMPLATES.get(key)
            if not template:
                continue

            findings.append({
                "phrase": quote,
                "category": template["category"],
                "gap": template["gap"],
                "question": template["question"],
                "page": page_num,
                "key": key
            })

    progress.empty()
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
                # Prefer extract_text; fallback to words
                text = page.extract_text() or ""
                text = text.strip()

                if len(text) < 50:
                    words = page.extract_words(x_tolerance=1)
                    text = " ".join([w["text"] for w in words])
                    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

                if text.strip():
                    pages_data.append({"page": i + 1, "text": text})
                    full_text_display += f"--- Page {i+1} ---\n{text}\n\n"

        st.text_area("Extracted Text Content", full_text_display, height=600)

with col2:
    st.subheader("2. Analysis")

    if "findings" not in st.session_state:
        st.session_state.findings = None

    # Model choice: reliable first
    model = st.selectbox(
        "Model",
        options=[
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
        ],
        index=0,
        help="4o is the most reliable/accurate. 4o-mini is cheaper/faster."
    )

    if pages_data:
        if st.button("Run Strict Analysis"):
            with st.spinner("Applying filters..."):
                st.session_state.findings = scan_document(pages_data, model=model)

        if st.session_state.findings is not None:
            results = st.session_state.findings
            st.info(f"**Scan complete.** Found {len(results)} items.")

            for item in results:
                with st.container():
                    st.markdown(f"### 🔹 {item['category']}")
                    st.caption(f"**Found:** \"{item['phrase']}\" [Page {item['page']}]")
                    st.markdown(f"**Gap:** {item['gap']}")
                    st.markdown(f"**Clarification:** {item['question']}")
                    st.divider()
    else:
        st.write("Upload a document to begin.")
