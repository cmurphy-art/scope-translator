import streamlit as st
import pdfplumber
import json
import re
import time
import random
from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(layout="wide", page_title="Scope Translator (OpenAI Hybrid - Canonical)")

# Secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("CRITICAL: OPENAI_API_KEY missing. Add it to Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Model choice
MODEL_OPTIONS = [
    "gpt-4o-mini",   # fastest/cheapest, very good for this task
    "gpt-4o",        # more accurate, higher cost
]
DEFAULT_MODEL = "gpt-4o-mini"

# -----------------------------
# SAFE RESPONSE TEMPLATES (deterministic voice)
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
        "question": "Does the document account for excusable delays or reciprocal conditions?"
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

ALLOWED_KEYS = set(RESPONSE_TEMPLATES.keys())

# Deterministic pre-scan triggers (include collapsed-spacing variants)
TRIGGERS = {
    "UNDEFINED_BOUNDARY": [
        "match existing", "matchexisting",
        "tie into", "tieinto",
        "patch", "patching",
        "repair", "repairs",
    ],
    "SUBJECTIVE_QUALITY": [
        "industry standard", "industrystandard",
        "workmanlike", "satisfaction of", "satisfactionof",
    ],
    "UNDEFINED_SCOPE": [
        "turnkey", "complete system", "completesystem",
        "including but not limited to", "includingbutnotlimitedto",
    ],
    "EXPLICIT_LIABILITY": [
        "liquidated damages", "liquidateddamages",
        "time is of the essence", "timeisoftheessence",
        "indemnify", "indemnification",
    ],
    "COORDINATION_GAP": [
        "coordinate with", "coordinatewith",
        "verify in field", "verifyinfield",
        "by others", "byothers",
    ],
    "IF_POSSIBLE": [
        "if possible", "ifpossible",
        "where possible", "wherepossible",
        "if feasible", "iffeasible",
    ],
    "AS_NEEDED": [
        "as needed", "asneeded",
        "as required", "asrequired",
        "as necessary", "asnecessary",
    ],
    "REQUIRED_UPGRADES": [
        "required upgrades", "requiredupgrades",
        "bring to code", "bringtocode",
        "code upgrades", "codeupgrades",
        "upgrade", "upgrades"
    ]
}

# -----------------------------
# TEXT UTILITIES
# -----------------------------
def clean_json_text(text: str) -> str:
    text = re.sub(r"```json\s*", "", text or "")
    text = re.sub(r"```", "", text)
    return text.strip()

def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def nospace(s: str) -> str:
    return re.sub(r"\s+", "", norm_spaces(s))

def repair_spacing(s: str) -> str:
    """
    Attempts to fix collapsed PDF text like:
    'December6,2024Preparedfor:VivianKwok...'
    """
    if not s:
        return ""

    # Insert space between letter<->digit boundaries
    s = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", s)

    # Insert space after punctuation when followed by a letter
    s = re.sub(r"([,;:])([A-Za-z])", r"\1 \2", s)

    # Insert space between lower->Upper (camelCase-ish)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_text_into_chunks(text: str, max_chars: int = 2200):
    text = text.strip()
    if not text:
        return []
    # Prefer splitting on numbered items / sentences, but keep it simple + stable
    parts = re.split(r"(\n\n+)", text)
    chunks = []
    cur = ""
    for p in parts:
        if len(cur) + len(p) <= max_chars:
            cur += p
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def chunk_has_any_trigger(chunk: str) -> bool:
    c_space = norm_spaces(chunk)
    c_nospace = nospace(chunk)
    for key, variants in TRIGGERS.items():
        for t in variants:
            if " " in t:
                if norm_spaces(t) in c_space:
                    return True
            else:
                if t in c_nospace:
                    return True
    return False

# -----------------------------
# LLM CALL (with retry/backoff)
# -----------------------------
SYSTEM_INSTRUCTION = """
ROLE: You are a strict classifier.
You DO NOT write explanations. You only return JSON.

TASK:
Scan the provided TEXT for phrases that match the trigger families below.
Return a JSON LIST of objects with:
- "trigger_text": the exact phrase as it appears in the provided TEXT (copy/paste exact characters)
- "classification": one of the KEYS below

KEYS:
UNDEFINED_BOUNDARY (match existing, tie into, patch, repair)
SUBJECTIVE_QUALITY (industry standard, workmanlike, satisfaction of)
UNDEFINED_SCOPE (turnkey, complete system, including but not limited to)
EXPLICIT_LIABILITY (liquidated damages, time is of the essence, indemnify)
COORDINATION_GAP (coordinate with, verify in field, by others)
IF_POSSIBLE (if possible, where possible, if feasible)
AS_NEEDED (as needed, as required, as necessary)
REQUIRED_UPGRADES (required upgrades, bring to code, code upgrades, upgrade)

CONSTRAINTS:
- Return ONLY a raw JSON list. No markdown. No extra keys.
- If nothing found, return [].
"""

def call_llm_for_chunk(model_name: str, chunk: str):
    """
    Returns list of {"trigger_text": "...", "classification": "..."} or []
    """
    # Try a few times with exponential backoff on 429 / transient errors
    attempts = 5
    base_sleep = 0.8

    for a in range(attempts):
        try:
            resp = client.responses.create(
                model=model_name,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": SYSTEM_INSTRUCTION}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": f"TEXT:\n{chunk}"}],
                    },
                ],
                temperature=0,
                max_output_tokens=600,
            )
            text = resp.output_text
            data = json.loads(clean_json_text(text))
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                return []
            return data

        except Exception as e:
            msg = str(e)

            # Backoff on 429 / rate limits / overloaded
            is_rate = ("429" in msg) or ("Rate limit" in msg) or ("rate_limit" in msg)
            is_transient = ("502" in msg) or ("503" in msg) or ("504" in msg) or ("timeout" in msg.lower())

            if a < attempts - 1 and (is_rate or is_transient):
                sleep_s = base_sleep * (2 ** a) + random.uniform(0, 0.25)
                time.sleep(sleep_s)
                continue

            # For non-transient errors, surface it
            raise

    return []

# -----------------------------
# SCAN ENGINE
# -----------------------------
def scan_document(pages_data, model_name: str):
    findings = []
    seen = set()

    # Progress based on estimated total chunks
    all_chunks = []
    for p in pages_data:
        fixed = repair_spacing(p["text"])
        chunks = split_text_into_chunks(fixed)
        all_chunks.extend([(p["page"], c) for c in chunks])
    total = max(len(all_chunks), 1)

    progress = st.progress(0.0)
    processed = 0

    for page_num, chunk in all_chunks:
        processed += 1
        progress.progress(min(processed / total, 1.0))

        if len(chunk) < 20:
            continue

        # Deterministic pre-scan (don’t waste model calls)
        if not chunk_has_any_trigger(chunk):
            continue

        try:
            data = call_llm_for_chunk(model_name, chunk)
        except Exception:
            # Fail soft: don’t kill the run
            continue

        if not data:
            continue

        for item in data:
            key = (item or {}).get("classification", "")
            quote = (item or {}).get("trigger_text", "")

            if key not in ALLOWED_KEYS or not quote:
                continue

            # Validation that survives collapsed spacing:
            # compare nospace(quote) in nospace(chunk)
            if nospace(quote) not in nospace(chunk):
                continue

            unique_id = f"{page_num}|{key}|{nospace(quote)}"
            if unique_id in seen:
                continue
            seen.add(unique_id)

            t = RESPONSE_TEMPLATES[key]
            findings.append(
                {
                    "category": t["category"],
                    "gap": t["gap"],
                    "question": t["question"],
                    "phrase": quote,
                    "page": page_num,
                }
            )

    progress.empty()

    # Sort: page then category (stable + readable)
    findings.sort(key=lambda x: (x["page"], x["category"], nospace(x["phrase"])))
    return findings

# -----------------------------
# UI
# -----------------------------
st.title("Scope Translator (OpenAI Hybrid - Canonical)")
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
                # Prefer extract_text first (keeps natural spacing when possible)
                text = page.extract_text() or ""

                # Fallback: words
                if len(text.strip()) < 50:
                    words = page.extract_words(x_tolerance=2, y_tolerance=2)
                    text = " ".join([w.get("text", "") for w in words])

                text = text.strip()
                if text:
                    pages_data.append({"page": i + 1, "text": text})
                    fixed = repair_spacing(text)
                    full_text_display += f"--- Page {i+1} ---\n{fixed}\n\n"

        st.text_area("Extracted Text (Repaired for Spacing)", full_text_display, height=600)

with col2:
    st.subheader("2. Analysis")

    model_name = st.selectbox("Model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_MODEL))

    if "findings" not in st.session_state:
        st.session_state.findings = None

    if pages_data:
        if st.button("Run Analysis"):
            with st.spinner("Applying filters..."):
                st.session_state.findings = scan_document(pages_data, model_name)

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
