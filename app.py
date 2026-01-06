import streamlit as st
import pdfplumber
import json
import re
import time
import random
from openai import OpenAI

# ----------------------------
# CONFIGURATION
# ----------------------------
st.set_page_config(layout="wide", page_title="Scope Translator (OpenAI)")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("CRITICAL: OPENAI_API_KEY missing. Add it to Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Choose reliability first. You can later swap to a cheaper model.
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-5.2")

# ----------------------------
# SAFE LIBRARY (Templates)
# ----------------------------
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

# Optional: cheap pre-scan so you do not call the API for empty chunks
PRE_SCAN_PHRASES = [
    "match existing", "tie into", "patch", "repair",
    "industry standard", "workmanlike", "satisfaction of",
    "turnkey", "complete system", "including but not limited to",
    "liquidated damages", "time is of the essence", "indemnify",
    "coordinate with", "verify in field", "by others",
    "if possible", "where possible", "if feasible",
    "as needed", "as required", "as necessary",
    "required upgrades", "bring to code", "code upgrades"
]

SYSTEM_INSTRUCTION = """
ROLE: You are a strict Classifier. You DO NOT write text. You only select keys.

TASK: Analyze the text snippet. Identify ambiguities using the KEYS below.

KEYS (Select the best fit):
1. UNDEFINED_BOUNDARY (Triggers: "match existing", "tie into", "patch", "repair")
2. SUBJECTIVE_QUALITY (Triggers: "industry standard", "workmanlike", "satisfaction of")
3. UNDEFINED_SCOPE (Triggers: "turnkey", "complete system", "including but not limited to")
4. EXPLICIT_LIABILITY (Triggers: "liquidated damages", "time is of the essence", "indemnify")
5. COORDINATION_GAP (Triggers: "coordinate with", "verify in field", "by others")
6. IF_POSSIBLE (Triggers: "if possible", "where possible", "if feasible")
7. AS_NEEDED (Triggers: "as needed", "as required", "as necessary")
8. REQUIRED_UPGRADES (Triggers: "required upgrades", "bring to code", "code upgrades")

CONSTRAINTS:
- Return ONLY a raw JSON list.
- Extract the EXACT quote as it appears in the provided text.
- Do NOT provide reasoning. Only provide "classification" and "trigger_text".
- Return [] if nothing found.

OUTPUT FORMAT:
[{"trigger_text": "...", "classification": "UNDEFINED_BOUNDARY"}]
"""

# ----------------------------
# UTILITIES
# ----------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t

def clean_json_text(text: str) -> str:
    if not text:
        return "[]"
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

def split_text_into_chunks(text: str, max_chars: int = 2000):
    paragraphs = text.split("\n\n")
    chunks, cur = [], ""
    for para in paragraphs:
        if len(cur) + len(para) + 2 <= max_chars:
            cur += para + "\n\n"
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = para + "\n\n"
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def chunk_has_any_prescan_phrase(chunk: str) -> bool:
    c = normalize_text(chunk)
    for p in PRE_SCAN_PHRASES:
        if p in c:
            return True
    return False

def rebuild_spacing_from_words(page) -> str:
    """
    Rebuilds readable text using word positions.
    This fixes the "no spacing" output for many PDFs.
    """
    words = page.extract_words(
        x_tolerance=1,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=True
    )
    if not words:
        return ""

    # Sort by vertical then horizontal position
    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))

    lines = []
    current = []
    current_top = None
    prev_x1 = None

    for w in words:
        top = w["top"]
        text = w["text"]

        if current_top is None:
            current_top = top

        # New line if vertical jump is big
        if abs(top - current_top) > 5:
            if current:
                lines.append("".join(current).strip())
            current = [text]
            current_top = top
            prev_x1 = w["x1"]
            continue

        # Same line: insert a space if there is a visible gap
        if prev_x1 is not None:
            gap = w["x0"] - prev_x1
            if gap > 2.0:
                current.append(" ")
        current.append(text)
        prev_x1 = w["x1"]

    if current:
        lines.append("".join(current).strip())

    page_text = "\n".join([ln for ln in lines if ln.strip()])

    # Light cleanup for common stuck-together patterns
    page_text = re.sub(r"(?<=\w)(?=[A-Z])", " ", page_text)  # aB -> a B
    page_text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", page_text)  # 1Bathroom -> 1 Bathroom
    page_text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", page_text)  # Bath1 -> Bath 1
    page_text = re.sub(r"\s+", " ", page_text).replace(" \n", "\n")

    # Re-introduce line breaks at obvious section dividers if they got flattened
    page_text = page_text.replace(" __", "\n__")
    return page_text.strip()

def call_openai_classifier(text_chunk: str, max_retries: int = 4):
    """
    Responses API call with exponential backoff for 429 and transient failures.
    """
    prompt = f"{SYSTEM_INSTRUCTION}\n\nTEXT TO ANALYZE:\n{text_chunk}"

    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=prompt
            )
            raw = resp.output_text
            data = json.loads(clean_json_text(raw))
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                return []
            return data
        except Exception as e:
            msg = str(e).lower()
            transient = ("429" in msg) or ("rate" in msg) or ("timeout" in msg) or ("tempor" in msg) or ("overloaded" in msg)
            if not transient or attempt == max_retries - 1:
                return []
            # backoff: 1.5s, 3s, 6s, 12s plus jitter
            sleep_s = (1.5 * (2 ** attempt)) + random.uniform(0, 0.6)
            time.sleep(sleep_s)

    return []

def scan_document(pages_data):
    findings = []
    seen = set()

    total_chunks = sum(len(split_text_into_chunks(p["text"])) for p in pages_data)
    progress = st.progress(0)
    done = 0

    for page in pages_data:
        page_num = page["page"]
        raw_text = page["text"]
        chunks = split_text_into_chunks(raw_text, max_chars=2000)

        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < 20:
                done += 1
                progress.progress(min(done / max(total_chunks, 1), 1.0))
                continue

            # Optional cost-control: skip obviously irrelevant chunks
            if not chunk_has_any_prescan_phrase(chunk):
                done += 1
                progress.progress(min(done / max(total_chunks, 1), 1.0))
                continue

            data = call_openai_classifier(chunk)
            if data:
                for item in data:
                    key = item.get("classification", "")
                    quote = item.get("trigger_text", "")

                    if not key or not quote:
                        continue

                    # Trust anchor: quote must exist in chunk
                    if normalize_text(quote) not in normalize_text(chunk):
                        continue

                    uid = f"{page_num}|{normalize_text(quote)}|{key}"
                    if uid in seen:
                        continue

                    tpl = RESPONSE_TEMPLATES.get(key)
                    if not tpl:
                        continue

                    seen.add(uid)
                    findings.append({
                        "phrase": quote,
                        "category": tpl["category"],
                        "gap": tpl["gap"],
                        "question": tpl["question"],
                        "page": page_num
                    })

            done += 1
            progress.progress(min(done / max(total_chunks, 1), 1.0))

    progress.empty()
    return findings

# ----------------------------
# UI
# ----------------------------
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
                page_text = rebuild_spacing_from_words(page)
                if page_text:
                    pages_data.append({"page": i + 1, "text": page_text})
                    full_text_display += f"--- Page {i+1} ---\n{page_text}\n\n"

        st.text_area("Extracted Text Content", full_text_display, height=600)

with col2:
    st.subheader("2. Analysis")

    if "findings" not in st.session_state:
        st.session_state.findings = None

    if pages_data:
        if st.button("Run Strict Analysis"):
            with st.spinner("Applying filters..."):
                st.session_state.findings = scan_document(pages_data)

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
