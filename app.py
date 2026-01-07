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

# --- USER INTERFACE ---
import html

def highlight_text(full_text: str, phrase: str) -> str:
    """
    Returns HTML where every case-insensitive occurrence of `phrase` is wrapped in <mark>.
    Safely escapes other text.
    """
    if not full_text:
        return ""

    if not phrase or len(phrase.strip()) < 2:
        return "<pre style='white-space: pre-wrap;'>" + html.escape(full_text) + "</pre>"

    # Case-insensitive find all occurrences
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    matches = list(pattern.finditer(full_text))
    if not matches:
        return "<pre style='white-space: pre-wrap;'>" + html.escape(full_text) + "</pre>"

    out = []
    last = 0
    for m in matches:
        out.append(html.escape(full_text[last:m.start()]))
        out.append("<mark>" + html.escape(full_text[m.start():m.end()]) + "</mark>")
        last = m.end()
    out.append(html.escape(full_text[last:]))

    return "<pre style='white-space: pre-wrap;'>" + "".join(out) + "</pre>"


st.title("Scope Translator")
st.markdown("**Ethos:** Move the burden from the person to the document.")
st.divider()

# Session state for click-to-jump
if "findings" not in st.session_state:
    st.session_state.findings = None
if "selected_page" not in st.session_state:
    st.session_state.selected_page = 1
if "selected_phrase" not in st.session_state:
    st.session_state.selected_phrase = ""

# Simple CSS to create fixed-height, scrollable panes
st.markdown(
    """
    <style>
      .scroll-pane {
        height: 78vh;
        overflow-y: auto;
        padding-right: 8px;
        border: 1px solid rgba(49,51,63,0.2);
        border-radius: 8px;
        padding: 10px;
        background: rgba(255,255,255,0.02);
      }
      mark {
        padding: 0px 3px;
        border-radius: 4px;
      }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1.6, 1.0])

# We will keep a page lookup for jump-to
pages_data = []
page_text_by_num = {}

with col1:
    st.subheader("1. Source Document")
    uploaded_file = st.file_uploader("Upload Scope PDF", type="pdf")

    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                # Keep your existing extraction approach
                words = page.extract_words(x_tolerance=1)
                page_text = " ".join([w["text"] for w in words])
                page_text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", page_text)

                page_num = i + 1
                if page_text:
                    pages_data.append({"page": page_num, "text": page_text})
                    page_text_by_num[page_num] = page_text

        if pages_data:
            max_page = max(page_text_by_num.keys())
            st.session_state.selected_page = min(st.session_state.selected_page, max_page)

            st.caption("Click a finding on the right to jump and highlight it here.")

            # Page selector stays usable even without clicking
            st.session_state.selected_page = st.number_input(
                "Page",
                min_value=1,
                max_value=max_page,
                value=int(st.session_state.selected_page),
                step=1
            )

            selected_text = page_text_by_num.get(int(st.session_state.selected_page), "")

            # Render page text with highlight
            rendered = highlight_text(selected_text, st.session_state.selected_phrase)

            st.markdown(
                f"<div class='scroll-pane'>{rendered}</div>",
                unsafe_allow_html=True
            )
        else:
            st.warning("No text extracted from this PDF.")

with col2:
    st.subheader("2. Findings")

    if pages_data:
        # Run scan
        if st.button("Run Analysis"):
            with st.spinner("Scanning..."):
                # IMPORTANT: uses your existing scan_document unchanged
                st.session_state.findings = scan_document(pages_data)

        results = st.session_state.findings

        if results:
            st.info(f"**Scan complete.** Found {len(results)} items.")

            # Scrollable findings list
            st.markdown("<div class='scroll-pane'>", unsafe_allow_html=True)

            for idx, item in enumerate(results):
                cat = item.get("category") or item.get("type") or "Finding"
                phrase = item.get("phrase") or item.get("quote") or ""
                page = int(item.get("page", 1))
                gap = item.get("gap", "")
                question = item.get("question", "")

                # One click jumps the left pane to the right page + highlights phrase
                if st.button(f"Go to Page {page}", key=f"go_{idx}_{page}"):
                    st.session_state.selected_page = page
                    st.session_state.selected_phrase = phrase
                    st.rerun()

                st.markdown(f"### 🔹 {cat}")
                st.caption(f'**Found:** "{phrase}" [Page {page}]')
                if gap:
                    st.markdown(f"**Gap:** {gap}")
                if question:
                    st.markdown(f"**Clarification:** {question}")
                st.divider()

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.write("Run the analysis to see findings.")
    else:
        st.write("Upload a document to begin.")
