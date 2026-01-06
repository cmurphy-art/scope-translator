import streamlit as st
import pdfplumber
import json
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any

from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(layout="wide", page_title="Scope Translator (OpenAI Hybrid)")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("CRITICAL: OPENAI_API_KEY missing. Add it to Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

DEFAULT_MODEL = "gpt-4o"       # most reliable/accurate
CHEAP_MODEL = "gpt-4o-mini"    # cheaper, still solid

# ----------------------------
# SAFE RESPONSE TEMPLATES (Neutral, document-centered)
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

ALLOWED_KEYS = set(RESPONSE_TEMPLATES.keys())

# ----------------------------
# TEXT NORMALIZATION & EXTRACTION
# ----------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_page_text(page) -> str:
    """
    Practical extraction:
    1) Try extract_text() (keeps some structure when possible)
    2) Fallback to extract_words() (your "nuclear" option)
    3) Fix glued words in a conservative way
    """
    txt = page.extract_text() or ""
    txt = txt.strip()

    if len(txt) < 50:
        words = page.extract_words(x_tolerance=1)
        txt = " ".join([w["text"] for w in words])

    # Light de-glue for camelcase artifacts from word-join
    txt = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", txt)

    # Normalize whitespace
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

# ----------------------------
# CANDIDATE SNIPPET FINDER (broad, not brittle)
# ----------------------------
CANDIDATE_PATTERNS = [
    # conditional / discretion
    r"\b(if possible|where possible|if feasible)\b",
    # "as needed" family
    r"\b(as needed|as required|as necessary)\b",
    # match/tie-in boundary
    r"\b(match existing|to match existing|closely match existing|tie into|tie-in|patch(ing)?|repair)\b",
    # open-ended scope
    r"\b(turnkey|complete system|including but not limited to)\b",
    # coordination / handoff ambiguity
    r"\b(coordinate with|by others|verify in field)\b",
    # explicit contractual triggers
    r"\b(liquidated damages|time is of the essence|indemnify)\b",
    # upgrades / code language
    r"\b(required upgrades?|bring to code|code upgrades?)\b",
    # placeholders / undefined scope markers
    r"\b(TBD|T\.B\.D\.)\b",
]

# Split into “snippets” without needing original PDF line breaks
def split_into_snippets(text: str, max_len: int = 220) -> List[str]:
    """
    We split on punctuation and bullets-ish separators to get clause-sized text.
    """
    if not text:
        return []

    rough = re.split(r"(?<=[\.\;\:\?\!])\s+|(?=\s-\s)|(?=\s•\s)|(?=\s\d+\.)", text)
    snippets = []
    for s in rough:
        s = s.strip()
        if not s:
            continue
        # cap length (keep front portion; enough to contain the trigger)
        if len(s) > max_len:
            s = s[:max_len].rstrip()
        snippets.append(s)
    return snippets

def find_candidate_snippets(page_text: str, max_candidates: int = 30) -> List[str]:
    snippets = split_into_snippets(page_text)
    hits = []
    for s in snippets:
        s_norm = normalize_text(s)
        if len(s_norm) < 8:
            continue
        for pat in CANDIDATE_PATTERNS:
            if re.search(pat, s_norm, flags=re.IGNORECASE):
                hits.append(s)
                break

    # dedupe while preserving order
    seen = set()
    uniq = []
    for s in hits:
        key = normalize_text(s)
        if key in seen:
            continue
        uniq.append(s)
        seen.add(key)

    return uniq[:max_candidates]

# ----------------------------
# LLM CLASSIFIER (batched per page)
# ----------------------------
SYSTEM_INSTRUCTION = """
ROLE: You are a strict classifier.

TASK:
You will be given a list of candidate snippets from a construction scope document.
Return a JSON list of findings. Each finding must:
- Choose exactly ONE classification key from the KEYS list.
- Provide trigger_text that is an EXACT substring from the snippet (copy/paste).
- Provide snippet_index (integer) to indicate which snippet it came from.

KEYS:
- UNDEFINED_BOUNDARY
- SUBJECTIVE_QUALITY
- UNDEFINED_SCOPE
- EXPLICIT_LIABILITY
- COORDINATION_GAP
- IF_POSSIBLE
- AS_NEEDED
- REQUIRED_UPGRADES

CONSTRAINTS:
- Return ONLY valid JSON (no markdown).
- If nothing applies, return [].
- Do NOT invent text that isn't in the snippet.
"""

def safe_json_load(text: str) -> List[Dict[str, Any]]:
    try:
        return json.loads(text)
    except:
        # attempt to strip ``` wrappers if present
        cleaned = re.sub(r"```json\s*|\s*```", "", text).strip()
        try:
            return json.loads(cleaned)
        except:
            return []

def call_openai_classifier(model: str, snippets: List[str], max_retries: int = 3) -> List[Dict[str, Any]]:
    if not snippets:
        return []

    payload = {
        "snippets": [{"i": i, "text": s} for i, s in enumerate(snippets)]
    }

    user_input = (
        "CANDIDATE SNIPPETS (JSON):\n"
        + json.dumps(payload, ensure_ascii=False)
        + "\n\nReturn findings now."
    )

    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                instructions=SYSTEM_INSTRUCTION,
                input=user_input,
                # keep output tight
                max_output_tokens=800,
            )
            data = safe_json_load(resp.output_text)
            if isinstance(data, dict):
                data = [data]
            return data if isinstance(data, list) else []
        except Exception as e:
            msg = str(e)
            # simple exponential backoff on rate/overload
            if "429" in msg or "rate" in msg.lower() or "overload" in msg.lower():
                time.sleep(2 ** attempt)
                continue
            # otherwise: fail fast for visibility
            raise

    return []

# ----------------------------
# MAIN SCAN
# ----------------------------
def scan_document(pages_data: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    findings = []
    seen = set()

    progress_bar = st.progress(0)
    total_pages = len(pages_data)

    for idx, page_obj in enumerate(pages_data):
        page_num = page_obj["page"]
        page_text = page_obj["text"]

        progress_bar.progress((idx + 1) / max(total_pages, 1))

        candidates = find_candidate_snippets(page_text, max_candidates=35)

        # If nothing even looks ambiguous, skip API entirely
        if not candidates:
            continue

        llm_items = call_openai_classifier(model_name, candidates)

        for item in llm_items:
            key = (item.get("classification") or "").strip()
            trig = (item.get("trigger_text") or "").strip()
            snip_i = item.get("snippet_index")

            if key not in ALLOWED_KEYS:
                continue
            if trig == "" or snip_i is None:
                continue
            if not (0 <= int(snip_i) < len(candidates)):
                continue

            snippet = candidates[int(snip_i)]

            # Trust anchor: must be exact substring of snippet
            if normalize_text(trig) not in normalize_text(snippet):
                continue

            # Dedup: page + normalized trigger + key
            uid = f"{page_num}|{normalize_text(trig)}|{key}"
            if uid in seen:
                continue
            seen.add(uid)

            tpl = RESPONSE_TEMPLATES[key]
            findings.append({
                "category": tpl["category"],
                "phrase": trig,
                "gap": tpl["gap"],
                "question": tpl["question"],
                "page": page_num,
            })

    progress_bar.empty()

    # sort by page, then category for readability
    findings.sort(key=lambda x: (x["page"], x["category"]))
    return findings

# ----------------------------
# UI
# ----------------------------
st.title("Scope Translator (OpenAI Hybrid)")
st.markdown("**Ethos:** Move the burden from the person to the document.")
st.divider()

with st.sidebar:
    st.subheader("Model")
    model_choice = st.selectbox(
        "Choose model",
        options=[DEFAULT_MODEL, CHEAP_MODEL],
        index=0
    )
    st.caption("Use gpt-4o for best accuracy. Use gpt-4o-mini for cheaper runs.")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("1. Source Document")
    uploaded_file = st.file_uploader("Upload Scope PDF", type="pdf")

    pages_data = []
    full_text_display = ""

    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = extract_page_text(page)
                if page_text:
                    pages_data.append({"page": i + 1, "text": page_text})
                    full_text_display += f"--- Page {i+1} ---\n{page_text}\n\n"

        st.text_area("Extracted Text Content", full_text_display, height=600)

with col2:
    st.subheader("2. Analysis")

    if "findings" not in st.session_state:
        st.session_state.findings = None

    if pages_data:
        if st.button("Run Analysis"):
            with st.spinner("Applying filters and classifying candidates..."):
                st.session_state.findings = scan_document(pages_data, model_choice)

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
