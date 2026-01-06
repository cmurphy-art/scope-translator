import streamlit as st
import pdfplumber
import json
import re
import time
from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(layout="wide", page_title="Scope Translator (OpenAI Hybrid - Canonical)")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("CRITICAL: OPENAI_API_KEY missing. Add it to Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

DEFAULT_MODEL = "gpt-4o"
CHEAP_MODEL = "gpt-4o-mini"

# ----------------------------
# SAFE TEMPLATES (neutral)
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
# CANONICALIZATION (the key fix)
# ----------------------------
def canon(s: str) -> str:
    """Lowercase and remove all non-alphanumerics. Makes glued PDFs matchable."""
    if not s:
        return ""
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def normalize_ws(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

# ----------------------------
# PDF extraction (kept simple; your PDF is already "glued" either way)
# ----------------------------
def extract_page_text(page) -> str:
    txt = (page.extract_text() or "").strip()
    if len(txt) < 50:
        words = page.extract_words(x_tolerance=1)
        txt = " ".join([w["text"] for w in words])
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

# ----------------------------
# Candidate finding (uses canonical matching so "ifpossibletoraiseit" hits)
# ----------------------------
# These are the TRIGGERS we try to detect in glued text
CANON_TRIGGERS = [
    "ifpossible", "wherepossible", "iffeasible",
    "asneeded", "asrequired", "asnecessary",
    "matchexisting", "tomatchexisting", "closelymatchexisting",
    "tieinto", "tiein",
    "patch", "patching", "repair",
    "turnkey", "completesystem", "includingbutnotlimitedto",
    "coordinatewith", "byothers", "verifyinfield",
    "liquidateddamages", "timeisoftheessence", "indemnify",
    "requiredupgrades", "bringtocode", "codeupgrades",
    "tbd"
]

def split_into_snippets(text: str, max_len: int = 260) -> list[str]:
    """
    Even with glued text, we can split on punctuation / numbering to get usable snippets.
    """
    if not text:
        return []
    parts = re.split(r"(?<=[\.\;\:\?\!])\s+|(?=\s\d+\.)", text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > max_len:
            p = p[:max_len].rstrip()
        out.append(p)
    return out

def find_candidate_snippets(page_text: str, max_candidates: int = 40) -> list[str]:
    snippets = split_into_snippets(page_text)
    hits = []

    for s in snippets:
        c = canon(s)
        if len(c) < 10:
            continue
        if any(t in c for t in CANON_TRIGGERS):
            hits.append(s)

    # dedupe preserve order
    seen = set()
    uniq = []
    for s in hits:
        k = canon(s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(s)

    return uniq[:max_candidates]

# ----------------------------
# OpenAI classifier (batched per page)
# ----------------------------
SYSTEM_INSTRUCTION = """
ROLE: You are a strict classifier.

You are given candidate snippets extracted from a construction scope document.
Return a JSON list of findings.

Each finding must contain:
- snippet_index: integer (which snippet it came from)
- classification: one of these KEYS
- trigger_text: an EXACT substring copied from the snippet (copy/paste exactly)

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
- Return ONLY valid JSON. No markdown.
- If nothing applies, return [].
- Do NOT invent text. trigger_text must appear in the snippet exactly.
"""

def safe_json_load(s: str):
    if not s:
        return []
    s = re.sub(r"```json\s*|\s*```", "", s).strip()
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return [data]
        return data if isinstance(data, list) else []
    except:
        return []

def call_classifier(model: str, snippets: list[str], max_retries: int = 3):
    payload = {"snippets": [{"i": i, "text": t} for i, t in enumerate(snippets)]}
    user_input = "CANDIDATE SNIPPETS (JSON):\n" + json.dumps(payload, ensure_ascii=False)

    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                instructions=SYSTEM_INSTRUCTION,
                input=user_input,
                max_output_tokens=900,
            )
            return safe_json_load(resp.output_text)
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate" in msg or "overload" in msg:
                time.sleep(2 ** attempt)
                continue
            raise
    return []

# ----------------------------
# Main scan
# ----------------------------
def scan_document(pages_data: list[dict], model_name: str):
    findings = []
    seen = set()

    progress = st.progress(0)
    total = len(pages_data)

    for idx, p in enumerate(pages_data):
        page_num = p["page"]
        page_text = p["text"]

        progress.progress((idx + 1) / max(total, 1))

        snippets = find_candidate_snippets(page_text)
        if not snippets:
            continue

        llm_items = call_classifier(model_name, snippets)

        for item in llm_items:
            key = (item.get("classification") or "").strip()
            trig = (item.get("trigger_text") or "").strip()
            snip_i = item.get("snippet_index")

            if key not in ALLOWED_KEYS:
                continue
            if trig == "" or snip_i is None:
                continue
            try:
                snip_i = int(snip_i)
            except:
                continue
            if not (0 <= snip_i < len(snippets)):
                continue

            snippet = snippets[snip_i]

            # Trust anchor: allow glued matching via canonical form
            if canon(trig) not in canon(snippet):
                continue

            uid = f"{page_num}|{canon(trig)}|{key}"
            if uid in seen:
                continue
            seen.add(uid)

            tpl = RESPONSE_TEMPLATES[key]
            findings.append({
                "category": tpl["category"],
                "phrase": trig,
                "gap": tpl["gap"],
                "question": tpl["question"],
                "page": page_num
            })

    progress.empty()
    findings.sort(key=lambda x: (x["page"], x["category"]))
    return findings

# ----------------------------
# UI
# ----------------------------
st.title("Scope Translator (OpenAI Hybrid - Canonical)")
st.markdown("**Ethos:** Move the burden from the person to the document.")
st.divider()

with st.sidebar:
    st.subheader("Model")
    model_choice = st.selectbox("Choose model", [DEFAULT_MODEL, CHEAP_MODEL], index=0)
    st.caption("Use gpt-4o for best accuracy. gpt-4o-mini for cheaper runs.")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("1. Source Document")
    uploaded = st.file_uploader("Upload Scope PDF", type="pdf")

    pages_data = []
    full_display = ""

    if uploaded is not None:
        with pdfplumber.open(uploaded) as pdf:
            for i, page in enumerate(pdf.pages):
                text = extract_page_text(page)
                if text:
                    pages_data.append({"page": i + 1, "text": text})
                    full_display += f"--- Page {i+1} ---\n{text}\n\n"

        st.text_area("Extracted Text Content", full_display, height=600)

with col2:
    st.subheader("2. Analysis")
    if "findings" not in st.session_state:
        st.session_state.findings = None

    if pages_data:
        if st.button("Run Analysis"):
            with st.spinner("Finding candidates and classifying..."):
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
