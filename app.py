import streamlit as st
import pdfplumber
import json
import re
from openai import OpenAI

# =========================
# CONFIG
# =========================
st.set_page_config(layout="wide", page_title="Prephase Scope Auditor")

# --- OpenAI API Key ---
if "OPENAI_API_KEY" in st.secrets and st.secrets["OPENAI_API_KEY"].strip():
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
else:
    st.error("CRITICAL: OpenAI API key missing. Add OPENAI_API_KEY to Streamlit Secrets.")
    st.stop()

# =========================
# SAFE LIBRARY (Templates)
# =========================
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

# =========================
# BRAIN (Strict JSON Classifier)
# =========================
SYSTEM_INSTRUCTION = """
ROLE: You are a strict Classifier. You DO NOT write prose. You only output JSON.

TASK: Analyze the text snippet. Identify specific ambiguity triggers using the KEYS below.

KEYS (Select the best fit):
1. UNDEFINED_BOUNDARY (Triggers: "match existing", "tie into", "patch", "repair", "blend", "feather")
2. SUBJECTIVE_QUALITY (Triggers: "industry standard", "workmanlike", "proper", "optimize", "best practice", "satisfaction of")
3. UNDEFINED_SCOPE (Triggers: "turnkey", "complete system", "including but not limited to", "TBD", "allowance")
4. EXPLICIT_LIABILITY (Triggers: "liquidated damages", "time is of the essence", "indemnify", "penalty")
5. COORDINATION_GAP (Triggers: "coordinate with", "verify in field", "by others", "owner to provide")
6. IF_POSSIBLE (Triggers: "if possible", "where possible", "if feasible", "potentially", "possibly")
7. AS_NEEDED (Triggers: "as needed", "as required", "as necessary")
8. REQUIRED_UPGRADES (Triggers: "required upgrades", "bring to code", "code upgrades", "required by AHJ", "building department")

CONSTRAINTS:
- Return ONLY a raw JSON list.
- Each item must include an EXACT quote copied from the text (trigger_text).
- Do NOT add explanations.
- Do NOT return duplicates.
- Return [] if nothing found.

OUTPUT FORMAT:
[{"trigger_text":"...","classification":"UNDEFINED_BOUNDARY"}]
""".strip()

# =========================
# UTILITIES
# =========================
def normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())

def clean_json_text(text: str) -> str:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

def split_text_into_chunks(text: str, max_chars: int = 2000):
    # Chunk by sentences/paragraph-ish boundaries to reduce truncation
    parts = re.split(r"\n\s*\n", text)
    chunks, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        candidate = (buf + "\n\n" + p).strip() if buf else p
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            # If a single part is huge, hard-slice it
            if len(p) > max_chars:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i+max_chars])
                buf = ""
            else:
                buf = p
    if buf:
        chunks.append(buf)
    return chunks

def extract_pdf_pages(uploaded_file):
    pages_data = []
    full_text_display = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            # Extract words to reduce weird spacing issues
            words = page.extract_words(x_tolerance=1)
            page_text = " ".join([w["text"] for w in words])
            # Add space between camelCase-ish joins
            page_text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", page_text)
            # Add space after commas if missing
            page_text = re.sub(r",(?=\S)", ", ", page_text)

            if page_text and page_text.strip():
                pages_data.append({"page": i + 1, "text": page_text})
                full_text_display += f"--- Page {i+1} ---\n{page_text}\n\n"
    return pages_data, full_text_display

def openai_classify_chunk(model_name: str, chunk: str):
    # Strict instruction, but still guard parsing
    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"TEXT TO ANALYZE:\n{chunk}"}
        ],
    )
    raw = resp.output_text or ""
    raw = clean_json_text(raw)
    return json.loads(raw)

def scan_document(pages_data, model_name: str, enabled_keys: set[str]):
    findings = []
    seen = set()

    total_chunks = sum(len(split_text_into_chunks(p["text"])) for p in pages_data)
    done = 0
    progress = st.progress(0)

    for page_obj in pages_data:
        page_num = page_obj["page"]
        raw_text = page_obj["text"]

        for chunk in split_text_into_chunks(raw_text):
            if len(chunk.strip()) < 10:
                continue

            try:
                data = openai_classify_chunk(model_name, chunk)
            except Exception:
                # If parsing/model hiccups, skip this chunk quietly
                data = []

            if isinstance(data, list) and data:
                norm_chunk = normalize_text(chunk)

                for item in data:
                    key = (item or {}).get("classification")
                    quote = (item or {}).get("trigger_text")

                    if not key or not quote:
                        continue
                    if key not in RESPONSE_TEMPLATES:
                        continue
                    if key not in enabled_keys:
                        continue

                    norm_quote = normalize_text(quote)

                    # Trust anchor: quote must actually exist in chunk text
                    if norm_quote not in norm_chunk:
                        continue

                    # Dedup by page + quote + key
                    uid = f"{page_num}|{key}|{norm_quote}"
                    if uid in seen:
                        continue
                    seen.add(uid)

                    t = RESPONSE_TEMPLATES[key]
                    findings.append({
                        "phrase": quote,
                        "classification": key,
                        "category": t["category"],
                        "gap": t["gap"],
                        "question": t["question"],
                        "page": page_num,
                    })

            done += 1
            if total_chunks > 0:
                progress.progress(min(done / total_chunks, 1.0))

    progress.empty()
    return findings

def highlight_text(text: str, phrase: str):
    """
    Returns HTML-marked text with first match highlighted.
    Safe enough for your use case (scoped to extracted text), but still minimal.
    """
    if not text or not phrase:
        return text

    # Try exact first, then normalized fallback
    try_phrase = phrase.strip()
    idx = text.lower().find(try_phrase.lower())
    if idx >= 0:
        before = text[:idx]
        mid = text[idx:idx+len(try_phrase)]
        after = text[idx+len(try_phrase):]
        return (
            f"{escape_html(before)}"
            f"<mark style='background-color: #ffe66d; padding: 0 2px;'>{escape_html(mid)}</mark>"
            f"{escape_html(after)}"
        )

    return escape_html(text)

def escape_html(s: str):
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )

# =========================
# SESSION STATE
# =========================
if "findings" not in st.session_state:
    st.session_state.findings = []
if "selected_page" not in st.session_state:
    st.session_state.selected_page = 1
if "selected_phrase" not in st.session_state:
    st.session_state.selected_phrase = ""

# =========================
# UI
# =========================
st.title("Prephase Scope Auditor")
st.markdown("**Ethos:** Move the burden from the person to the document.")
st.divider()

col1, col2 = st.columns([1.6, 1])

# ---------- LEFT: Source + Extracted Text ----------
with col1:
    st.subheader("1. Source Document")
    uploaded_file = st.file_uploader("Upload Scope PDF", type="pdf")

    pages_data = []
    full_text_display = ""

    if uploaded_file is not None:
        pages_data, full_text_display = extract_pdf_pages(uploaded_file)

        if not pages_data:
            st.warning("No readable text was extracted. This PDF may be image-only (scanned) or drawing-heavy.")
        else:
            st.caption("Click a finding on the right to jump and highlight it here.")

            # Page control driven by state
            max_page = max(p["page"] for p in pages_data)
            st.session_state.selected_page = st.number_input(
                "Page",
                min_value=1,
                max_value=max_page,
                value=min(st.session_state.selected_page, max_page),
                step=1,
            )

            current_page_text = next((p["text"] for p in pages_data if p["page"] == st.session_state.selected_page), "")
            highlighted = highlight_text(current_page_text, st.session_state.selected_phrase)

            # Fixed-height viewer
            with st.container(height=560, border=True):
                st.markdown(
                    f"<div style='white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; font-size: 13px; line-height: 1.45;'>"
                    f"{highlighted}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ---------- RIGHT: Findings (fixed height + internal scroll) ----------
with col2:
    st.subheader("2. Findings")

    model_name = st.selectbox(
        "Model",
        options=[
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4.1",
        ],
        index=0,
        help="gpt-4o-mini is fast/cheap. gpt-4o is usually stronger.",
    )

    # Simple UI checkbox filter (optional but UI-only)
    st.markdown("**Include categories:**")
    default_on = {"IF_POSSIBLE", "AS_NEEDED", "UNDEFINED_BOUNDARY", "REQUIRED_UPGRADES", "COORDINATION_GAP", "UNDEFINED_SCOPE", "SUBJECTIVE_QUALITY", "EXPLICIT_LIABILITY"}
    c1, c2 = st.columns(2)
    with c1:
        show_boundary = st.checkbox("Match / Tie-in / Patch", value=True)
        show_conditionals = st.checkbox("Conditional language", value=True)
        show_as_needed = st.checkbox("As needed / required", value=True)
        show_upgrades = st.checkbox("Required upgrades", value=True)
    with c2:
        show_coord = st.checkbox("Coordination gaps", value=True)
        show_scope = st.checkbox("Undefined scope / TBD", value=True)
        show_quality = st.checkbox("Subjective quality", value=True)
        show_liability = st.checkbox("Explicit liability", value=True)

    enabled_keys = set()
    if show_boundary:
        enabled_keys.add("UNDEFINED_BOUNDARY")
    if show_conditionals:
        enabled_keys.add("IF_POSSIBLE")
    if show_as_needed:
        enabled_keys.add("AS_NEEDED")
    if show_upgrades:
        enabled_keys.add("REQUIRED_UPGRADES")
    if show_coord:
        enabled_keys.add("COORDINATION_GAP")
    if show_scope:
        enabled_keys.add("UNDEFINED_SCOPE")
    if show_quality:
        enabled_keys.add("SUBJECTIVE_QUALITY")
    if show_liability:
        enabled_keys.add("EXPLICIT_LIABILITY")

    run = st.button("Run Analysis", type="primary", use_container_width=True, disabled=(not pages_data))

    if run and pages_data:
        with st.spinner("Applying filters..."):
            st.session_state.findings = scan_document(pages_data, model_name=model_name, enabled_keys=enabled_keys)

    if st.session_state.findings:
        results = st.session_state.findings
        st.info(f"Scan complete. Found {len(results)} items.")

        # Fixed-height scroll panel for findings
        with st.container(height=560, border=True):
            for idx, item in enumerate(results):
                # One-click jump + highlight
                def _select(i=item):
                    st.session_state.selected_page = int(i["page"])
                    st.session_state.selected_phrase = i["phrase"]

                header = f'{item["category"]} | Page {item["page"]} | "{item["phrase"]}"'
                st.button(
                    header,
                    key=f"jump_{idx}",
                    on_click=_select,
                    use_container_width=True
                )

                st.markdown(f"**Gap:** {item['gap']}")
                st.markdown(f"**Clarification:** {item['question']}")
                st.divider()
    else:
        st.caption("Upload a document to begin.")
