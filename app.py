import streamlit as st
import pdfplumber
import json
import re
import html
from openai import OpenAI

# ------------------------------------------------------------
# CONFIG (DO NOT CHANGE API PROVIDER)
# ------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Prephase Scope Auditor")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("CRITICAL: Missing OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------------------------------------------------
# SAFE LIBRARY (BRAIN OUTPUT STYLE UNCHANGED)
# ------------------------------------------------------------
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

SYSTEM_INSTRUCTION = """
ROLE: You are a strict Classifier. You DO NOT write prose. You only select keys.

TASK: Analyze the text snippet. Identify specific ambiguities using the KEYS below.

KEYS:
1. UNDEFINED_BOUNDARY (Triggers: "match existing", "tie into", "patch", "repair")
2. SUBJECTIVE_QUALITY (Triggers: "industry standard", "workmanlike", "satisfaction of")
3. UNDEFINED_SCOPE (Triggers: "turnkey", "complete system", "including but not limited to")
4. EXPLICIT_LIABILITY (Triggers: "liquidated damages", "time is of the essence", "indemnify")
5. COORDINATION_GAP (Triggers: "coordinate with", "verify in field", "by others")
6. IF_POSSIBLE (Triggers: "if possible", "where possible", "if feasible", "possibly", "potentially")
7. AS_NEEDED (Triggers: "as needed", "as required", "as necessary", "if needed")
8. REQUIRED_UPGRADES (Triggers: "required upgrades", "bring to code", "code upgrades", "upgrade")

CONSTRAINTS:
- Return ONLY a raw JSON list.
- Extract the EXACT quote from the provided text.
- Do NOT provide reasoning. Only provide the "classification" KEY.
- Return [] if nothing found.

OUTPUT FORMAT:
[{"trigger_text": "...", "classification": "UNDEFINED_BOUNDARY"}]
"""

# ------------------------------------------------------------
# TEXT HELPERS
# ------------------------------------------------------------
def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t

def spacing_fix(t: str) -> str:
    """
    Best-effort spacing repair for pdfplumber word-joins.
    Conservative: only adds spaces in obvious joins.
    """
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t)

    # lowerCaseUpperCase
    t = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t)
    # letterDigit and digitLetter
    t = re.sub(r"(?<=[A-Za-z])(?=[0-9])", " ", t)
    t = re.sub(r"(?<=[0-9])(?=[A-Za-z])", " ", t)
    # punctuationNextLetter
    t = re.sub(r"(?<=[,.;:])(?=[A-Za-z])", " ", t)
    # periodNextLetter
    t = re.sub(r"(?<=[.])(?=[A-Za-z])", " ", t)

    return t.strip()

def split_text_into_chunks(text: str, max_chars: int = 2200) -> list[str]:
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
        add = p + "\n\n"
        if len(cur) + len(add) <= max_chars:
            cur += add
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = add
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

# ------------------------------------------------------------
# LLM CALL (BRAIN UNCHANGED; PROVIDER = OPENAI)
# ------------------------------------------------------------
def classify_chunk(model_name: str, chunk: str) -> list[dict]:
    prompt = f"{SYSTEM_INSTRUCTION}\n\nTEXT TO ANALYZE:\n{chunk}"
    resp = client.responses.create(
        model=model_name,
        input=prompt,
        temperature=0
    )
    raw = (resp.output_text or "").strip()

    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```", "", raw).strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []

def scan_document(pages_data: list[dict], model_name: str, include_keys: set[str]) -> list[dict]:
    findings = []
    seen = set()

    progress = st.progress(0)
    total_pages = len(pages_data)

    for i, p in enumerate(pages_data):
        page_num = p["page"]
        page_text = p["text"]

        chunks = split_text_into_chunks(page_text)
        for chunk in chunks:
            if len(chunk) < 10:
                continue

            data = classify_chunk(model_name, chunk)

            for item in data:
                key = (item.get("classification") or "").strip()
                quote = (item.get("trigger_text") or "").strip()

                if not key or not quote:
                    continue
                if key not in RESPONSE_TEMPLATES:
                    continue
                if key not in include_keys:
                    continue

                # Trust anchor: quote must exist in chunk
                if normalize_text(quote) not in normalize_text(chunk):
                    continue

                uid = f"{page_num}|{normalize_text(quote)}|{key}"
                if uid in seen:
                    continue
                seen.add(uid)

                t = RESPONSE_TEMPLATES[key]
                findings.append({
                    "uid": uid,
                    "phrase": quote,
                    "classification": key,
                    "category": t["category"],
                    "gap": t["gap"],
                    "question": t["question"],
                    "page": page_num
                })

        progress.progress((i + 1) / max(total_pages, 1))

    progress.empty()
    findings.sort(key=lambda x: (x["page"], x["category"], len(x["phrase"])))
    return findings

# ------------------------------------------------------------
# UI RENDER: FULL DOCUMENT SCROLL VIEWER + HIGHLIGHT + JUMP
# ------------------------------------------------------------
def build_full_document_text(pages_text_by_num: dict[int, str]) -> str:
    parts = []
    for p in sorted(pages_text_by_num.keys()):
        parts.append(f"--- Page {p} ---\n{pages_text_by_num[p]}\n")
    return "\n".join(parts).strip()

def render_document_with_highlight(full_text: str, selected_page: int | None, highlight_phrase: str | None):
    """
    Scrollable document viewer. If highlight_phrase exists, highlight and scroll to it.
    Also adds anchors per page so we can at least jump to the page.
    """
    safe_text = html.escape(full_text)

    # Add page anchors for scrolling
    # Replace "--- Page X ---" headings with anchored headings
    def add_page_anchor(match):
        page_num = match.group(1)
        heading = match.group(0)
        return f'<div id="page_{page_num}"></div>{heading}'
    safe_text = re.sub(r"--- Page (\d+) ---", add_page_anchor, safe_text)

    target_id = None

    # Highlight phrase if provided
    if highlight_phrase:
        phrase_esc = html.escape(highlight_phrase)
        pattern = re.compile(re.escape(phrase_esc), re.IGNORECASE)

        # If we know selected_page, try to highlight within that page block first
        if selected_page is not None:
            # Find the page section boundaries in escaped text
            # We search for the escaped marker "--- Page N ---"
            page_marker = f"--- Page {selected_page} ---"
            page_marker_esc = html.escape(page_marker)

            start_idx = safe_text.find(page_marker_esc)
            if start_idx != -1:
                # End at next page marker, or end of document
                next_marker = re.search(r"--- Page \d+ ---", html.unescape(safe_text[start_idx + 1:]))
                # If we can’t cleanly compute end via unescape, we just do a best-effort search:
                # Find the next marker in the escaped text by searching for "--- Page " after start.
                next_idx = safe_text.find(html.escape("--- Page "), start_idx + len(page_marker_esc))
                end_idx = next_idx if next_idx != -1 else len(safe_text)

                page_slice = safe_text[start_idx:end_idx]
                m = pattern.search(page_slice)
                if m:
                    a, b = m.span()
                    page_slice = (
                        page_slice[:a]
                        + '<mark id="hit" style="background:#f2d34f;padding:0 2px;border-radius:3px;">'
                        + page_slice[a:b]
                        + "</mark>"
                        + page_slice[b:]
                    )
                    safe_text = safe_text[:start_idx] + page_slice + safe_text[end_idx:]
                    target_id = "hit"

        # If we didn’t find within the page, try first occurrence in entire doc
        if target_id is None:
            m = pattern.search(safe_text)
            if m:
                a, b = m.span()
                safe_text = (
                    safe_text[:a]
                    + '<mark id="hit" style="background:#f2d34f;padding:0 2px;border-radius:3px;">'
                    + safe_text[a:b]
                    + "</mark>"
                    + safe_text[b:]
                )
                target_id = "hit"

    # Decide fallback jump target
    if target_id is None and selected_page is not None:
        target_id = f"page_{selected_page}"

    html_block = f"""
    <div id="docBox" style="
        height: 560px;
        overflow-y: auto;
        padding: 14px;
        border: 1px solid rgba(0,0,0,0.10);
        border-radius: 10px;
        background: rgba(255,255,255,0.02);
        white-space: pre-wrap;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.45;">
        {safe_text}
    </div>

    <script>
      const box = document.getElementById("docBox");
      const target = document.getElementById("{target_id}" ) || null;

      if (box && target) {{
        // scrollIntoView inside the box: compute relative offset
        const top = target.offsetTop - 120;
        box.scrollTo({{ top: top, behavior: "smooth" }});
      }}
    </script>
    """
    st.components.v1.html(html_block, height=600)

# ------------------------------------------------------------
# APP
# ------------------------------------------------------------
st.title("Prephase Scope Auditor")
st.markdown("**Ethos:** Move the burden from the person to the document.")
st.divider()

# Session state
if "findings" not in st.session_state:
    st.session_state.findings = None
if "selected_uid" not in st.session_state:
    st.session_state.selected_uid = None
if "selected_page" not in st.session_state:
    st.session_state.selected_page = None
if "selected_phrase" not in st.session_state:
    st.session_state.selected_phrase = None
if "checked_map" not in st.session_state:
    # uid -> bool
    st.session_state.checked_map = {}

col1, col2 = st.columns([1.6, 1.0])

with col1:
    st.subheader("1. Source Document")
    uploaded_file = st.file_uploader("Upload Scope PDF", type="pdf")

    pages_data = []
    pages_text_by_num = {}

    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                words = page.extract_words(x_tolerance=1)
                text = " ".join([w["text"] for w in words])
                text = spacing_fix(text)

                page_num = i + 1
                if text.strip():
                    pages_data.append({"page": page_num, "text": text})
                    pages_text_by_num[page_num] = text
                else:
                    # Still track empty pages for page anchors if needed
                    pages_text_by_num[page_num] = ""

        if pages_data:
            st.caption("Click a finding on the right to jump and highlight it here.")
            full_text = build_full_document_text(pages_text_by_num)
            render_document_with_highlight(
                full_text=full_text,
                selected_page=st.session_state.selected_page,
                highlight_phrase=st.session_state.selected_phrase
            )
        else:
            st.warning("No readable text was extracted from this PDF.")

with col2:
    st.subheader("2. Findings")

    model = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4.1"],
        index=0,
        help="Start with gpt-4o-mini for cost + speed. Use gpt-4.1 if you want stricter extraction behavior."
    )

    # Category filter (optional / you’re on the fence, so keep but tuck away)
    key_to_label = {
        "UNDEFINED_BOUNDARY": "Match / Tie-in / Patch",
        "IF_POSSIBLE": "Conditional language",
        "AS_NEEDED": "As needed / required",
        "REQUIRED_UPGRADES": "Required upgrades",
        "COORDINATION_GAP": "Coordination gaps",
        "UNDEFINED_SCOPE": "Undefined scope / TBD",
        "SUBJECTIVE_QUALITY": "Subjective quality",
        "EXPLICIT_LIABILITY": "Explicit liability",
    }
    all_keys = list(key_to_label.keys())

    with st.expander("Include categories (optional)", expanded=False):
        selected_labels = st.multiselect(
            "Include categories:",
            options=[key_to_label[k] for k in all_keys],
            default=[key_to_label[k] for k in all_keys],
        )
        include_keys = {k for k in all_keys if key_to_label[k] in selected_labels}

    if pages_data:
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Scanning..."):
                st.session_state.findings = scan_document(pages_data, model_name=model, include_keys=include_keys)
                st.session_state.selected_uid = None
                st.session_state.selected_page = None
                st.session_state.selected_phrase = None

    results = st.session_state.findings
    if results is not None:
        st.info(f"Scan complete. Found {len(results)} items.")

        # Fixed-height findings column with internal scroll
        with st.container(height=560, border=True):
            if len(results) == 0:
                st.write("No findings.")
            else:
                # If we have a selected card, attempt to auto-scroll to it
                # (best-effort; Streamlit DOM can vary by deploy)
                if st.session_state.selected_uid:
                    st.markdown(
                        f"""
                        <script>
                          const el = window.parent.document.getElementById("card_{st.session_state.selected_uid}");
                          if (el) {{
                            el.scrollIntoView({{behavior: "smooth", block: "center"}});
                          }}
                        </script>
                        """,
                        unsafe_allow_html=True
                    )

                for idx, item in enumerate(results):
                    uid = item["uid"]
                    is_selected = (st.session_state.selected_uid == uid)

                    # Anchor for scrolling the right panel (best effort)
                    st.markdown(f'<div id="card_{uid}"></div>', unsafe_allow_html=True)

                    # “Selected card” styling
                    card_bg = "rgba(255,255,255,0.06)" if is_selected else "rgba(255,255,255,0.02)"
                    card_border = "2px solid rgba(242,211,79,0.85)" if is_selected else "1px solid rgba(0,0,0,0.12)"

                    st.markdown(
                        f"""
                        <div style="
                          padding: 10px 10px 8px 10px;
                          border-radius: 10px;
                          border: {card_border};
                          background: {card_bg};
                          margin-bottom: 10px;">
                        """,
                        unsafe_allow_html=True
                    )

                    top = st.columns([0.18, 0.82])

                    # Checkbox (handled)
                    with top[0]:
                        current_val = bool(st.session_state.checked_map.get(uid, False))
                        new_val = st.checkbox("Done", value=current_val, key=f"done_{uid}")
                        st.session_state.checked_map[uid] = new_val

                    with top[1]:
                        btn_label = f'{item["category"]} | Page {item["page"]} | "{item["phrase"]}"'
                        if st.button(btn_label, key=f"pick_{uid}"):
                            st.session_state.selected_uid = uid
                            st.session_state.selected_page = int(item["page"])
                            st.session_state.selected_phrase = item["phrase"]

                    st.caption(f'Gap: {item["gap"]}')
                    st.caption(f'Clarification: {item["question"]}')

                    st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.write("Upload a document to begin.")
