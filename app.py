import streamlit as st
import pdfplumber
import json
import re
import html as html_lib
from openai import OpenAI

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Prephase Scope Auditor")

st.title("Prephase Scope Auditor")
st.markdown("**Ethos:** Move the burden from the person to the document.")
st.divider()

if "OPENAI_API_KEY" not in st.secrets:
    st.error("CRITICAL: Missing OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# ------------------------------------------------------------
# SAFE LIBRARY (BRAIN: unchanged output style)
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
    # missing space after period when next char is letter
    t = re.sub(r"(?<=[.])(?=[A-Za-z])", " ", t)

    return t.strip()


def split_text_into_chunks(text: str, max_chars: int = 2200) -> list[str]:
    """
    Chunking prevents truncation and reduces missed hits.
    """
    text = text.strip()
    if not text:
        return []

    # Try paragraph-ish splits; fallback is still okay
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
# LLM CALL (BRAIN unchanged; provider is OpenAI)
# ------------------------------------------------------------
def classify_chunk(model_name: str, chunk: str) -> list[dict]:
    prompt = f"{SYSTEM_INSTRUCTION}\n\nTEXT TO ANALYZE:\n{chunk}"
    resp = client.responses.create(
        model=model_name,
        input=prompt,
        temperature=0
    )

    raw = (resp.output_text or "").strip()

    # Strip code fences if present
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


def scan_document(pages_data: list[dict], model_name: str, allowed_keys: set[str]) -> list[dict]:
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
                key = item.get("classification", "")
                quote = item.get("trigger_text", "")

                if not key or not quote:
                    continue
                if key not in RESPONSE_TEMPLATES:
                    continue
                if key not in allowed_keys:
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
                    "category": t["category"],
                    "gap": t["gap"],
                    "question": t["question"],
                    "page": page_num,
                    "key": key,
                })

        progress.progress((i + 1) / max(total_pages, 1))

    progress.empty()

    findings.sort(key=lambda x: (x["page"], x["category"], len(x["phrase"])))
    return findings


# ------------------------------------------------------------
# UI: Scrollable Source Document (all pages) with jump-to-highlight
# ------------------------------------------------------------
def render_document_scroll(pages_text_by_num: dict[int, str], selected_page: int | None, highlight_phrase: str | None):
    """
    Renders a single scrollable viewer containing all pages.
    If a selection exists, we:
      - highlight first occurrence of phrase on the selected page (if found)
      - auto-scroll to the page anchor
      - also try to scroll to the highlight on that page
    """
    if not pages_text_by_num:
        st.warning("No readable text was extracted from this PDF.")
        return

    # Build HTML for all pages
    blocks = []
    hit_id = None

    for page_num in sorted(pages_text_by_num.keys()):
        raw_text = pages_text_by_num[page_num] or ""
        safe_text = html_lib.escape(raw_text)

        page_anchor = f"page_{page_num}"
        page_html = safe_text

        # Only highlight on selected page
        if highlight_phrase and selected_page == page_num:
            phrase_esc = html_lib.escape(highlight_phrase)
            pattern = re.compile(re.escape(phrase_esc), re.IGNORECASE)
            m = pattern.search(page_html)
            if m:
                start, end = m.span()
                before = page_html[:start]
                mid = page_html[start:end]
                after = page_html[end:]
                hit_id = "hit"
                page_html = (
                    before
                    + f'<mark id="{hit_id}" style="background:#f2d34f; padding:0 2px; border-radius:3px;">'
                    + mid
                    + "</mark>"
                    + after
                )

        blocks.append(
            f"""
            <div id="{page_anchor}" style="padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.08);">
              <div style="opacity:0.75; font-size:12px; margin-bottom:8px;">--- Page {page_num} ---</div>
              <div style="white-space: pre-wrap;">{page_html}</div>
            </div>
            """
        )

    selected_anchor = f"page_{selected_page}" if selected_page else None

    html_block = f"""
    <div id="docBox" style="
        height: 560px;
        overflow-y: auto;
        padding: 12px;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 10px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.45;">
      {''.join(blocks)}
    </div>

    <script>
      (function() {{
        const box = document.getElementById("docBox");
        if (!box) return;

        const anchorId = {json.dumps(selected_anchor)};
        const hitId = {json.dumps(hit_id)};

        // First scroll to page anchor
        if (anchorId) {{
          const anchor = document.getElementById(anchorId);
          if (anchor) {{
            // Scroll within the box
            box.scrollTo({{ top: anchor.offsetTop - 20, behavior: "smooth" }});
          }}
        }}

        // Then scroll to highlight, if present
        if (hitId) {{
          const hit = document.getElementById(hitId);
          if (hit) {{
            setTimeout(() => {{
              box.scrollTo({{ top: hit.offsetTop - 120, behavior: "smooth" }});
            }}, 180);
          }}
        }}
      }})();
    </script>
    """
    st.components.v1.html(html_block, height=610)


# ------------------------------------------------------------
# STATE
# ------------------------------------------------------------
if "findings" not in st.session_state:
    st.session_state.findings = None
if "selected_uid" not in st.session_state:
    st.session_state.selected_uid = None
if "selected_page" not in st.session_state:
    st.session_state.selected_page = None
if "selected_phrase" not in st.session_state:
    st.session_state.selected_phrase = None
if "done_map" not in st.session_state:
    st.session_state.done_map = {}  # uid -> bool


# ------------------------------------------------------------
# LAYOUT
# ------------------------------------------------------------
col1, col2 = st.columns([1.6, 1.0])

with col1:
    st.subheader("1. Source Document")
    uploaded_file = st.file_uploader("Upload Scope PDF", type="pdf")

    pages_data = []
    pages_text_by_num = {}

    if uploaded_file is not None:
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Robust-ish extraction using words
                    words = page.extract_words(x_tolerance=1)
                    text = " ".join([w.get("text", "") for w in words if w.get("text")])
                    text = spacing_fix(text)

                    page_num = i + 1
                    if text.strip():
                        pages_data.append({"page": page_num, "text": text})
                        pages_text_by_num[page_num] = text
        except Exception as e:
            st.error(f"PDF read error: {e}")
            pages_data = []
            pages_text_by_num = {}

    if uploaded_file is not None and pages_text_by_num:
        st.caption("Click a finding on the right to jump and highlight it here.")
        render_document_scroll(
            pages_text_by_num,
            st.session_state.selected_page,
            st.session_state.selected_phrase
        )


with col2:
    st.subheader("2. Findings")

    model = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4.1"],
        index=0,
        help="Use gpt-4o-mini for speed/cost. Use gpt-4.1 for stricter behavior."
    )

    # Optional category filter (you said you're on the fence — so this is tucked away)
    CATEGORY_UI = {
        "UNDEFINED_BOUNDARY": "Match / Tie-in / Patch",
        "IF_POSSIBLE": "Conditional language",
        "AS_NEEDED": "As needed / required",
        "REQUIRED_UPGRADES": "Required upgrades",
        "COORDINATION_GAP": "Coordination gaps",
        "UNDEFINED_SCOPE": "Undefined scope / TBD",
        "SUBJECTIVE_QUALITY": "Subjective quality",
        "EXPLICIT_LIABILITY": "Explicit liability",
    }

    with st.expander("Include categories (optional)", expanded=False):
        cols = st.columns(2)
        selected_keys = set()

        # default: all on
        for i, (k, label) in enumerate(CATEGORY_UI.items()):
            with cols[i % 2]:
                checked = st.checkbox(label, value=True, key=f"cat_{k}")
                if checked:
                    selected_keys.add(k)

        if not selected_keys:
            st.warning("No categories selected; scan would return 0 items.")

    if uploaded_file is not None and pages_data:
        if st.button("Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Scanning..."):
                st.session_state.findings = scan_document(
                    pages_data=pages_data,
                    model_name=model,
                    allowed_keys=selected_keys if selected_keys else set(CATEGORY_UI.keys())
                )
                # Do not force-clear selection; keep it unless it no longer exists
                if st.session_state.findings:
                    existing_uids = {f["uid"] for f in st.session_state.findings}
                    if st.session_state.selected_uid not in existing_uids:
                        st.session_state.selected_uid = None
                        st.session_state.selected_page = None
                        st.session_state.selected_phrase = None

    results = st.session_state.findings

    if results is None:
        st.write("Upload a document to begin.")
    else:
        st.info(f"Scan complete. Found {len(results)} items.")

        # Fixed-height findings column with internal scroll
        with st.container(height=560, border=True):
            if len(results) == 0:
                st.write("No findings.")
            else:
                for idx, item in enumerate(results):
                    uid = item["uid"]
                    is_selected = (st.session_state.selected_uid == uid)

                    # DONE checkbox (persisted in session)
                    done_key = f"done_{uid}"
                    if done_key not in st.session_state:
                        st.session_state[done_key] = bool(st.session_state.done_map.get(uid, False))

                    # Inline row: checkbox + button
                    left, right = st.columns([0.12, 0.88], vertical_alignment="center")
                    with left:
                        done_val = st.checkbox("Done", key=done_key, label_visibility="collapsed")
                        st.session_state.done_map[uid] = bool(done_val)

                    with right:
                        btn_label = f'{item["category"]} | Page {item["page"]} | "{item["phrase"]}"'
                        if st.button(
                            btn_label,
                            key=f"pick_{uid}",
                            type="primary" if is_selected else "secondary",
                            use_container_width=True
                        ):
                            st.session_state.selected_uid = uid
                            st.session_state.selected_page = int(item["page"])
                            st.session_state.selected_phrase = item["phrase"]
                            # This prevents the “double click” symptom by forcing an immediate rerun
                            st.rerun()

                    st.caption(f'Gap: {item["gap"]}')
                    st.caption(f'Clarification: {item["question"]}')
                    st.divider()
