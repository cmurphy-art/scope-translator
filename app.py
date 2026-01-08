import streamlit as st
import pdfplumber
import json
import re
import html
import pandas as pd
from openai import OpenAI

# ------------------------------------------------------------
# CONFIG (UI UNCHANGED)
# ------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Prephase Scope Translator (UI Jump + Scroll)")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("CRITICAL: Missing OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------------------------------------------------
# RULES (Spreadsheet-backed brain expansion)
# ------------------------------------------------------------
RULES_CSV_PATH = "rules.csv"  # put in the repo next to app.py

REQUIRED_RULE_COLS = [
    "trigger_phrase",
    "category_tag",
    "target_section",
    "risk_explanation",
    "builder_impact",
    "clarification_question",
    "confidence_requirement",
    "suppression_rule",
]

# ------------------------------------------------------------
# SAFE LIBRARY (UNCHANGED TEMPLATE FALLBACKS)
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
7. AS_NEEDED (Triggers: "as needed", "as required", "as necessary", "if needed", "modify as needed", "remove and replace as needed")
8. REQUIRED_UPGRADES (Triggers: "required upgrades", "bring to code", "code upgrades", "upgrade", "changes required by the building department")

CONSTRAINTS:
- Return ONLY a raw JSON list.
- Extract the EXACT quote from the provided text.
- Do NOT provide reasoning. Only provide the "classification" KEY.
- Return [] if nothing found.

OUTPUT FORMAT:
[{"trigger_text": "...", "classification": "UNDEFINED_BOUNDARY"}]
"""

# ------------------------------------------------------------
# TEXT HELPERS (UNCHANGED)
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
    Keep conservative. Do not rewrite content, only add spaces in obvious joins.
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

    paras = re.split(r"\n\s*\n", text)
    chunks = []
    cur = ""

    for p in paras:
        p = p.strip()
        if not p:
            continue

        add = (p + "\n\n")
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
# RULES LOADER + MATCHER (NEW - BRAIN ONLY)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_rules_from_csv(path: str) -> list[dict]:
    """
    Loads your spreadsheet as the canonical brain lookup table.
    Longest trigger_phrase wins.
    """
    try:
        df = pd.read_csv(path).fillna("")
    except Exception:
        return []

    missing = [c for c in REQUIRED_RULE_COLS if c not in df.columns]
    if missing:
        return []

    rules = df[REQUIRED_RULE_COLS].to_dict(orient="records")

    for r in rules:
        r["_norm_trigger"] = normalize_text(str(r["trigger_phrase"]))
        r["_trigger_len"] = len(r["_norm_trigger"])

    rules = [r for r in rules if r["_norm_trigger"]]
    rules.sort(key=lambda x: x["_trigger_len"], reverse=True)
    return rules

def find_best_rule_for_phrase(rules: list[dict], phrase: str) -> dict | None:
    """
    Best match: find the longest trigger_phrase that appears within the found phrase.
    """
    norm_phrase = normalize_text(phrase)
    for r in rules:
        if r["_norm_trigger"] in norm_phrase:
            return r
    return None

# ------------------------------------------------------------
# LLM CALL (UNCHANGED PROVIDER)
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

def scan_document(pages_data: list[dict], model_name: str) -> list[dict]:
    findings = []
    seen = set()

    # Load rules once (cached)
    rules = load_rules_from_csv(RULES_CSV_PATH)

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

                # Trust anchor: quote must exist in chunk
                if normalize_text(quote) not in normalize_text(chunk):
                    continue

                uid = f"{page_num}|{normalize_text(quote)}|{key}"
                if uid in seen:
                    continue
                seen.add(uid)

                # Fallback template
                t = RESPONSE_TEMPLATES[key]

                # Spreadsheet enrichment (if rules present + match found)
                rule = find_best_rule_for_phrase(rules, quote) if rules else None

                if rule:
                    findings.append({
                        "phrase": quote,
                        "page": page_num,

                        # Spreadsheet-driven fields
                        "category_tag": rule["category_tag"],
                        "target_section": rule["target_section"],
                        "risk_explanation": rule["risk_explanation"],
                        "builder_impact": rule["builder_impact"],
                        "clarification_question": rule["clarification_question"],
                        "confidence_requirement": rule["confidence_requirement"],
                        "suppression_rule": rule["suppression_rule"],

                        # Keep original classifier too (useful for debugging)
                        "classification": key,
                    })
                else:
                    # Keep old behavior if no rule matched
                    findings.append({
                        "phrase": quote,
                        "page": page_num,
                        "category": t["category"],
                        "gap": t["gap"],
                        "question": t["question"],
                        "classification": key,
                    })

        progress.progress((i + 1) / max(total_pages, 1))

    progress.empty()

    findings.sort(key=lambda x: (x["page"], x.get("category_tag", x.get("category", "")), len(x["phrase"])))
    return findings

# ------------------------------------------------------------
# UI: Page Viewer with jump-to-highlight (UNCHANGED)
# ------------------------------------------------------------
def render_page_with_highlight(page_text: str, highlight_phrase: str | None):
    """
    Renders a scrollable div. If highlight_phrase is present, highlights first occurrence and scrolls to it.
    """
    safe_text = html.escape(page_text)

    if not highlight_phrase:
        html_block = f"""
        <div id="pageBox" style="height: 520px; overflow-y: auto; padding: 14px; border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 13px; line-height: 1.45;">
        {safe_text}
        </div>
        """
        st.components.v1.html(html_block, height=560)
        return

    phrase_esc = html.escape(highlight_phrase)
    pattern = re.compile(re.escape(phrase_esc), re.IGNORECASE)

    match = pattern.search(safe_text)
    if not match:
        html_block = f"""
        <div id="pageBox" style="height: 520px; overflow-y: auto; padding: 14px; border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 13px; line-height: 1.45;">
        {safe_text}
        </div>
        """
        st.components.v1.html(html_block, height=560)
        return

    start, end = match.span()
    before = safe_text[:start]
    mid = safe_text[start:end]
    after = safe_text[end:]

    highlighted = before + '<mark id="hit" style="background: #f2d34f; padding: 0 2px; border-radius: 3px;">' + mid + "</mark>" + after

    html_block = f"""
    <div id="pageBox" style="height: 520px; overflow-y: auto; padding: 14px; border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 13px; line-height: 1.45;">
    {highlighted}
    </div>

    <script>
      const box = document.getElementById("pageBox");
      const hit = document.getElementById("hit");
      if (box && hit) {{
        const top = hit.offsetTop - 80;
        box.scrollTo({{ top: top, behavior: "smooth" }});
      }}
    </script>
    """
    st.components.v1.html(html_block, height=560)

# ------------------------------------------------------------
# APP (UI UNCHANGED)
# ------------------------------------------------------------
st.title("Scope Translator")
st.markdown("**Ethos:** Move the burden from the person to the document.")
st.divider()

# State
if "findings" not in st.session_state:
    st.session_state.findings = None
if "selected_page" not in st.session_state:
    st.session_state.selected_page = 1
if "selected_phrase" not in st.session_state:
    st.session_state.selected_phrase = None
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None
if "checked" not in st.session_state:
    st.session_state.checked = set()

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

                if text.strip():
                    page_num = i + 1
                    pages_data.append({"page": page_num, "text": text})
                    pages_text_by_num[page_num] = text

        if pages_data:
            st.caption("Click a finding on the right to jump and highlight it here.")

            max_page = max(pages_text_by_num.keys())
            st.session_state.selected_page = max(1, min(st.session_state.selected_page, max_page))

            page_choice = st.number_input(
                "Page",
                min_value=1,
                max_value=max_page,
                value=int(st.session_state.selected_page),
                step=1
            )
            st.session_state.selected_page = int(page_choice)

            current_text = pages_text_by_num.get(st.session_state.selected_page, "")
            render_page_with_highlight(current_text, st.session_state.selected_phrase)
        else:
            st.warning("No readable text was extracted from this PDF.")

with col2:
    st.subheader("2. Findings")

    model = st.selectbox(
        "Model",
        options=[
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1"
        ],
        index=0,
        help="Start with gpt-4o-mini for cost + speed. Use gpt-4.1 if you want stricter extraction behavior."
    )

    # Optional: show rule load status (does not change UI structure)
    rules_loaded = len(load_rules_from_csv(RULES_CSV_PATH)) > 0
    if not rules_loaded:
        st.caption("Note: rules.csv not detected or invalid. Using fallback template language.")

    if pages_data:
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Scanning..."):
                st.session_state.findings = scan_document(pages_data, model_name=model)
                st.session_state.selected_phrase = None
                st.session_state.selected_idx = None
                st.session_state.checked = set()

    results = st.session_state.findings
    if results is not None:
        st.info(f"Scan complete. Found {len(results)} items.")

        with st.container(height=560, border=True):
            if len(results) == 0:
                st.write("No findings.")
            else:
                for idx, item in enumerate(results):
                    # checkbox state
                    is_checked = idx in st.session_state.checked
                    is_selected = (st.session_state.selected_idx == idx)

                    # label (keep the same look/feel, but still readable)
                    btn_label = f'Page {item["page"]} | "{item["phrase"]}"'

                    c1, c2 = st.columns([0.14, 0.86])
                    with c1:
                        checked_now = st.checkbox("", value=is_checked, key=f"chk_{idx}")
                    with c2:
                        if st.button(
                            btn_label,
                            key=f"pick_{idx}",
                            type="primary" if is_selected else "secondary",
                            use_container_width=True
                        ):
                            st.session_state.selected_idx = idx
                            st.session_state.selected_page = int(item["page"])
                            st.session_state.selected_phrase = item["phrase"]

                    # persist checkbox state
                    if checked_now:
                        st.session_state.checked.add(idx)
                    else:
                        if idx in st.session_state.checked:
                            st.session_state.checked.remove(idx)

                    # Display: prefer spreadsheet fields if present; fallback otherwise
                    if "category_tag" in item:
                        st.caption(f'Tag: {item.get("category_tag","")}')
                        if item.get("target_section"):
                            st.caption(f'Section: {item.get("target_section","")}')
                        if item.get("risk_explanation"):
                            st.caption(f'Risk: {item.get("risk_explanation","")}')
                        if item.get("builder_impact"):
                            st.caption(f'Impact: {item.get("builder_impact","")}')
                        if item.get("clarification_question"):
                            st.caption(f'Question: {item.get("clarification_question","")}')
                        if item.get("confidence_requirement"):
                            st.caption(f'Confidence: {item.get("confidence_requirement","")}')
                        if item.get("suppression_rule"):
                            st.caption(f'Suppression: {item.get("suppression_rule","")}')
                    else:
                        st.caption(f'Category: {item.get("category","")}')
                        st.caption(f'Gap: {item.get("gap","")}')
                        st.caption(f'Clarification: {item.get("question","")}')

                    st.divider()
    else:
        st.write("Upload a document to begin.")
