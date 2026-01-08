import streamlit as st
import pdfplumber
import json
import re
import html
import csv
from pathlib import Path
from openai import OpenAI

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Prephase Scope Auditor")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("CRITICAL: Missing OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

RULES_PATH_CANDIDATES = [
    Path("rules.csv"),
    Path("./rules.csv"),
]

# ------------------------------------------------------------
# HELPERS
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
    Conservative: adds spaces in obvious joins, doesn't rewrite wording.
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


def split_text_into_chunks(text: str, max_chars: int = 2400) -> list[str]:
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


def find_rules_path() -> Path | None:
    for p in RULES_PATH_CANDIDATES:
        if p.exists() and p.is_file():
            return p
    return None


def load_rules() -> tuple[dict, list[str], str | None]:
    """
    Loads rules.csv and returns:
      - rules_by_phrase: dict keyed by normalized trigger_phrase
      - phrases: list of trigger phrases (original)
      - warning: optional warning string if file missing/invalid
    """
    rules_path = find_rules_path()
    if not rules_path:
        return {}, [], "Note: rules.csv not detected. Using fallback template language."

    try:
        with rules_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {
                "trigger_phrase",
                "category_tag",
                "target_section",
                "risk_explanation",
                "builder_impact",
                "clarification_question",
                "confidence_requirement",
                "suppression_rule",
            }
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                return {}, [], "Note: rules.csv detected but headers are invalid. Using fallback template language."

            rules_by_phrase = {}
            phrases = []
            for row in reader:
                phrase = (row.get("trigger_phrase") or "").strip()
                if not phrase:
                    continue
                key = normalize_text(phrase)
                # last write wins if duplicates exist
                rules_by_phrase[key] = {
                    "trigger_phrase": phrase,
                    "category_tag": (row.get("category_tag") or "").strip(),
                    "target_section": (row.get("target_section") or "").strip(),
                    "risk_explanation": (row.get("risk_explanation") or "").strip(),
                    "builder_impact": (row.get("builder_impact") or "").strip(),
                    "clarification_question": (row.get("clarification_question") or "").strip(),
                    "confidence_requirement": (row.get("confidence_requirement") or "").strip(),
                    "suppression_rule": (row.get("suppression_rule") or "").strip(),
                }
                phrases.append(phrase)

            if not rules_by_phrase:
                return {}, [], "Note: rules.csv detected but no usable rows found. Using fallback template language."

            # de-dupe phrases while preserving order
            seen = set()
            phrases_unique = []
            for p in phrases:
                k = normalize_text(p)
                if k in seen:
                    continue
                seen.add(k)
                phrases_unique.append(p)

            return rules_by_phrase, phrases_unique, None

    except Exception:
        return {}, [], "Note: rules.csv detected but could not be read. Using fallback template language."


def build_system_instruction(trigger_phrases: list[str]) -> str:
    """
    Build the extraction instruction. We keep this strict so the output is predictable.
    """
    # Keep the list compact but explicit.
    phrase_lines = "\n".join([f"- {p}" for p in trigger_phrases])

    return f"""
ROLE: You are a strict extractor. You do NOT write prose.

TASK:
You are given TEXT. Identify any occurrences of the trigger phrases below.

TRIGGER PHRASES (case-insensitive):
{phrase_lines}

RULES:
- Return ONLY a raw JSON list.
- Each item must be an object with:
  - "trigger_phrase": one of the phrases from the list above (exactly as written in the list)
  - "trigger_text": the EXACT quote from the provided TEXT that matches (preserve original capitalization/spaces/punctuation as it appears)
- Do NOT include reasoning.
- If nothing is found, return [].

OUTPUT FORMAT:
[{{"trigger_phrase":"if possible","trigger_text":"If possible"}}]
""".strip()


def classify_chunk(model_name: str, chunk: str, system_instruction: str) -> list[dict]:
    prompt = f"{system_instruction}\n\nTEXT:\n{chunk}"
    resp = client.responses.create(
        model=model_name,
        input=prompt,
        temperature=0
    )
    raw = (resp.output_text or "").strip()

    # strip code fences if any
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


def scan_document(pages_data: list[dict], model_name: str, rules_by_phrase: dict, trigger_phrases: list[str]) -> list[dict]:
    """
    Returns findings list.
    Each finding includes internal confidence/suppression but UI won't render them.
    """
    findings = []
    seen = set()

    system_instruction = build_system_instruction(trigger_phrases)

    progress = st.progress(0)
    total_pages = len(pages_data)

    for i, p in enumerate(pages_data):
        page_num = p["page"]
        page_text = p["text"]

        chunks = split_text_into_chunks(page_text)
        for chunk in chunks:
            if len(chunk) < 10:
                continue

            hits = classify_chunk(model_name, chunk, system_instruction)

            for item in hits:
                trig_phrase = (item.get("trigger_phrase") or "").strip()
                trig_text = (item.get("trigger_text") or "").strip()
                if not trig_phrase or not trig_text:
                    continue

                rule = rules_by_phrase.get(normalize_text(trig_phrase))
                if not rule:
                    continue

                # Trust anchor: extracted quote must exist in the chunk (normalized)
                if normalize_text(trig_text) not in normalize_text(chunk):
                    continue

                uid = f"{page_num}|{normalize_text(trig_phrase)}|{normalize_text(trig_text)}"
                if uid in seen:
                    continue
                seen.add(uid)

                findings.append({
                    "uid": uid,
                    "page": page_num,
                    "phrase": trig_text,  # what we highlight
                    "trigger_phrase": trig_phrase,
                    "category_tag": rule["category_tag"],
                    "target_section": rule["target_section"],
                    "risk_explanation": rule["risk_explanation"],
                    "builder_impact": rule["builder_impact"],
                    "clarification_question": rule["clarification_question"],
                    "confidence_requirement": rule["confidence_requirement"],  # internal
                    "suppression_rule": rule["suppression_rule"],              # internal
                })

        progress.progress((i + 1) / max(total_pages, 1))

    progress.empty()

    findings.sort(key=lambda x: (x["page"], x["category_tag"], len(x["phrase"])))
    return findings


def render_full_document_scroll(pages_text_by_num: dict[int, str], selected_page: int | None, highlight_phrase: str | None):
    """
    One scrollable reader for the entire doc, with page separators.
    If highlight_phrase is provided, we highlight first occurrence on the selected_page and scroll to it.
    """
    # Build HTML for all pages
    parts = []
    hit_id = None

    for page_num in sorted(pages_text_by_num.keys()):
        raw_text = pages_text_by_num[page_num] or ""
        safe_text = html.escape(raw_text)

        # Default page content
        page_body = safe_text

        # If this is the selected page, try to highlight the phrase
        if highlight_phrase and selected_page == page_num:
            phrase_esc = html.escape(highlight_phrase)
            pattern = re.compile(re.escape(phrase_esc), re.IGNORECASE)
            m = pattern.search(page_body)
            if m:
                start, end = m.span()
                before = page_body[:start]
                mid = page_body[start:end]
                after = page_body[end:]
                hit_id = "hit"
                page_body = before + f'<mark id="{hit_id}" style="background:#f2d34f; padding:0 2px; border-radius:3px;">' + mid + "</mark>" + after

        parts.append(f"""
        <div id="page_{page_num}" style="padding: 10px 10px 14px 10px; border-bottom: 1px solid rgba(255,255,255,0.10);">
          <div style="opacity:0.75; font-size:12px; margin-bottom:6px;">Page {page_num}</div>
          <div style="white-space: pre-wrap;">{page_body}</div>
        </div>
        """)

    joined = "\n".join(parts)

    # Scroll logic: if hit exists scroll to hit, else if selected_page scroll to that page anchor
    scroll_js = ""
    if hit_id:
        scroll_js = """
        <script>
          const box = document.getElementById("docBox");
          const hit = document.getElementById("hit");
          if (box && hit) {
            const top = hit.offsetTop - 120;
            box.scrollTo({ top: top, behavior: "smooth" });
          }
        </script>
        """
    elif selected_page:
        scroll_js = f"""
        <script>
          const box = document.getElementById("docBox");
          const el = document.getElementById("page_{int(selected_page)}");
          if (box && el) {{
            const top = el.offsetTop - 40;
            box.scrollTo({{ top: top, behavior: "smooth" }});
          }}
        </script>
        """

    html_block = f"""
    <div id="docBox" style="height: 560px; overflow-y: auto; padding: 0px; border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 13px; line-height: 1.45;">
      {joined}
    </div>
    {scroll_js}
    """
    st.components.v1.html(html_block, height=600)


# ------------------------------------------------------------
# APP
# ------------------------------------------------------------
st.title("Scope Translator")
st.markdown("**Ethos:** Move the burden from the person to the document.")
st.divider()

# State
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

rules_by_phrase, trigger_phrases, rules_warning = load_rules()

col1, col2 = st.columns([1.6, 1.0])

with col1:
    st.subheader("1. Source Document")
    uploaded_file = st.file_uploader("Upload Scope PDF", type="pdf")

    pages_data = []
    pages_text_by_num = {}

    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                # word-based extraction for more consistent “exact quote” matching
                words = page.extract_words(x_tolerance=1)
                text = " ".join([w.get("text", "") for w in words if w.get("text")])
                text = spacing_fix(text)

                if text.strip():
                    page_num = i + 1
                    pages_data.append({"page": page_num, "text": text})
                    pages_text_by_num[page_num] = text

        if pages_data:
            st.caption("Click a finding on the right to jump and highlight it here.")
            render_full_document_scroll(
                pages_text_by_num=pages_text_by_num,
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

    if rules_warning:
        st.caption(rules_warning)

    if pages_data:
        if st.button("Run Analysis", type="primary", use_container_width=True):
            if not trigger_phrases or not rules_by_phrase:
                st.error("rules.csv missing or invalid. Upload/rename it to rules.csv at repo root.")
            else:
                with st.spinner("Scanning..."):
                    st.session_state.findings = scan_document(
                        pages_data=pages_data,
                        model_name=model,
                        rules_by_phrase=rules_by_phrase,
                        trigger_phrases=trigger_phrases
                    )
                    st.session_state.selected_phrase = None
                    st.session_state.selected_page = None
                    st.session_state.selected_uid = None

    results = st.session_state.findings
    if results is not None:
        st.info(f"Scan complete. Found {len(results)} items.")

        with st.container(height=560, border=True):
            if len(results) == 0:
                st.write("No findings.")
            else:
                for idx, item in enumerate(results):
                    uid = item["uid"]

                    # checkbox state
                    done_key = f"done_{uid}"
                    if done_key not in st.session_state:
                        st.session_state[done_key] = bool(st.session_state.done_map.get(uid, False))

                    # Selected button styling
                    is_selected = (st.session_state.selected_uid == uid)

                    # Row layout: checkbox + button
                    cbox_col, btn_col = st.columns([0.12, 0.88], vertical_alignment="center")
                    with cbox_col:
                        new_done = st.checkbox(" ", key=done_key)
                        st.session_state.done_map[uid] = bool(new_done)

                    with btn_col:
                        btn_label = f'Page {item["page"]} | "{item["phrase"]}"'
                        if st.button(
                            btn_label,
                            key=f"pick_{uid}",
                            type="primary" if is_selected else "secondary",
                            use_container_width=True
                        ):
                            st.session_state.selected_uid = uid
                            st.session_state.selected_page = int(item["page"])
                            st.session_state.selected_phrase = item["phrase"]

                    # Card details (NO confidence/suppression shown)
                    st.caption(f'Tag: {item.get("category_tag","")}')
                    st.caption(f'Section: {item.get("target_section","")}')
                    st.caption(f'Risk: {item.get("risk_explanation","")}')
                    st.caption(f'Impact: {item.get("builder_impact","")}')
                    st.caption(f'Question: {item.get("clarification_question","")}')
                    st.divider()
    else:
        st.write("Upload a document to begin.")
