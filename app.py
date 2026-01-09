import streamlit as st
import pdfplumber
import csv
import io
import re
import html
import hashlib
import shutil
from dataclasses import dataclass
from typing import Optional, List, Dict

from openai import OpenAI

# OCR deps (pip)
import pytesseract
import pypdfium2 as pdfium
from PIL import Image

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Prephase Scope Auditor")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("CRITICAL: Missing OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # do not change

RULES_FILENAME = "rules.csv"

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
    """Conservative spacing repair for pdf word-joins."""
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

def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ------------------------------------------------------------
# RULES LOADING (BRAIN)
# ------------------------------------------------------------
@dataclass
class Rule:
    trigger_phrase: str
    category_tag: str
    target_section: str
    risk_explanation: str
    builder_impact: str
    clarification_question: str
    confidence_requirement: str
    suppression_rule: str

def load_rules_csv(path: str) -> List[Rule]:
    rules: List[Rule] = []
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required_cols = [
                "trigger_phrase",
                "category_tag",
                "target_section",
                "risk_explanation",
                "builder_impact",
                "clarification_question",
                "confidence_requirement",
                "suppression_rule",
            ]
            for c in required_cols:
                if c not in reader.fieldnames:
                    raise ValueError(f"Missing column in rules.csv: {c}")

            for row in reader:
                trig = (row.get("trigger_phrase") or "").strip()
                if not trig:
                    continue
                rules.append(
                    Rule(
                        trigger_phrase=trig,
                        category_tag=(row.get("category_tag") or "").strip(),
                        target_section=(row.get("target_section") or "").strip(),
                        risk_explanation=(row.get("risk_explanation") or "").strip(),
                        builder_impact=(row.get("builder_impact") or "").strip(),
                        clarification_question=(row.get("clarification_question") or "").strip(),
                        confidence_requirement=(row.get("confidence_requirement") or "").strip(),
                        suppression_rule=(row.get("suppression_rule") or "").strip(),
                    )
                )
    except FileNotFoundError:
        return []
    except Exception:
        return []
    return rules

# ------------------------------------------------------------
# OCR FALLBACK (BRAIN-SIDE ONLY)
# ------------------------------------------------------------
def ocr_available() -> bool:
    # pytesseract needs the system "tesseract" binary
    return shutil.which("tesseract") is not None

@st.cache_data(show_spinner=False)
def ocr_page_text(pdf_bytes: bytes, page_index: int) -> str:
    """
    OCR a single page (0-indexed) using pdfium renderer + pytesseract.
    Cached by pdf hash + page index via st.cache_data.
    """
    doc = pdfium.PdfDocument(pdf_bytes)
    page = doc.get_page(page_index)
    # Render at higher scale for better OCR
    bitmap = page.render(scale=2.0)
    pil_image = bitmap.to_pil()
    text = pytesseract.image_to_string(pil_image, lang="eng")
    return spacing_fix(text)

def extract_pages_text(uploaded_pdf_bytes: bytes) -> Dict[int, str]:
    """
    Extract page text using pdfplumber first.
    If a page yields no usable text, OCR that page.
    Returns {page_num (1-indexed): text}
    """
    pages_text: Dict[int, str] = {}

    # 1) pdfplumber pass
    with pdfplumber.open(io.BytesIO(uploaded_pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            words = page.extract_words(x_tolerance=1) or []
            text = " ".join([w.get("text", "") for w in words]).strip()
            text = spacing_fix(text)

            page_num = i + 1
            pages_text[page_num] = text

    # 2) OCR fallback (only for empty/near-empty pages)
    # Keep threshold conservative so we don't OCR normal text PDFs.
    needs_ocr = [p for p, t in pages_text.items() if len((t or "").strip()) < 20]

    if needs_ocr:
        if not ocr_available():
            st.warning(
                "This PDF appears to be scanned (no selectable text). "
                "OCR is not available on this deployment. "
                "Add a packages.txt file with `tesseract-ocr` and redeploy."
            )
            return pages_text

        # Run OCR only for pages that need it
        for page_num in needs_ocr:
            idx0 = page_num - 1
            ocr_text = ocr_page_text(uploaded_pdf_bytes, idx0)
            # Only replace if OCR produced something meaningful
            if len((ocr_text or "").strip()) >= 20:
                pages_text[page_num] = ocr_text

    return pages_text

# ------------------------------------------------------------
# SCANNING (RULES ENGINE)
# ------------------------------------------------------------
def scan_with_rules(pages_text_by_num: Dict[int, str], rules: List[Rule]) -> List[dict]:
    findings: List[dict] = []
    seen = set()

    page_items = sorted(pages_text_by_num.items(), key=lambda x: x[0])
    progress = st.progress(0)
    total = max(len(page_items), 1)

    for idx, (page_num, page_text) in enumerate(page_items):
        if not page_text or len(page_text.strip()) < 10:
            progress.progress((idx + 1) / total)
            continue

        for r in rules:
            phrase = r.trigger_phrase.strip()
            if not phrase:
                continue

            # Case-insensitive literal search
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            for m in pattern.finditer(page_text):
                hit = m.group(0)
                uid = f"{page_num}|{normalize_text(hit)}|{normalize_text(phrase)}"
                if uid in seen:
                    continue
                seen.add(uid)

                findings.append(
                    {
                        "uid": uid,
                        "phrase": hit,
                        "category_tag": r.category_tag,
                        "target_section": r.target_section,
                        "risk_explanation": r.risk_explanation,
                        "builder_impact": r.builder_impact,
                        "clarification_question": r.clarification_question,
                        # keep these in data (brain), but DO NOT display on cards:
                        "confidence_requirement": r.confidence_requirement,
                        "suppression_rule": r.suppression_rule,
                        "page": page_num,
                    }
                )

        progress.progress((idx + 1) / total)

    progress.empty()
    findings.sort(key=lambda x: (x["page"], x["category_tag"], len(x["phrase"])))
    return findings

# ------------------------------------------------------------
# UI: Page Viewer with jump-to-highlight (UNCHANGED)
# ------------------------------------------------------------
def render_page_with_highlight(page_text: str, highlight_phrase: Optional[str]):
    """
    Renders a scrollable div. If highlight_phrase is present, highlights first occurrence and scrolls to it.
    """
    safe_text = html.escape(page_text or "")

    # Default: no highlight
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
# APP
# ------------------------------------------------------------
st.title("Scope Auditor")
st.markdown("Move the burden from the person to the document.")
st.divider()

# State
if "findings" not in st.session_state:
    st.session_state.findings = None
if "selected_page" not in st.session_state:
    st.session_state.selected_page = 1
if "selected_phrase" not in st.session_state:
    st.session_state.selected_phrase = None
if "selected_uid" not in st.session_state:
    st.session_state.selected_uid = None
if "done_map" not in st.session_state:
    st.session_state.done_map = {}  # uid -> bool

rules = load_rules_csv(RULES_FILENAME)
rules_ok = len(rules) > 0

col1, col2 = st.columns([1.6, 1.0])

with col1:
    st.subheader("1. Source Document")
    uploaded_file = st.file_uploader("Upload Scope PDF", type="pdf")

    pages_text_by_num: Dict[int, str] = {}

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        pages_text_by_num = extract_pages_text(pdf_bytes)

        if pages_text_by_num and any(len((t or "").strip()) >= 10 for t in pages_text_by_num.values()):
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
        options=["gpt-4o-mini", "gpt-4o", "gpt-4.1"],
        index=0,
        help="UI only. The rules engine drives findings; model selection is kept for continuity."
    )

    if not rules_ok:
        st.caption("Note: rules.csv not detected or invalid. Using fallback template language.")
    else:
        st.caption("Using rules.csv for detection.")

    if uploaded_file is not None and pages_text_by_num:
        if st.button("Run Analysis", type="primary"):
            with st.spinner("Scanning..."):
                if rules_ok:
                    st.session_state.findings = scan_with_rules(pages_text_by_num, rules)
                else:
                    st.session_state.findings = []
                st.session_state.selected_phrase = None
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
                    is_selected = (st.session_state.selected_uid == uid)

                    done_key = f"done_{uid}"
                    if done_key not in st.session_state.done_map:
                        st.session_state.done_map[done_key] = False

                    top_cols = st.columns([0.14, 0.86])
                    with top_cols[0]:
                        st.session_state.done_map[done_key] = st.checkbox(
                            "",
                            value=st.session_state.done_map[done_key],
                            key=f"chk_{uid}"
                        )

                    with top_cols[1]:
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

                    # Card details (NO confidence/suppression)
                    st.caption(f'Tag: {item["category_tag"]}')
                    st.caption(f'Section: {item["target_section"]}')
                    st.caption(f'Risk: {item["risk_explanation"]}')
                    st.caption(f'Impact: {item["builder_impact"]}')
                    st.caption(f'Question: {item["clarification_question"]}')
                    st.divider()
    else:
        st.write("Upload a document to begin.")
