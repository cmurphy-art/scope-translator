# Scope Translator V17 (Quota-Aware / One Call Per Page)

import streamlit as st
import pdfplumber
import google.generativeai as genai
import json
import re
import hashlib
import time

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Scope Translator V17 (Quota-Aware)")

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("CRITICAL: Google API Key missing.")
    st.stop()

# ------------------------------------------------------------
# RESPONSE TEMPLATES (PUPPET MASTER)
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

# ------------------------------------------------------------
# MODEL PICKER
# ------------------------------------------------------------
def get_best_available_model():
    try:
        available = [
            m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        for m in ["models/gemini-2.5-flash", "models/gemini-1.5-flash", "models/gemini-pro"]:
            if m in available:
                return m
        return available[0] if available else "models/gemini-pro"
    except:
        return "models/gemini-pro"

# ------------------------------------------------------------
# STRICT CLASSIFIER PROMPT
# ------------------------------------------------------------
SYSTEM_INSTRUCTION = """
ROLE: You are a strict Classifier. You DO NOT write text. You only select keys.

TASK: Analyze the text snippet. Identify specific ambiguities using the KEYS below.

KEYS:
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
- Extract the EXACT quote.
- Do NOT provide reasoning.
- IGNORE performance verbs like "optimize", "maximize", "ensure".
- Return [] if nothing found.

OUTPUT FORMAT:
[{"trigger_text": "...", "classification": "UNDEFINED_BOUNDARY"}]
"""

# ------------------------------------------------------------
# NORMALIZATION / VALIDATION
# ------------------------------------------------------------
def clean_json_text(text):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

def safe_load_json(text):
    try:
        return json.loads(clean_json_text(text))
    except:
        return []

def normalize_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())

def generate_id(page, key, quote):
    raw = f"p{page}|{key}|{normalize_text(quote)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

# ------------------------------------------------------------
# PRE-SCAN (SAVES DAILY QUOTA)
# ------------------------------------------------------------
TRIGGER_PATTERNS = [
    r"\bmatch existing\b", r"\btie into\b", r"\bpatch\b", r"\brepair\b",
    r"\bindustry standard\b", r"\bworkmanlike\b", r"\bsatisfaction of\b",
    r"\bturnkey\b", r"\bcomplete system\b", r"\bincluding but not limited to\b",
    r"\bliquidated damages\b", r"\btime is of the essence\b", r"\bindemnify\b",
    r"\bcoordinate with\b", r"\bverify in field\b", r"\bby others\b",
    r"\bif possible\b", r"\bwhere possible\b", r"\bif feasible\b",
    r"\bas needed\b", r"\bas required\b", r"\bas necessary\b",
    r"\brequired upgrades\b", r"\bbring to code\b", r"\bcode upgrades\b",
]

TRIGGER_RE = re.compile("|".join(TRIGGER_PATTERNS), flags=re.IGNORECASE)

def page_has_triggers(text):
    return bool(TRIGGER_RE.search(text or ""))

# ------------------------------------------------------------
# GEMINI CALL WITH BACKOFF (BURST PROTECTION ONLY)
# ------------------------------------------------------------
def generate_with_retry(model, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            if "429" in str(e):
                time.sleep(6 * (2 ** attempt))
                continue
            raise
    raise RuntimeError("Retry limit exceeded")

# ------------------------------------------------------------
# SCAN ENGINE (ONE CALL PER PAGE)
# ------------------------------------------------------------
def scan_document(pages_data):
    findings = []
    seen = set()

    model_name = get_best_available_model()
    model = genai.GenerativeModel(
        model_name,
        generation_config={"temperature": 0.0, "top_p": 0.1, "max_output_tokens": 1024}
    )

    progress = st.progress(0)
    total_pages = len(pages_data)

    for i, page in enumerate(pages_data):
        progress.progress((i + 1) / max(total_pages, 1))

        page_num = page["page"]
        text = page["text"]

        if not page_has_triggers(text):
            continue

        prompt = f"{SYSTEM_INSTRUCTION}\n\nTEXT TO ANALYZE:\n{text}"

        try:
            response = generate_with_retry(model, prompt)
            data = safe_load_json(response.text)
        except Exception as e:
            if "quota" in str(e).lower():
                st.error("Daily Gemini quota reached. Stop and retry after reset.")
                break
            continue

        norm_page = normalize_text(text)

        for item in data or []:
            key = item.get("classification")
            quote = item.get("trigger_text")

            if key not in RESPONSE_TEMPLATES or not quote:
                continue

            if normalize_text(quote) not in norm_page:
                continue

            uid = generate_id(page_num, key, quote)
            if uid in seen:
                continue
            seen.add(uid)

            tpl = RESPONSE_TEMPLATES[key]
            findings.append({
                "phrase": quote,
                "category": tpl["category"],
                "gap": tpl["gap"],
                "question": tpl["question"],
                "page": page_num
            })

    progress.empty()
    return findings

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("Scope Translator V17 (Quota-Aware)")
st.markdown("**Ethos:** One Gemini call per page. Neutral, document-centered analysis.")
st.divider()

col1, col2 = st.columns([1.5, 1])

with col1:
    uploaded = st.file_uploader("Upload Scope PDF", type="pdf")
    pages_data = []
    display_text = ""

    if uploaded:
        with pdfplumber.open(uploaded) as pdf:
            for i, page in enumerate(pdf.pages):
                words = page.extract_words(x_tolerance=1)
                text = " ".join(w["text"] for w in words)
                text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
                if text:
                    pages_data.append({"page": i + 1, "text": text})
                    display_text += f"--- Page {i+1} ---\n{text}\n\n"

        st.text_area("Extracted Text", display_text, height=600)

with col2:
    if pages_data:
        st.caption(
            f"Estimated Gemini calls this run: "
            f"{sum(1 for p in pages_data if page_has_triggers(p['text']))}"
        )

        if st.button("Run Strict Analysis"):
            with st.spinner("Analyzing (Quota-Aware)..."):
                st.session_state.results = scan_document(pages_data)

        results = st.session_state.get("results")
        if results is not None:
            st.info(f"Scan complete. Found {len(results)} items.")
            for item in results:
                st.markdown(f"### 🔹 {item['category']}")
                st.caption(f"Found: \"{item['phrase']}\" [Page {item['page']}]")
                st.markdown(f"**Gap:** {item['gap']}")
                st.markdown(f"**Clarification:** {item['question']}")
                st.divider()
    else:
        st.write("Upload a document to begin.")
