import streamlit as st
import pdfplumber
import google.generativeai as genai
import json
import re

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Prephase Scope Auditor")

# --- API KEY (Gemini) ---
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("CRITICAL: Google API Key missing. Add it to Streamlit Secrets.")
    st.stop()

# --- SAFE LIBRARY (TEMPLATES) ---
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

# --- MODEL PICKER (UNCHANGED DEFAULTS) ---
def get_best_available_model():
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        preferred_order = ["models/gemini-2.5-flash", "models/gemini-1.5-flash", "models/gemini-pro"]
        for model in preferred_order:
            if model in available_models:
                return model
        return available_models[0] if available_models else "models/gemini-pro"
    except:
        return "models/gemini-pro"

# --- BRAIN PROMPT (UNCHANGED) ---
SYSTEM_INSTRUCTION = """
ROLE: You are a strict Classifier. You DO NOT write text. You only select keys.

TASK: Analyze the text snippet. Identify specific ambiguities using the KEYS below.

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
- Extract the EXACT quote from the snippet.
- Do NOT provide reasoning. Only provide the "classification" KEY.
- IGNORE performance verbs like "optimize", "maximize", "ensure".
- Return [] if nothing found.

OUTPUT FORMAT:
[{"trigger_text": "...", "classification": "UNDEFINED_BOUNDARY"}]
"""

# --- UTILITIES ---
def clean_json_text(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```', '', text)
    return text.strip()

def normalize_text(text):
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip().lower())

def split_text_into_chunks(text, max_chars=2000):
    chunks = []
    current_chunk = ""
    paragraphs = text.split('\n\n')

    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# --- SCAN ENGINE (UNCHANGED LOGIC; UI STATE ADDED) ---
def scan_document(pages_data, model_name):
    findings = []
    seen_hashes = set()

    model = genai.GenerativeModel(model_name)

    total_pages = len(pages_data)
    progress_bar = st.progress(0)

    for i, page_obj in enumerate(pages_data):
        page_num = page_obj["page"]
        raw_text = page_obj["text"]
        progress_bar.progress((i + 1) / total_pages)

        chunks = split_text_into_chunks(raw_text)

        for chunk in chunks:
            if len(chunk.strip()) < 10:
                continue

            try:
                full_prompt = f"{SYSTEM_INSTRUCTION}\n\nTEXT TO ANALYZE:\n{chunk}"
                response = model.generate_content(full_prompt)

                try:
                    data = json.loads(clean_json_text(response.text))
                except:
                    continue

                if not data:
                    continue

                for item in data:
                    key = item.get("classification")
                    quote = item.get("trigger_text")

                    if not quote or not key:
                        continue

                    # Trust anchor
                    norm_quote = normalize_text(quote)
                    norm_chunk = normalize_text(chunk)
                    if norm_quote not in norm_chunk:
                        continue

                    unique_id = f"{page_num}-{norm_quote}-{key}"
                    if unique_id in seen_hashes:
                        continue

                    if key in RESPONSE_TEMPLATES:
                        t = RESPONSE_TEMPLATES[key]
                        seen_hashes.add(unique_id)
                        findings.append({
                            "id": unique_id,
                            "phrase": quote,
                            "category": t["category"],
                            "gap": t["gap"],
                            "question": t["question"],
                            "page": page_num
                        })

            except Exception:
                continue

    progress_bar.empty()
    return findings

# --- SESSION STATE ---
if "findings" not in st.session_state:
    st.session_state.findings = None

if "active_finding" not in st.session_state:
    st.session_state.active_finding = None

if "reviewed_map" not in st.session_state:
    st.session_state.reviewed_map = {}  # finding_id -> bool

# --- UI ---
st.title("Prephase Scope Auditor")
st.markdown("**Ethos:** Move the burden from the person to the document.")
st.divider()

# Controls row (model + scan)
controls_left, controls_right = st.columns([1.2, 1])

with controls_left:
    st.subheader("1. Source Document")
    uploaded_file = st.file_uploader("Upload Scope PDF", type="pdf")

with controls_right:
    st.subheader("2. Run Analysis")
    default_model = get_best_available_model()
    model_name = st.selectbox(
        "Model",
        options=[default_model, "models/gemini-2.5-flash", "models/gemini-1.5-flash", "models/gemini-pro"],
        index=0
    )

pages_data = []
full_text_display = ""

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            # Your current extraction approach (kept)
            words = page.extract_words(x_tolerance=1)
            page_text = " ".join([w["text"] for w in words])
            page_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', page_text)

            if page_text:
                pages_data.append({"page": i + 1, "text": page_text})

# Two-column main layout (PDF left, findings right)
col_left, col_right = st.columns([1.6, 1])

with col_right:
    if pages_data:
        if st.button("Run Audit"):
            with st.spinner("Applying filters..."):
                st.session_state.findings = scan_document(pages_data, model_name)

                # Initialize reviewed map for new findings (do not wipe existing checked items if IDs match)
                for f in (st.session_state.findings or []):
                    st.session_state.reviewed_map.setdefault(f["id"], False)

                # Reset active selection on new scan
                st.session_state.active_finding = None

# LEFT: Extracted text with highlight of active finding
with col_left:
    st.subheader("Extracted Text")

    if pages_data:
        highlighted_text = ""

        active = st.session_state.active_finding
        active_phrase = active["phrase"] if active else None
        active_page = active["page"] if active else None

        for page in pages_data:
            page_header = f"\n\n--- Page {page['page']} ---\n"
            page_text = page["text"]

            if active_phrase and page["page"] == active_page and active_phrase in page_text:
                # Simple, readable highlight marker
                page_text = page_text.replace(active_phrase, f"🔶 {active_phrase} 🔶")

            highlighted_text += page_header + page_text

        st.text_area("PDF Text", highlighted_text, height=750)
    else:
        st.info("Upload a PDF to see extracted text.")

# RIGHT: Findings panel fixed height with internal scroll + checkboxes
with col_right:
    st.subheader("Findings")

    results = st.session_state.findings or []
    if results:
        reviewed_count = sum(1 for f in results if st.session_state.reviewed_map.get(f["id"], False))
        remaining = len(results) - reviewed_count

        m1, m2, m3 = st.columns(3)
        m1.metric("Total", len(results))
        m2.metric("Reviewed", reviewed_count)
        m3.metric("Remaining", remaining)

        show_reviewed_only = st.checkbox("Show reviewed only", value=False)

        with st.container(height=750):
            for idx, item in enumerate(results):
                fid = item["id"]
                is_reviewed = st.session_state.reviewed_map.get(fid, False)

                if show_reviewed_only and not is_reviewed:
                    continue

                # Click target (sets active finding)
                if st.button(
                    f"{item['category']}  •  Page {item['page']}",
                    key=f"select_{fid}"
                ):
                    st.session_state.active_finding = item

                # Checkbox (reviewed state)
                checked = st.checkbox(
                    "Reviewed",
                    value=is_reviewed,
                    key=f"review_{fid}"
                )
                st.session_state.reviewed_map[fid] = checked

                st.caption(f"**Found:** “{item['phrase']}”")
                st.markdown(f"**Gap:** {item['gap']}")
                st.markdown(f"**Clarification:** {item['question']}")
                st.divider()

    elif pages_data:
        st.write("Run the audit to see findings.")
    else:
        st.write("Upload a document to begin.")
