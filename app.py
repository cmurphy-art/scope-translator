import streamlit as st
import pdfplumber
import google.generativeai as genai
import json
import re
import time

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Scope Translator V14.1 (Quota + Shape Safe)")

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("CRITICAL: Google API Key missing. Add it to Streamlit Secrets.")
    st.stop()

# --- THE SAFE LIBRARY ---
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

# --- UTILITIES ---

def normalize_text(text):
    """Lower + collapse whitespace for robust substring checks."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())

def split_text_into_chunks(text, max_chars=2000):
    """Chunk by paragraphs first to keep context."""
    chunks = []
    current = ""
    parts = text.split("\n\n")
    for p in parts:
        if len(current) + len(p) + 2 <= max_chars:
            current += p + "\n\n"
        else:
            if current.strip():
                chunks.append(current)
            current = p + "\n\n"
    if current.strip():
        chunks.append(current)
    return chunks

def get_best_available_model():
    """Prefer Flash, fallback to Pro."""
    try:
        available = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        for m in ["models/gemini-2.5-flash", "models/gemini-1.5-flash", "models/gemini-pro"]:
            if m in available:
                return m
        return available[0] if available else "models/gemini-pro"
    except:
        return "models/gemini-pro"

def clean_json_text(text):
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

def safe_load_json(text):
    try:
        return json.loads(clean_json_text(text))
    except:
        return None

def call_with_backoff(model, prompt, max_retries=4):
    """
    Quota-safe call:
    - If we hit 429, wait for the suggested retry delay if present, otherwise exponential backoff.
    - Returns response.text or None.
    """
    base_sleep = 2.0
    for attempt in range(max_retries + 1):
        try:
            resp = model.generate_content(prompt)
            return resp.text
        except Exception as e:
            msg = str(e)

            if "429" not in msg:
                # Non-quota error: surface once, then stop retrying
                raise

            # Try to parse "retry in Xs" from the error string
            m = re.search(r"retry (?:in|after)\s+([0-9.]+)\s*s", msg, re.IGNORECASE)
            if m:
                wait_s = float(m.group(1)) + 0.5
            else:
                wait_s = min(30.0, base_sleep * (2 ** attempt))

            time.sleep(wait_s)

    return None

# --- THE BRAIN ---
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
- Do NOT provide reasoning. Only provide the "classification" KEY.
- IGNORE performance verbs like "optimize", "maximize", "ensure".
- Return [] if nothing found.

OUTPUT FORMAT:
[{"trigger_text": "...", "classification": "UNDEFINED_BOUNDARY"}]
"""

def scan_document(pages_data):
    findings = []
    seen = set()

    # Optional pacing: keep it light, let backoff handle true 429s
    per_call_pause_s = 0.25

    active_model = get_best_available_model()
    model = genai.GenerativeModel(active_model)

    progress_bar = st.progress(0)
    total_chunks = sum(len(split_text_into_chunks(p["text"])) for p in pages_data) or 1
    done_chunks = 0

    # Debug counters (shown in UI)
    parse_failures = 0
    shape_failures = 0
    invalid_items = 0
    quote_misses = 0
    quota_retries = 0

    for page_obj in pages_data:
        page_num = page_obj["page"]
        raw_text = page_obj["text"]

        chunks = split_text_into_chunks(raw_text)

        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < 10:
                done_chunks += 1
                progress_bar.progress(min(done_chunks / total_chunks, 1.0))
                continue

            full_prompt = f"{SYSTEM_INSTRUCTION}\n\nTEXT TO ANALYZE:\n{chunk}"

            # Pace slightly
            time.sleep(per_call_pause_s)

            try:
                out_text = None
                try:
                    out_text = call_with_backoff(model, full_prompt, max_retries=4)
                except Exception as e:
                    # Surface unexpected errors
                    st.error(f"Error on Page {page_num}: {str(e)}")
                    done_chunks += 1
                    progress_bar.progress(min(done_chunks / total_chunks, 1.0))
                    continue

                if not out_text:
                    done_chunks += 1
                    progress_bar.progress(min(done_chunks / total_chunks, 1.0))
                    continue

                raw = safe_load_json(out_text)
                if raw is None:
                    parse_failures += 1
                    done_chunks += 1
                    progress_bar.progress(min(done_chunks / total_chunks, 1.0))
                    continue

                # SHAPE FIX: dict -> list
                if isinstance(raw, dict):
                    raw = [raw]

                if not isinstance(raw, list):
                    shape_failures += 1
                    done_chunks += 1
                    progress_bar.progress(min(done_chunks / total_chunks, 1.0))
                    continue

                norm_chunk = normalize_text(chunk)

                for item in raw:
                    if not isinstance(item, dict):
                        invalid_items += 1
                        continue

                    key = (item.get("classification") or "").strip()
                    quote = (item.get("trigger_text") or "").strip()

                    if not key or not quote:
                        invalid_items += 1
                        continue

                    # Trust anchor: quote must exist in the chunk
                    norm_quote = normalize_text(quote)
                    if norm_quote not in norm_chunk:
                        quote_misses += 1
                        continue

                    template = RESPONSE_TEMPLATES.get(key)
                    if not template:
                        invalid_items += 1
                        continue

                    unique_id = f"{page_num}|{key}|{norm_quote}"
                    if unique_id in seen:
                        continue
                    seen.add(unique_id)

                    findings.append({
                        "phrase": quote,
                        "category": template["category"],
                        "gap": template["gap"],
                        "question": template["question"],
                        "page": page_num
                    })

            finally:
                done_chunks += 1
                progress_bar.progress(min(done_chunks / total_chunks, 1.0))

    progress_bar.empty()

    # Minimal debug visibility (optional)
    st.caption(
        f"Debug: parse_fail={parse_failures} | shape_fail={shape_failures} | "
        f"invalid_items={invalid_items} | quote_miss={quote_misses}"
    )

    return findings

# --- UI ---
st.title("Scope Translator V14.1 (Quota + Shape Safe)")
st.markdown("**Ethos:** This tool identifies undefined conditions using strict, pre-defined neutral language.")
st.divider()

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("1. Source Document")
    uploaded_file = st.file_uploader("Upload Scope PDF", type="pdf")

    pages_data = []
    full_text_display = ""

    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                words = page.extract_words(x_tolerance=1)
                page_text = " ".join([w["text"] for w in words])
                page_text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", page_text)

                if page_text.strip():
                    pages_data.append({"page": i + 1, "text": page_text})
                    full_text_display += f"--- Page {i+1} ---\n{page_text}\n\n"

        st.text_area("Extracted Text Content", full_text_display, height=600)

with col2:
    st.subheader("2. Analysis")

    if "findings" not in st.session_state:
        st.session_state.findings = None

    if pages_data:
        if st.button("Run Strict Analysis"):
            with st.spinner("Applying Filters..."):
                st.session_state.findings = scan_document(pages_data)

        if st.session_state.findings is not None:
            results = st.session_state.findings
            st.info(f"**Scan Complete.** Found {len(results)} items.")

            for item in results:
                with st.container():
                    st.markdown(f"### 🔹 {item['category']}")
                    st.caption(f"**Found:** \"{item['phrase']}\" [Page {item['page']}]")
                    st.markdown(f"**Gap:** {item['gap']}")
                    st.markdown(f"**Clarification:** {item['question']}")
                    st.divider()
    else:
        st.write("Upload a document to begin.")
