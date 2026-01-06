import streamlit as st
import pdfplumber
import google.generativeai as genai
import json
import re
import hashlib
from collections import Counter

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Prephase Scope Translator (V11)")

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("CRITICAL: Google API Key missing. Add it to Streamlit Secrets.")
    st.stop()

# --- THE DOCUMENT-CENTERED LIBRARY (V11) ---
TEMPLATE_LIBRARY = {
    # COMMON CLARIFICATION AREA
    "match existing": {
        "gap": "The document requires matching but does not define the physical boundary.",
        "question": "Does the document specify if the transition point is the immediate area or the nearest corner?"
    },
    "tie into": {
        "gap": "The document requires a connection but does not define the method or extent.",
        "question": "Does the document specify the specific point of connection and the required assembly?"
    },
    "turnkey": {
        "gap": "The term 'turnkey' is used without a specific definition of scope limits.",
        "question": "Does the document explicitly include or exclude accessories, final cleaning, and furniture?"
    },
    "including but not limited to": {
        "gap": "The scope is described as open-ended.",
        "question": "Is there a specific exclusion list that limits this requirement?"
    },
    "as required": {
        "gap": "The document requires work 'as required' without a defined quantity or standard.",
        "question": "Is there a 'not-to-exceed' quantity or specific performance standard defined?"
    },
    
    # IMPLIED RESPONSIBILITY
    "coordinate with": {
        "gap": "The document requires coordination but does not assign priority.",
        "question": "Does the schedule assign priority between trades to prevent stacking?"
    },
    "by others": {
        "gap": "The document references work by others without defining the handoff condition.",
        "question": "Is the specific handoff date and required condition defined?"
    },
    "industry standard": {
        "gap": "The document uses a subjective quality metric.",
        "question": "Does the contract reference a specific AWI/TCNA grade or measurable tolerance?"
    },
    "workmanlike": {
        "gap": "The document uses a subjective acceptance standard.",
        "question": "Is there a measurable standard (e.g. Level 4) for acceptance?"
    },

    # EXPLICIT COMMITMENT (OPERATIONAL)
    "verify in field": {
        "gap": "The document requires field verification. The mechanism for handling discrepancies is not specified.",
        "question": "Does the document define the process for adjustment if field conditions differ from plans?"
    },
    "field measure": {
        "gap": "The document requires field measurement. The schedule impact is not defined.",
        "question": "Does the schedule allow time for measurement prior to fabrication?"
    },
    "continuous supervision": {
        "gap": "The document requires continuous supervision. The staffing definition is not specified.",
        "question": "Does this require a non-working superintendent, or is a working lead acceptable?"
    },

    # CONTRACTUAL COMMITMENT (LEGAL - NEUTRALIZED)
    "liquidated damages": {
        "gap": "A liquidated damages clause is present. The document does not specify the triggering conditions.",
        "question": "Does the document specify the start condition, measurement method, and any stated exceptions?"
    },
    "time is of the essence": {
        "gap": "The document uses strict timing language. Delay categories or exceptions are not defined in this excerpt.",
        "question": "Does the document define any exceptions, notice requirements, or owner-caused delay handling?"
    },
    "indemnify": {
        "gap": "An indemnification obligation is present. The document does not define the scope limits in this excerpt.",
        "question": "Does the document define the scope of indemnity and any stated limits or exclusions?"
    }
}

GENERIC_TEMPLATES = {
    "Common Clarification Area": {
        "gap": "A condition is listed but currently undefined.",
        "question": "Does the document specify the exact physical boundary or standard?"
    },
    "Implied Responsibility": {
        "gap": "A responsibility is implied but not detailed.",
        "question": "Is the specific limit of this responsibility defined in the specs?"
    },
    "Explicit Commitment": {
        "gap": "An operational requirement is listed. The parameters are not specified.",
        "question": "Does the document define the specific execution requirements for this item?"
    },
    "Contractual Commitment": {
        "gap": "A contractual obligation is present. The document does not specify scope limits in this excerpt.",
        "question": "Does the document define the scope, limits, and conditions of this obligation?"
    }
}

ALLOWED_TYPES = {
    "Contractual Commitment",
    "Explicit Commitment",
    "Implied Responsibility",
    "Common Clarification Area"
}

CATEGORY_ORDER = {
    "Contractual Commitment": 0,
    "Explicit Commitment": 1,
    "Implied Responsibility": 2,
    "Common Clarification Area": 3
}

# --- OPTIMIZATION HELPERS ---

def normalize_trigger(text):
    """For clean dictionary lookups."""
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s-]", "", t) 
    return t

# Pre-compute normalized keys for fast matching
TEMPLATE_KEYS_NORM = [(normalize_trigger(k), v) for k, v in TEMPLATE_LIBRARY.items()]

# Flat list for fast pre-scanning (Synced with Prompt)
PRE_SCAN_TRIGGERS = list(TEMPLATE_LIBRARY.keys()) + [
    "warrant", "guarantee", "bond", "best practice", "subject to approval", 
    "submit daily reports"
]

SYSTEM_INSTRUCTION = """
ROLE: You are a strict Scope Auditor. 
Your ONLY job is to extract text and classify it. You do NOT offer advice.

TAXONOMY (STRICT):
1. "Contractual Commitment" 
   (Triggers: "liquidated damages", "time is of the essence", "indemnify", "warrant", "guarantee", "bond")
   
2. "Explicit Commitment"
   (Triggers: "verify in field", "field measure", "continuous supervision", "submit daily reports")

3. "Implied Responsibility"
   (Triggers: "coordinate with", "by others", "industry standard", "workmanlike", "best practice")
   
4. "Common Clarification Area"
   (Triggers: "match existing", "tie into", "turnkey", "including but not limited to", "as required", "subject to approval")

OUTPUT FORMAT (JSON ARRAY ONLY):
[
  {
    "exact_quote": "Copy the text EXACTLY from the source.",
    "finding_type": "One of the 4 Taxonomy categories above",
    "trigger_phrase": "The specific 2-4 word phrase that triggered this (e.g. 'match existing', 'verify in field')"
  }
]
"""

# --- LOGIC ENGINE ---

def get_best_available_model():
    try:
        available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        priorities = ["models/gemini-2.5-flash", "models/gemini-1.5-flash", "models/gemini-pro"]
        for p in priorities:
            if p in available: return p
        return available[0] if available else "models/gemini-pro"
    except:
        return "models/gemini-pro"

def clean_json_text(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```', '', text)
    return text.strip()

def safe_load_json(text):
    try:
        return json.loads(clean_json_text(text))
    except:
        return []

def normalize_text(text):
    """For rigorous quote matching."""
    text = text.lower()
    text = re.sub(r"[“”\"']", "", text)
    text = re.sub(r"[-–—]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_has_triggers(chunk):
    """Fast Pre-Scan: Checks if chunk contains any triggers before calling API."""
    norm_chunk = normalize_trigger(chunk) # Use same norm logic as triggers
    for t in PRE_SCAN_TRIGGERS:
        if normalize_trigger(t) in norm_chunk:
            return True
    return False

def generate_id(finding_type, quote, page):
    raw = f"{finding_type}|p{page}|{normalize_text(quote)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def apply_template(finding):
    raw_trigger = finding.get("trigger_phrase", "")
    trigger = normalize_trigger(raw_trigger)
    category = finding.get("finding_type")
    
    # 0. Guard against empty triggers
    if not trigger:
        fallback = GENERIC_TEMPLATES.get(category, GENERIC_TEMPLATES["Common Clarification Area"])
        finding["document_gap"] = fallback["gap"]
        finding["clarification_question"] = fallback["question"]
        return finding
    
    # 1. Bi-Directional Fuzzy Match using Pre-computed Keys
    for key_norm, temp in TEMPLATE_KEYS_NORM:
        if key_norm in trigger or trigger in key_norm:
            finding["document_gap"] = temp["gap"]
            finding["clarification_question"] = temp["question"]
            return finding
            
    # 2. Category Fallback
    fallback = GENERIC_TEMPLATES.get(category, GENERIC_TEMPLATES["Common Clarification Area"])
    finding["document_gap"] = fallback["gap"]
    finding["clarification_question"] = fallback["question"]
            
    return finding

def chunk_text(text, chunk_size=2500):
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    if len(paragraphs) < 2:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            if len(current_chunk) + len(sent) < chunk_size:
                current_chunk += sent + " "
            else:
                chunks.append(current_chunk)
                current_chunk = sent + " "
        if current_chunk: chunks.append(current_chunk)
        return chunks

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk.strip(): chunks.append(current_chunk)
            current_chunk = para + "\n\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk)
    return chunks

def scan_document(pages_data):
    findings = []
    seen_hashes = set()
    
    active_model = get_best_available_model()
    model = genai.GenerativeModel(
        active_model,
        generation_config={
            "temperature": 0.0, 
            "top_p": 0.1,
            "max_output_tokens": 1024
        }
    )
    
    progress_bar = st.progress(0)
    total_steps = len(pages_data)
    
    for i, page_obj in enumerate(pages_data):
        page_num = page_obj['page']
        text = page_obj['text']
        progress_bar.progress((i + 1) / total_steps)
        
        chunks = chunk_text(text)
        
        for chunk in chunks:
            if not chunk_has_triggers(chunk):
                continue

            # Normalize chunk for model consumption & validation
            chunk_for_model = re.sub(r"\s+", " ", chunk).strip()
            
            try:
                full_prompt = f"{SYSTEM_INSTRUCTION}\n\nTEXT TO AUDIT:\n{chunk_for_model}"
                response = model.generate_content(full_prompt)
                
                raw_data = safe_load_json(response.text)
                if isinstance(raw_data, dict): raw_data = [raw_data]
                
                if raw_data:
                    for item in raw_data:
                        # 1. Type Validation
                        ft = item.get("finding_type", "")
                        if ft not in ALLOWED_TYPES:
                            continue

                        # 2. Quote Validation (Match against what model saw)
                        quote = item.get("exact_quote", "")
                        if normalize_text(quote) not in normalize_text(chunk_for_model):
                            continue 
                            
                        # 3. Apply Template
                        item = apply_template(item)
                        
                        # 4. Dedupe
                        unique_id = generate_id(item['finding_type'], quote, page_num)
                        if unique_id not in seen_hashes:
                            findings.append({
                                "type": item['finding_type'],
                                "gap": item['document_gap'],
                                "question": item['clarification_question'],
                                "quote": quote,
                                "page": page_num
                            })
                            seen_hashes.add(unique_id)
            except Exception as e:
                print(f"Error on Page {page_num}: {e}")
                continue
                
    progress_bar.empty()
    
    sorted_findings = sorted(findings, key=lambda x: (CATEGORY_ORDER.get(x["type"], 99), x["page"]))
    return sorted_findings

# --- USER INTERFACE ---
st.title("Prephase Scope Translator") 
st.markdown("**Ethos:** Move the burden from the person to the document.")
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
                text = page.extract_text()
                if not text or len(text) < 50:
                    words = page.extract_words(x_tolerance=1)
                    text = ' '.join([w['text'] for w in words])
                    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
                
                if text:
                    pages_data.append({'page': i+1, 'text': text})
                    full_text_display += f"--- Page {i+1} ---\n{text}\n\n"
        
        st.text_area("Extracted Text Content", full_text_display, height=600)

with col2:
    st.subheader("2. Audit Results")
    
    if "findings" not in st.session_state:
        st.session_state.findings = None

    if pages_data:
        if st.button("Run Audit"):
            with st.spinner("Scanning for undefined conditions..."):
                st.session_state.findings = scan_document(pages_data)
        
        if st.session_state.findings:
            results = st.session_state.findings
            
            # Counts Dashboard
            counts = Counter([item['type'] for item in results])
            st.info(f"**Audit Complete.** Found {len(results)} items.")
            
            # Display Count Metrics
            cols = st.columns(4)
            cols[0].metric("Contractual", counts.get("Contractual Commitment", 0))
            cols[1].metric("Explicit", counts.get("Explicit Commitment", 0))
            cols[2].metric("Implied", counts.get("Implied Responsibility", 0))
            cols[3].metric("Clarification", counts.get("Common Clarification Area", 0))
            
            st.divider()

            for item in results:
                with st.container():
                    st.markdown(f"### 🔹 {item['type']}")
                    st.markdown(f"> *\"{item['quote']}\"*")
                    st.caption(f"**Page {item['page']}**")
                    st.markdown(f"**Gap:** {item['gap']}")
                    st.markdown(f"**Clarification:** {item['question']}")
                    st.divider()
    else:
        st.write("Upload a document to begin.")
