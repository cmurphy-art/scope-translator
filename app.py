import streamlit as st
import pdfplumber
import google.generativeai as genai
import json
import re

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Scope Translator V12 (Strict)")

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("CRITICAL: Google API Key missing. Add it to Streamlit Secrets.")
    st.stop()

# --- THE SAFE LIBRARY (The Puppet Master) ---
# The AI selects the KEY. Python inserts the TEXT.
# This prevents the AI from ever using "scary" language.

RESPONSE_TEMPLATES = {
    "UNDEFINED_BOUNDARY": {
        "category": "Contractual Commitment",
        "gap": "The document requires matching or connecting but does not currently define the physical boundary.",
        "question": "Is the transition point the immediate area or the nearest corner?"
    },
    "SUBJECTIVE_QUALITY": {
        "category": "Implied Responsibility",
        "gap": "The term implies a quality standard without citing a specific metric.",
        "question": "Does the contract reference a specific AWI, TCNA, or industry grade for acceptance?"
    },
    "UNDEFINED_SCOPE": {
        "category": "Contractual Commitment",
        "gap": "The document implies a complete result ('turnkey') but does not list specific inclusions.",
        "question": "Does the document specify boundaries regarding accessories, furniture, or final cleaning?"
    },
    "EXPLICIT_LIABILITY": {
        "category": "Explicit Commitment",
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

# --- THE AUTO-PILOT ---
def get_best_available_model():
    """Finds the best model your key has access to."""
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        preferred_order = ["models/gemini-2.5-flash", "models/gemini-1.5-flash", "models/gemini-pro"]
        for model in preferred_order:
            if model in available_models: return model
        return available_models[0] if available_models else "models/gemini-pro"
    except:
        return "models/gemini-pro"

# --- THE BRAIN (Strict JSON Mode) ---
SYSTEM_INSTRUCTION = """
ROLE: You are a strict Classifier. You DO NOT write text. You only select keys.

TASK: Analyze the text snippet. Identify specific ambiguities using the KEYS below.

KEYS (Select the best fit):
1. UNDEFINED_BOUNDARY (Triggers: "match existing", "tie into", "patch", "repair")
2. SUBJECTIVE_QUALITY (Triggers: "industry standard", "workmanlike", "satisfaction of", "best practice")
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
- Return [] if nothing found.

OUTPUT FORMAT:
[{"trigger_text": "...", "classification": "UNDEFINED_BOUNDARY"}]
"""

def clean_json_text(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```', '', text)
    return text.strip()

def scan_document(pages_data):
    findings = []
    progress_bar = st.progress(0)
    
    active_model = get_best_available_model()
    model = genai.GenerativeModel(active_model)
    
    total_pages = len(pages_data)
    
    for i, page_obj in enumerate(pages_data):
        page_num = page_obj['page']
        text = page_obj['text']
        progress_bar.progress((i + 1) / total_pages)
        
        try:
            full_prompt = f"{SYSTEM_INSTRUCTION}\n\nTEXT TO ANALYZE:\n{text}"
            response = model.generate_content(full_prompt)
            data = json.loads(clean_json_text(response.text))
            
            if data:
                for item in data:
                    # THE FIREWALL: We only use the KEY from AI to look up OUR text
                    key = item.get("classification")
                    quote = item.get("trigger_text")
                    
                    if key in RESPONSE_TEMPLATES:
                        template = RESPONSE_TEMPLATES[key]
                        findings.append({
                            "phrase": quote,  # The actual text from doc
                            "category": template["category"], # OUR Safe Category
                            "gap": template["gap"], # OUR Safe Gap
                            "question": template["question"], # OUR Safe Question
                            "page": page_num
                        })
        except Exception:
            continue
                
    progress_bar.empty()
    return findings

# --- USER INTERFACE ---
st.title("Scope Translator V12 (Strict)") 
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
                # NUCLEAR OPTION (Verified Spacing)
                words = page.extract_words(x_tolerance=1)
                page_text = ' '.join([w['text'] for w in words])
                page_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', page_text)
                
                if page_text:
                    pages_data.append({'page': i+1, 'text': page_text})
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
        
        if st.session_state.findings:
            results = st.session_state.findings
            st.info(f"**Scan Complete.** Found {len(results)} items.")
            
            for item in results:
                with st.container():
                    # V11 CARD LAYOUT
                    st.markdown(f"### 🔹 {item['category']}") 
                    st.caption(f"**Found:** \"{item['phrase']}\" [Page {item['page']}]")
                    
                    st.markdown(f"**Gap:** {item['gap']}")
                    st.markdown(f"**Clarification:** {item['question']}")
                    st.divider()
    else:
        st.write("Upload a document to begin.")
