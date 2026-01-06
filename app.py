import streamlit as st
import pdfplumber
import google.generativeai as genai
import json
import re

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Scope Translator (Semantic)")

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("CRITICAL: Google API Key missing. Add it to Streamlit Secrets.")
    st.stop()

# --- THE AUTO-PILOT (Crucial for preventing 404 errors) ---
def get_best_available_model():
    """Finds the best model your key has access to."""
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Priority list (Newest/Fastest first)
        preferred_order = [
            "models/gemini-2.5-flash",
            "models/gemini-1.5-flash", 
            "models/gemini-pro",
        ]
        
        for model in preferred_order:
            if model in available_models:
                return model
        
        # Fallback to whatever is available
        return available_models[0] if available_models else None
    except:
        return "models/gemini-pro" # Blind fallback

# --- THE BRAIN ---
SYSTEM_INSTRUCTION = """
ROLE: You are a strict Scope Analyzer for construction documents.
TASK: Analyze the text snippet. Identify specific ambiguities using the TAXONOMY below.

TAXONOMY:
1. UNDEFINED_BOUNDARY (Triggers: "match existing", "tie into", "if possible", "as required")
2. SUBJECTIVE_QUALITY (Triggers: "industry standard", "workmanlike", "satisfaction of", "best practice")
3. UNDEFINED_SCOPE (Triggers: "turnkey", "including but not limited to", "complete system")
4. EXPLICIT_LIABILITY (Triggers: "liquidated damages", "time is of the essence", "indemnify")
5. COORDINATION_GAP (Triggers: "coordinate with", "verify in field", "by others")

CONSTRAINTS:
- Return ONLY a raw JSON list.
- Extract the EXACT quote.
- Do NOT provide advice.
- Return [] if nothing found.

OUTPUT FORMAT:
[{"trigger_text": "...", "classification": "...", "reasoning": "..."}]
"""

def clean_json_text(text):
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```', '', text)
    return text.strip()

def scan_document(pages_data):
    findings = []
    progress_bar = st.progress(0)
    
    # Select Model Dynamically
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
                    findings.append({
                        "phrase": item.get("classification", "Ambiguity").replace("_", " ").title(),
                        "category": "Detected by Semantic Analyst", 
                        "question": item.get("reasoning", "Clarification required."),
                        "snippet": item.get("trigger_text", "See text..."),
                        "page": page_num
                    })
        except Exception as e:
            print(f"Page {page_num} Error: {e}")
            continue
                
    progress_bar.empty()
    return findings

# --- USER INTERFACE ---
st.title("Scope Translator (Semantic Mode)") 
st.markdown("**Ethos:** This tool uses AI to identify undefined conditions. It moves the burden from the person to the document.")
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
        if st.button("Run Semantic Analysis"):
            with st.spinner("Consulting the Analyst..."):
                st.session_state.findings = scan_document(pages_data)
        
        if st.session_state.findings:
            results = st.session_state.findings
            st.info(f"**Scan Complete.** Found {len(results)} items requiring clarification.")
            
            for item in results:
                with st.container():
                    st.markdown(f"### 🔹 {item['phrase']}")
                    st.caption(f"**Category:** {item['category']}")
                    st.markdown(f"> *\"{item['snippet']}\"*")
                    st.markdown(f"**[Page {item['page']}]**")
                    st.markdown(f"**Clarification:** {item['question']}")
                    st.divider()
    else:
        st.write("Upload a document to begin.")
