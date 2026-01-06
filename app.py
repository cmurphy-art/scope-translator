import streamlit as st
import pdfplumber
import google.generativeai as genai
import json
import re

# --- CONFIGURATION: THE BRAIN ---

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("CRITICAL ERROR: Google API Key is missing.")

SYSTEM_INSTRUCTION = """
ROLE: You are a strict Scope Analyzer.
TASK: Analyze the text. Return a raw JSON list of ambiguities based on the TAXONOMY.

TAXONOMY:
1. UNDEFINED_BOUNDARY (e.g. "match existing", "tie into")
2. SUBJECTIVE_QUALITY (e.g. "industry standard", "satisfaction of")
3. UNDEFINED_SCOPE (e.g. "turnkey", "including but not limited to")
4. EXPLICIT_LIABILITY (e.g. "liquidated damages", "time is of the essence")
5. COORDINATION_GAP (e.g. "coordinate with", "by others")

OUTPUT FORMAT:
Return ONLY a JSON list. Do not use Markdown formatting.
Example:
[{"trigger_text": "...", "classification": "...", "reasoning": "..."}]
"""

# --- LOGIC ENGINE ---

def clean_json_text(text):
    """Cleans the AI response to ensure it is valid JSON."""
    # Remove markdown code blocks like ```json ... ```
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```', '', text)
    return text.strip()

def analyze_chunk_with_gemini(text_chunk):
    """Sends text to Gemini Pro (Standard Engine)."""
    try:
        # We use 'gemini-pro' which is the most reliable model
        model = genai.GenerativeModel(model_name="gemini-pro")
        
        # Combine system instruction with text for Gemini Pro 1.0
        full_prompt = f"{SYSTEM_INSTRUCTION}\n\nANALYSIS TEXT:\n{text_chunk}"
        
        response = model.generate_content(full_prompt)
        
        # Clean and Parse
        clean_text = clean_json_text(response.text)
        return json.loads(clean_text)
        
    except Exception as e:
        # If it fails, we log it but keep going
        print(f"Error: {e}")
        return []

def scan_document(pages_data):
    findings = []
    progress_bar = st.progress(0)
    total_pages = len(pages_data)
    
    for i, page_obj in enumerate(pages_data):
        page_num = page_obj['page']
        text = page_obj['text']
        
        progress_bar.progress((i + 1) / total_pages)
        
        analysis_results = analyze_chunk_with_gemini(text)
        
        if analysis_results:
            for item in analysis_results:
                findings.append({
                    "phrase": item.get("classification", "Ambiguity").replace("_", " ").title(),
                    "category": "Detected by Semantic Analyst", 
                    "question": item.get("reasoning", "Clarification required."),
                    "snippet": item.get("trigger_text", "See text..."),
                    "page": page_num
                })
                
    progress_bar.empty()
    return findings

# --- USER INTERFACE (UI) ---

st.set_page_config(layout="wide", page_title="Scope Translator (Universal Mode)")

st.title("Scope Translator (Universal Mode)") 
st.markdown("**Ethos:** AI-Powered Scope Analysis (Powered by Gemini Pro).")
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
