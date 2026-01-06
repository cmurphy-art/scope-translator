import streamlit as st
import pdfplumber
import google.generativeai as genai
import json
import re

# --- CONFIGURATION: THE BRAIN ---

# Configure Gemini using the Secret Key from Streamlit
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.error("Missing Google API Key. Please add it to Streamlit Secrets.")

# The Safety System Prompt (Layer 2)
# This controls the "Personality" of the AI.
SYSTEM_INSTRUCTION = """
ROLE: You are a strict Scope Analyzer for construction documents. 
You are NOT an advisor. You DO NOT write emails.

TASK: Analyze the provided text snippet. Identify if it contains specific types of ambiguity based on the TAXONOMY below.

TAXONOMY:
1. UNDEFINED_BOUNDARY (Triggers: "match existing", "tie into", "patch", "repair", "connect to", "interface with")
2. SUBJECTIVE_QUALITY (Triggers: "industry standard", "workmanlike", "satisfaction of", "best practice", "clean and smooth")
3. UNDEFINED_SCOPE (Triggers: "complete system", "turnkey", "including but not limited to", "necessary for", "as required")
4. EXPLICIT_LIABILITY (Triggers: "liquidated damages", "time is of the essence", "indemnify", "hold harmless")
5. COORDINATION_GAP (Triggers: "coordinate with", "verify in field", "by others", "provided by owner")

CONSTRAINTS (NON-NEGOTIABLE):
- You must ONLY return a raw JSON list. No markdown formatting (```json).
- Extract the EXACT quote from the text.
- Do NOT provide advice or predict risk (e.g., never say "this could cause delays").
- If no ambiguity is found, return an empty list [].

OUTPUT FORMAT (JSON List):
[
  {
    "trigger_text": "install new vanity to match existing",
    "classification": "UNDEFINED_BOUNDARY",
    "reasoning": "The text requires matching but does not define the physical limit."
  }
]
"""

# --- LOGIC ENGINE ---

def analyze_chunk_with_gemini(text_chunk):
    """Sends text to Gemini Flash for classification."""
    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_INSTRUCTION,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # Send text to model
        response = model.generate_content(text_chunk)
        return json.loads(response.text)
    except Exception as e:
        return []

def scan_document(pages_data):
    """
    The Semantic Engine: Sends pages to LLM for analysis.
    """
    findings = []
    
    # Progress bar (because AI takes a second)
    progress_bar = st.progress(0)
    total_pages = len(pages_data)
    
    for i, page_obj in enumerate(pages_data):
        page_num = page_obj['page']
        text = page_obj['text']
        
        # Update progress
        progress_bar.progress((i + 1) / total_pages)
        
        # Call the LLM
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

st.set_page_config(layout="wide", page_title="Scope Translator (Semantic)")

# Header
st.title("Scope Translator (Semantic Mode)") 
st.markdown("""
**Ethos:** This tool identifies undefined conditions in the scope using AI classification.
It does not offer legal advice. It is designed to move the burden from the person to the document.
""")
st.divider()

# Split Screen Layout
col1, col2 = st.columns([1.5, 1])

# LEFT COLUMN: The Document
with col1:
    st.subheader("1. Source Document")
    uploaded_file = st.file_uploader("Upload Scope PDF", type="pdf")
    
    pages_data = [] 
    full_text_display = ""

    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                # NUCLEAR OPTION: x_tolerance=1 to fix squashed words
                words = page.extract_words(x_tolerance=1)
                page_text = ' '.join([w['text'] for w in words])
                page_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', page_text)
                
                if page_text:
                    pages_data.append({'page': i+1, 'text': page_text})
                    full_text_display += f"--- Page {i+1} ---\n{page_text}\n\n"
        
        st.text_area("Extracted Text Content", full_text_display, height=600)

# RIGHT COLUMN: The Translator
with col2:
    st.subheader("2. Analysis")
    
    # Session state to keep results after button click
    if "findings" not in st.session_state:
        st.session_state.findings = None

    if pages_data:
        # The Action Button (saves money/time by not auto-running)
        if st.button("Run Semantic Analysis"):
            with st.spinner("Consulting the Analyst..."):
                st.session_state.findings = scan_document(pages_data)
        
        if st.session_state.findings:
            results = st.session_state.findings
            st.info(f"**Scan Complete.** Found {len(results)} items requiring clarification.")
            
            for item in results:
                with st.container():
                    # Card Styling
                    st.markdown(f"### 🔹 {item['phrase']}")
                    st.caption(f"**Category:** {item['category']}")
                    
                    st.markdown(f"> *\"{item['snippet']}\"*")
                    st.markdown(f"**[Page {item['page']}]**")
                    
                    st.markdown(f"**Clarification:** {item['question']}")
                    st.divider()
    else:
        st.write("Upload a document to begin.")
