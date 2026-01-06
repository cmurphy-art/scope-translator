import streamlit as st
import pdfplumber
import re

# --- CONFIGURATION: THE BRAIN ---
# This matches your Google Sheet logic exactly.

CONTEXT_TAGS = {
    "#FINISHES": ["paint", "drywall", "tile", "flooring", "millwork", "cabinetry", "carpet", "baseboard", "trim"],
    "#MEP": ["hvac", "plumbing", "electrical", "mechanical", "sprinkler", "lighting", "fixture", "switch", "outlet"],
    "#STRUCTURAL": ["concrete", "steel", "framing", "foundation", "masonry", "shoring", "rebar"],
    "#SITE_WORK": ["excavation", "grading", "utilities", "paving", "landscape", "demolition"],
    "#GENERAL_REQS": ["mobilization", "safety", "schedule", "supervision", "insurance", "permit"],
}

# The Trigger Library (Subset for Prototype)
TRIGGERS = [
    {
        "phrase": "match existing",
        "category": "Common Dispute Zone",
        "risk": "Undefined Physical Boundary",
        "question": "The document requires matching but does not currently define the physical boundary. Is the transition point the immediate area or the nearest corner?",
        "context_required": None
    },
    {
        "phrase": "coordinate with",
        "category": "Implied Responsibility",
        "risk": "Undefined Priority",
        "question": "The scope involves multiple trades. Does the document assign priority in this zone to prevent schedule stacking?",
        "context_required": ["#MEP", "#FINISHES"]
    },
    {
        "phrase": "industry standard",
        "category": "Implied Responsibility",
        "risk": "Subjective Quality Metric",
        "question": "The term 'standard' is subjective. Does the contract reference a specific AWI or TCNA grade for acceptance?",
        "context_required": None
    },
    {
        "phrase": "paint ready",
        "category": "Common Dispute Zone",
        "risk": "Ambiguous Surface Prep",
        "question": "Surface prep levels are currently undefined. Does 'paint ready' explicitly imply a Level 4 finish and primer?",
        "context_required": ["#FINISHES"]
    },
    {
        "phrase": "field verify",
        "category": "Explicit Commitment",
        "risk": "Liability for Design Errors",
        "question": "The drawings may pre-date current conditions. Does the scope provide a mechanism for adjustment if discrepancies are found?",
        "context_required": ["#STRUCTURAL", "#MEP"]
    },
    {
        "phrase": "by others",
        "category": "Implied Responsibility",
        "risk": "Boundary Gap",
        "question": "This item relies on a third party. Is the specific hand-off date and required condition defined elsewhere?",
        "context_required": None
    },
    {
        "phrase": "allowance for",
        "category": "Common Dispute Zone",
        "risk": "Budget Friction",
        "question": "An allowance is listed, but tax/labor inclusion is not specified. Are these costs included in the figure?",
        "context_required": None
    }
]

# --- LOGIC ENGINE ---

def determine_context(text_block):
    """Scans a text block to see which bucket it belongs to (MEP, Finishes, etc)."""
    text_lower = text_block.lower()
    detected_tags = []
    for tag, keywords in CONTEXT_TAGS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_tags.append(tag)
    return detected_tags

def scan_text(text):
    """The Deterministic Engine: Scans text for triggers based on rules."""
    findings = []
    
    # Simple chunking by newlines for this prototype
    paragraphs = text.split('\n\n') 
    
    for i, para in enumerate(paragraphs):
        if len(para.strip()) < 10: continue # Skip empty lines
        
        # 1. Identify Context
        tags = determine_context(para)
        
        # 2. Check Triggers
        for trigger in TRIGGERS:
            # Check if phrase is in text
            if trigger["phrase"] in para.lower():
                
                # Context Rule Check
                if trigger["context_required"]:
                    # If context is required, check if we found the right tag
                    if not any(t in tags for t in trigger["context_required"]):
                        continue # Skip if wrong context
                
                # Suppression Rule (Simple Version): Check for "Exclusions"
                if "exclusion" in para.lower() or "not included" in para.lower():
                    continue 

                # If we pass all checks, record the finding
                findings.append({
                    "phrase": trigger["phrase"],
                    "category": trigger["category"],
                    "risk": trigger["risk"],
                    "question": trigger["question"],
                    "snippet": para[:200] + "..." # First 200 chars for context
                })
    return findings

# --- USER INTERFACE (UI) ---

st.set_page_config(layout="wide", page_title="Prephase Scope Translator")

# Header
st.title("Scope Translator (Prototype)")
st.markdown("""
**Ethos:** This tool identifies undefined conditions in the scope. 
It does not offer legal advice. It is designed to move the burden from the person to the document.
""")
st.divider()

# Split Screen Layout
col1, col2 = st.columns([1.5, 1])

# LEFT COLUMN: The Document
with col1:
    st.subheader("1. Source Document")
    uploaded_file = st.file_uploader("Upload Scope PDF", type="pdf")
    
    pdf_text = ""
        if uploaded_file is not None:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    # Robust method: Extract words individually and join them
                    words = page.extract_words()
                    # Reconstruct text with spaces
                    page_text = ' '.join([w['text'] for w in words])
                    if page_text:
                        pdf_text += page_text + "\n\n"
        
        st.text_area("Extracted Text Content", pdf_text, height=600)

# RIGHT COLUMN: The Translator
with col2:
    st.subheader("2. Analysis")
    
    if pdf_text:
        results = scan_text(pdf_text)
        
        # Summary Header
        st.info(f"**Scan Complete.** Found {len(results)} items requiring clarification.")
        
        # Display Cards
        for item in results:
            with st.container():
                # Card Styling
                st.markdown(f"### 🚩 {item['phrase'].title()}")
                st.caption(f"**Category:** {item['category']}")
                
                # The "Tether" (Quote)
                st.markdown(f"> *\"{item['snippet']}\"*")
                
                # The Neutral Clarification
                st.markdown(f"**Clarification:** {item['question']}")
                
                st.divider()
    else:
        st.write("Upload a document to begin the audit.")
