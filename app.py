import streamlit as st
import pdfplumber
import re

# --- CONFIGURATION: THE BRAIN ---

CONTEXT_TAGS = {
    "#FINISHES": ["paint", "drywall", "tile", "flooring", "millwork", "cabinetry", "carpet", "baseboard", "trim"],
    "#MEP": ["hvac", "plumbing", "electrical", "mechanical", "sprinkler", "lighting", "fixture", "switch", "outlet"],
    "#STRUCTURAL": ["concrete", "steel", "framing", "foundation", "masonry", "shoring", "rebar"],
    "#SITE_WORK": ["excavation", "grading", "utilities", "paving", "landscape", "demolition"],
    "#GENERAL_REQS": ["mobilization", "safety", "schedule", "supervision", "insurance", "permit"],
}

# FULL LIBRARY (22 Triggers)
TRIGGERS = [
    {
        "phrase": "match existing",
        "category": "Common Clarification Area",
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
        "category": "Common Clarification Area",
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
        "category": "Common Clarification Area",
        "risk": "Budget Friction",
        "question": "An allowance is listed, but tax/labor inclusion is not specified. Are these costs included in the figure?",
        "context_required": None
    },
    {
        "phrase": "as directed by",
        "category": "Implied Responsibility",
        "risk": "Subjective Approval",
        "question": "The specific design criteria are not currently defined. Is there a mockup or 'not-to-exceed' spec that establishes the limit?",
        "context_required": None
    },
    {
        "phrase": "code compliant",
        "category": "Common Clarification Area",
        "risk": "Installer Liability",
        "question": "Design compliance is shifted to the installer here. Has this specific assembly been pre-vetted by the local jurisdiction?",
        "context_required": None
    },
    {
        "phrase": "including but not limited to",
        "category": "Common Clarification Area",
        "risk": "Open-Ended Scope",
        "question": "This phrase creates an indefinite scope. Is there a specific exclusion list that bounds this requirement?",
        "context_required": None
    },
    {
        "phrase": "at no additional cost",
        "category": "Explicit Commitment",
        "risk": "Unlimited Liability",
        "question": "This clause creates unlimited liability for undefined items. Is there a specific scope list that bounds this cost?",
        "context_required": None
    },
    {
        "phrase": "turnkey",
        "category": "Implied Responsibility",
        "risk": "Undefined Scope Limits",
        "question": "The term 'turnkey' is broad. Does the document specify boundaries regarding accessories, furniture, or final cleaning?",
        "context_required": None
    },
    {
        "phrase": "sole discretion",
        "category": "Common Clarification Area",
        "risk": "Subjective Acceptance",
        "question": "This phrase removes objective criteria. Is there an industry standard (AWI/TCNA) that can serve as the neutral benchmark?",
        "context_required": None
    },
    {
        "phrase": "satisfaction of architect",
        "category": "Common Clarification Area",
        "risk": "Subjective Acceptance",
        "question": "Acceptance is currently subjective. Is there a measurable standard for 'satisfaction' that replaces personal preference?",
        "context_required": None
    },
    {
        "phrase": "restore to original condition",
        "category": "Implied Responsibility",
        "risk": "Undefined Baseline",
        "question": "'Original condition' is subjective without a baseline. Is there a pre-construction photo report that establishes the standard?",
        "context_required": None
    },
    {
        "phrase": "continuous supervision",
        "category": "Explicit Commitment",
        "risk": "Resource Drain",
        "question": "'Continuous' implies a dedicated non-working role. Is this the intent, or is a working lead acceptable?",
        "context_required": None
    },
    {
        "phrase": "liquidated damages",
        "category": "Common Clarification Area",
        "risk": "Financial Penalty",
        "question": "Damages are listed. Does the contract also include a reciprocal bonus for early completion or excusable delays?",
        "context_required": None
    },
    {
        "phrase": "hazardous materials",
        "category": "Explicit Commitment",
        "risk": "Ambiguous Liability",
        "question": "Discovery protocols are undefined. Is the builder's responsibility limited to 'stop and notify' to ensure safety?",
        "context_required": None
    },
    {
        "phrase": "temporary protection",
        "category": "Implied Responsibility",
        "risk": "Undefined Cost",
        "question": "Protection requirements are broad. Is this a specific line-item allowance, or part of general conditions?",
        "context_required": ["#GENERAL_REQS", "#FINISHES"]
    },
    {
        "phrase": "permit fees",
        "category": "Explicit Commitment",
        "risk": "Cost Variance",
        "question": "Payment responsibility is unclear. Are these fees billable at cost, or included in the base contract sum?",
        "context_required": None
    },
    {
        "phrase": "accelerated schedule",
        "category": "Common Clarification Area",
        "risk": "Premium Labor Costs",
        "question": "The schedule implies pace beyond standard hours. Does the base bid include overtime premiums, or are those separate?",
        "context_required": None
    },
    {
        "phrase": "time is of the essence",
        "category": "Common Clarification Area",
        "risk": "Legal Exposure",
        "question": "This converts delays to breach. Does the contract account for excusable delays such as weather or supply chain?",
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

def scan_text(pages_data):
    """
    The Deterministic Engine: Scans text for triggers based on rules.
    Now accepts a list of page objects to track page numbers.
    """
    findings = []
    
    for page_obj in pages_data:
        page_num = page_obj['page']
        text = page_obj['text']
        
        # Simple chunking by newlines
        paragraphs = text.split('\n\n') 
        
        for para in paragraphs:
            if len(para.strip()) < 10: continue 
            
            # 1. Identify Context
            tags = determine_context(para)
            
            # 2. Check Triggers
            for trigger in TRIGGERS:
                if trigger["phrase"] in para.lower():
                    
                    # Context Rule Check
                    if trigger["context_required"]:
                        if not any(t in tags for t in trigger["context_required"]):
                            continue 
                    
                    # Suppression Rule
                    if "exclusion" in para.lower() or "not included" in para.lower():
                        continue 

                    # Record finding with Page Number
                    findings.append({
                        "phrase": trigger["phrase"],
                        "category": trigger["category"],
                        "risk": trigger["risk"],
                        "question": trigger["question"],
                        "snippet": para[:200] + "...",
                        "page": page_num
                    })
    return findings

# --- USER INTERFACE (UI) ---

st.set_page_config(layout="wide", page_title="Prephase Scope Translator")

# Header
st.title("Scope Translator V3") 
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
    
    pages_data = [] # List to store {page: 1, text: "..."}
    full_text_display = ""

    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            for i, page in enumerate(pdf.pages):
                # NUCLEAR OPTION FOR TEXT EXTRACTION
                words = page.extract_words(x_tolerance=0)
                page_text = ' '.join([w['text'] for w in words])
                page_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', page_text)
                
                if page_text:
                    # Store page data for logic engine
                    pages_data.append({'page': i+1, 'text': page_text})
                    # Store text for display
                    full_text_display += f"--- Page {i+1} ---\n{page_text}\n\n"
        
        st.text_area("Extracted Text Content", full_text_display, height=600)

# RIGHT COLUMN: The Translator
with col2:
    st.subheader("2. Analysis")
    
    if pages_data:
        results = scan_text(pages_data)
        
        # Summary Header
        st.info(f"**Scan Complete.** Found {len(results)} items requiring clarification.")
        
        # Display Cards
        for item in results:
            with st.container():
                # Card Styling - Neutral Icon
                st.markdown(f"### 🔹 {item['phrase'].title()}")
                st.caption(f"**Category:** {item['category']}")
                
                # The "Tether" (Quote) with Citation
                st.markdown(f"> *\"{item['snippet']}\"*")
                st.markdown(f"**[Page {item['page']}]**")
                
                # The Neutral Clarification
                st.markdown(f"**Clarification:** {item['question']}")
                
                st.divider()
    else:
        st.write("Upload a document to begin the audit.")
