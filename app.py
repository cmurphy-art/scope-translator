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

# SMART PATTERN LIBRARY (Regex Enabled)
# We use 'pattern' instead of 'phrase' to allow flexible matching.
TRIGGERS = [
    {
        "label": "Match Existing",
        "pattern": r"match(?:ing|es)?\s+(?:the\s+)?(?:existing|adjacent|original|new|prevailing)",
        "category": "Common Clarification Area",
        "question": "The document requires matching but does not currently define the physical boundary. Is the transition point the immediate area or the nearest corner?",
        "context_required": None
    },
    {
        "label": "Coordinate With",
        "pattern": r"coordinat(?:e|ion)\s+(?:with|between|of)",
        "category": "Implied Responsibility",
        "question": "The scope involves multiple trades. Does the document assign priority in this zone to prevent schedule stacking?",
        "context_required": ["#MEP", "#FINISHES"]
    },
    {
        "label": "Industry Standard",
        "pattern": r"industry\s+standard",
        "category": "Implied Responsibility",
        "question": "The term 'standard' is subjective. Does the contract reference a specific AWI or TCNA grade for acceptance?",
        "context_required": None
    },
    {
        "label": "Paint Ready",
        "pattern": r"(?:paint\s+ready|ready\s+for\s+paint)",
        "category": "Common Clarification Area",
        "question": "Surface prep levels are currently undefined. Does 'paint ready' explicitly imply a Level 4 finish and primer?",
        "context_required": ["#FINISHES"]
    },
    {
        "label": "Field Verify",
        "pattern": r"field\s+verify",
        "category": "Explicit Commitment",
        "question": "The drawings may pre-date current conditions. Does the scope provide a mechanism for adjustment if discrepancies are found?",
        "context_required": ["#STRUCTURAL", "#MEP"]
    },
    {
        "label": "By Others",
        "pattern": r"by\s+others",
        "category": "Implied Responsibility",
        "question": "This item relies on a third party. Is the specific hand-off date and required condition defined elsewhere?",
        "context_required": None
    },
    {
        "label": "Allowance",
        "pattern": r"allowance",
        "category": "Common Clarification Area",
        "question": "An allowance is listed, but tax/labor inclusion is not specified. Are these costs included in the figure?",
        "context_required": None
    },
    {
        "label": "Subjective Approval",
        "pattern": r"(?:as\s+directed\s+by|at\s+direction\s+of)",
        "category": "Implied Responsibility",
        "question": "The specific design criteria are not currently defined. Is there a mockup or 'not-to-exceed' spec that establishes the limit?",
        "context_required": None
    },
    {
        "label": "Code Compliant",
        "pattern": r"code\s+complian(?:t|ce)",
        "category": "Common Clarification Area",
        "question": "Design compliance is shifted to the installer here. Has this specific assembly been pre-vetted by the local jurisdiction?",
        "context_required": None
    },
    {
        "label": "Open-Ended Scope",
        "pattern": r"including\s+but\s+not\s+limited\s+to",
        "category": "Common Clarification Area",
        "question": "This phrase creates an indefinite scope. Is there a specific exclusion list that bounds this requirement?",
        "context_required": None
    },
    {
        "label": "No Additional Cost",
        "pattern": r"at\s+no\s+additional\s+cost",
        "category": "Explicit Commitment",
        "question": "This clause creates unlimited liability for undefined items. Is there a specific scope list that bounds this cost?",
        "context_required": None
    },
    {
        "label": "Turnkey",
        "pattern": r"turnkey",
        "category": "Implied Responsibility",
        "question": "The term 'turnkey' is broad. Does the document specify boundaries regarding accessories, furniture, or final cleaning?",
        "context_required": None
    },
    {
        "label": "Sole Discretion",
        "pattern": r"sole\s+discretion",
        "category": "Common Clarification Area",
        "question": "This phrase removes objective criteria. Is there an industry standard (AWI/TCNA) that can serve as the neutral benchmark?",
        "context_required": None
    },
    {
        "label": "Satisfaction of...",
        "pattern": r"(?:satisfaction\s+of|satisfactory\s+to|discretion\s+of)",
        "category": "Common Clarification Area",
        "question": "Acceptance is currently subjective. Is there a measurable standard for 'satisfaction' that replaces personal preference?",
        "context_required": None
    },
    {
        "label": "Restore Condition",
        "pattern": r"restore\s+to\s+(?:original|previous)\s+condition",
        "category": "Implied Responsibility",
        "question": "'Original condition' is subjective without a baseline. Is there a pre-construction photo report that establishes the standard?",
        "context_required": None
    },
    {
        "label": "Continuous Supervision",
        "pattern": r"continuous\s+supervision",
        "category": "Explicit Commitment",
        "question": "'Continuous' implies a dedicated non-working role. Is this the intent, or is a working lead acceptable?",
        "context_required": None
    },
    {
        "label": "Liquidated Damages",
        "pattern": r"liquidated\s+damages",
        "category": "Common Clarification Area",
        "question": "Damages are listed. Does the contract also include a reciprocal bonus for early completion or excusable delays?",
        "context_required": None
    },
    {
        "label": "Hazardous Materials",
        "pattern": r"hazardous\s+materials?",
        "category": "Explicit Commitment",
        "question": "Discovery protocols are undefined. Is the builder's responsibility limited to 'stop and notify' to ensure safety?",
        "context_required": None
    },
    {
        "label": "Temporary Protection",
        "pattern": r"temporary\s+protection",
        "category": "Implied Responsibility",
        "question": "Protection requirements are broad. Is this a specific line-item allowance, or part of general conditions?",
        "context_required": ["#GENERAL_REQS", "#FINISHES"]
    },
    {
        "label": "Permit Fees",
        "pattern": r"permit\s+fees?",
        "category": "Explicit Commitment",
        "question": "Payment responsibility is unclear. Are these fees billable at cost, or included in the base contract sum?",
        "context_required": None
    },
    {
        "label": "Accelerated Schedule",
        "pattern": r"accelerated\s+schedule",
        "category": "Common Clarification Area",
        "question": "The schedule implies pace beyond standard hours. Does the base bid include overtime premiums, or are those separate?",
        "context_required": None
    },
    {
        "label": "Time is of the Essence",
        "pattern": r"time\s+is\s+of\s+the\s+essence",
        "category": "Common Clarification Area",
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
    The Smart Pattern Engine: Scans text for Regex patterns.
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
            
            # 2. Check Triggers using Regex
            for trigger in TRIGGERS:
                # re.search allows for fuzzy matching based on the pattern
                match = re.search(trigger["pattern"], para, re.IGNORECASE)
                
                if match:
                    # Context Rule Check
                    if trigger["context_required"]:
                        if not any(t in tags for t in trigger["context_required"]):
                            continue 
                    
                    # Suppression Rule
                    if "exclusion" in para.lower() or "not included" in para.lower():
                        continue 

                    # Record finding with Page Number
                    findings.append({
                        "phrase": trigger["label"], # Use label for clean UI
                        "category": trigger["category"],
                        "risk": "Risk Detected", # Generic placeholder, UI uses question
                        "question": trigger["question"],
                        "snippet": para[:200] + "...",
                        "page": page_num
                    })
    return findings

# --- USER INTERFACE (UI) ---

st.set_page_config(layout="wide", page_title="Scope Translator Smart Mode")

# Header
st.title("Scope Translator (Smart Mode)") 
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
                # TUNED EXTRACTION: x_tolerance=1
                # This fixes "Primar y" while still keeping separate words apart.
                words = page.extract_words(x_tolerance=1)
                page_text = ' '.join([w['text'] for w in words])
                
                # Regex Patch for CamelCase
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
                st.markdown(f"### 🔹 {item['phrase']}")
                st.caption(f"**Category:** {item['category']}")
                
                # The "Tether" (Quote) with Citation
                st.markdown(f"> *\"{item['snippet']}\"*")
                st.markdown(f"**[Page {item['page']}]**")
                
                # The Neutral Clarification
                st.markdown(f"**Clarification:** {item['question']}")
                
                st.divider()
    else:
        st.write("Upload a document to begin the audit.")
