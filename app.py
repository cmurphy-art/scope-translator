import streamlit as st
import pdfplumber
import google.generativeai as genai
import json
import re

st.set_page_config(layout="wide", page_title="Scope Translator (Diagnostic)")
st.title("Scope Translator: Diagnostic Mode")

# --- STEP 1: VERIFY API KEY ---
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    # Show first 4 chars to prove it loaded (securely)
    st.success(f"API Key loaded from Secrets (starts with: `{api_key[:4]}...`)")
    genai.configure(api_key=api_key)
else:
    st.error("CRITICAL: No API Key found in Streamlit Secrets.")
    st.stop()

# --- STEP 2: CHECK AVAILABLE MODELS ---
st.subheader("1. Connectivity & Model Test")
active_model_name = None

try:
    with st.spinner("Pinging Google AI to find available models..."):
        # Ask Google which models this key can access
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        st.write("Models available to your key:", available_models)
        
        # Auto-select the best one found
        if "models/gemini-1.5-flash" in available_models:
            active_model_name = "models/gemini-1.5-flash"
        elif "models/gemini-pro" in available_models:
            active_model_name = "models/gemini-pro"
        elif available_models:
            active_model_name = available_models[0]
            
        if active_model_name:
            st.success(f"✅ Connection Successful! Using model: **{active_model_name}**")
        else:
            st.error("❌ Connection worked, but no text generation models are available to this key.")
            st.stop()

except Exception as e:
    st.error(f"❌ Connection Failed: {e}")
    st.info("Check your API Key in 'Secrets' for extra spaces or missing quotes.")
    st.stop()

# --- STEP 3: THE SCANNER ---
st.subheader("2. Document Scan")
uploaded_file = st.file_uploader("Upload Scope PDF", type="pdf")

if uploaded_file and st.button("Run Diagnostic Scan"):
    
    # Extract Text (Nuclear Option)
    full_text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=1)
            text = ' '.join([w['text'] for w in words])
            full_text += text + "\n"
    
    st.write(f"**Document Read:** Extracted {len(full_text)} characters.")
    
    # The Prompt
    SYSTEM_PROMPT = """
    You are a construction scope analyzer. Find ambiguities.
    
    TAXONOMY:
    1. UNDEFINED_BOUNDARY (e.g. "match existing")
    2. SUBJECTIVE_QUALITY (e.g. "industry standard")
    3. UNDEFINED_SCOPE (e.g. "turnkey", "complete system")
    
    OUTPUT: Return ONLY a raw JSON list. Example:
    [{"trigger_text": "match existing", "classification": "UNDEFINED_BOUNDARY", "reasoning": "..."}]
    """
    
    final_prompt = f"{SYSTEM_PROMPT}\n\nDOCUMENT CONTENT:\n{full_text[:10000]}" # Limit to first 10k chars for speed
    
    # Send to AI
    try:
        st.info("Sending text to AI Analyst...")
        model = genai.GenerativeModel(active_model_name)
        response = model.generate_content(final_prompt)
        
        # SHOW THE RAW OUTPUT (This is what we need to see!)
        st.subheader("3. Raw AI Response (Debug View)")
        st.code(response.text)
        
        # Attempt to Parse
        clean_json = re.sub(r'```json|```', '', response.text).strip()
        data = json.loads(clean_json)
        
        st.subheader("4. Processed Findings")
        for item in data:
            st.markdown(f"**🔹 {item.get('trigger_text')}**")
            st.caption(item.get('reasoning'))
            st.divider()
            
    except Exception as e:
        st.error(f"Processing Error: {e}")
