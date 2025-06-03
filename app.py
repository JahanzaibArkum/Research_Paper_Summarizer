import streamlit as st

# Set page config FIRST before any other Streamlit commands
st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📚",
    layout="centered"
)

# Now the rest of the imports
import json
import requests
from PyPDF2 import PdfReader
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables (make sure .env has GROQ_API_KEY)
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

# Load template
with open('template.json') as f:
    TEMPLATE = json.load(f)

def extract_text_from_pdf(url):
    response = requests.get(url)
    pdf_file = BytesIO(response.content)
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text[:3000]  # Truncate to fit model context

def generate_summary(text, audience, length):
    # Format prompt using template
    prompt = TEMPLATE["prompt_template"].format(
        audience=audience,
        length=f"{length} sentences",
        text=text
    )
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": TEMPLATE["system_prompt"]},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(GROQ_URL, headers=headers, json=data)
        response.raise_for_status()
        summary = response.json()["choices"][0]["message"]["content"]
        return summary.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #000000 50%, #2575fc 100%);
    }

    div.block-container {
        max-width: 1400px !important;
        padding: 2rem;
        margin: auto;
        position: relative;
        top: 6%;
        transform: translateY(-50%);
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>select, .stSlider>div>div>div>div {
        background-color: rgba(255,255,255,0.9) !important;
        color: #000000 !important;
    }
    .css-1d391kg {padding-top: 3rem;}
    h1, h2, h3, .stMarkdown {color: white !important;}
    .stButton>button {
        background: #ff4b4b !important;
        color: white !important;
        border: none;
        padding: 10px 24px;
        border-radius: 50px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: #ff2b2b !important;
        transform: scale(1.05);
    }
     /* Aggressive override for main container width */
    div.block-container {
        max-width: 1400px !important;
        padding-left: 4rem !important;
        padding-right: 4rem !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📚 Research Paper Summarizer")
st.markdown("### AI-powered summaries for academic papers )")

# Inputs
col1, col2 = st.columns(2)
with col1:
    url = st.text_input("PDF URL:", placeholder="https://arxiv.org/pdf/...")
    
with col2:
    audience = st.sidebar.selectbox(
        "Summary Type:",
        ("Beginner-Friendly", "Technical", "Concise", "Detailed"),
        index=0
    )

length = st.slider("Summary Length (sentences):", 3, 10, 5)

# Generate button
if st.button("Generate Summary", use_container_width=True):
    if not url:
        st.warning("Please enter a PDF URL")
    else:
        with st.spinner("Downloading and processing paper..."):
            try:
                text = extract_text_from_pdf(url)
                if not text:
                    st.error("Failed to extract text from PDF")
                    st.stop()
                
                with st.spinner("Generating summary (this may take 20-30 seconds)..."):
                    summary = generate_summary(text, audience, length)
                
                st.success("Summary Generated!")
                st.subheader("Research Summary")
                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.1); padding:20px; border-radius:10px;'>{summary}</div>", 
                    unsafe_allow_html=True
                )
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with ♥ using Llama3 70B via Groq API")
