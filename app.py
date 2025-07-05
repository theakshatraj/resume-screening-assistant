import streamlit as st
import fitz  # PyMuPDF
import requests
import datetime

# --- CONFIGURATION ---
API_KEY = "Vn2TDzh4Z1N-h7DPZN-fNQuZ3qCpp3mRNh4W4vvKzgL9"
PROJECT_ID = "7e4176a3-cd6f-49fa-929c-df95936bb40c"
ENDPOINT = "https://us-south.ml.cloud.ibm.com"
MODEL_ID = "google/flan-t5-xl"

PROMPT_TEMPLATE = """
You are an AI assistant. Categorize the following resume into one of these job categories:
[Data Science, Software Development, HR, Marketing, Finance, Product Management]

Resume:
{text}
"""

# --- HELPER FUNCTIONS ---

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def get_iam_token(api_key):
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    response = requests.post(iam_url, data={
        "apikey": api_key,
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
    }, headers={"Content-Type": "application/x-www-form-urlencoded"})
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"Failed to get IAM token: {response.text}")

def classify_resume(text):
    prompt = PROMPT_TEMPLATE.format(text=text[:3000])
    access_token = get_iam_token(API_KEY)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    payload = {
        "model_id": MODEL_ID,
        "input": prompt,
        "parameters": {"decoding_method": "greedy", "max_new_tokens": 20},
        "project_id": PROJECT_ID,
    }

    try:
        response = requests.post(f"{ENDPOINT}/ml/v1/text/generation?version=2024-05-01", headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result.get("results", [{}])[0].get("generated_text", "Uncategorized")
        else:
            return f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return str(e)

# --- UI STARTS HERE ---

st.set_page_config(page_title="Resume Screening Assistant", page_icon="📄", layout="centered")
st.markdown(
    """
    <style>
        .main {background-color: #fafafa;}
        .stButton>button {
            background-color: #2563eb;
            color: white;
            font-weight: 500;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stTextArea textarea {
            font-size: 0.9rem;
            line-height: 1.5;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>📄 Resume Screening Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem;'>Upload a resume and get the predicted job category using IBM watsonx.ai</p>", unsafe_allow_html=True)
st.divider()

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], label_visibility="collapsed")

if uploaded_file:
    with st.spinner("Analyzing the resume..."):
        text = extract_text_from_pdf(uploaded_file)
        category = classify_resume(text)

    st.success("✅ Resume successfully classified!")
    st.markdown(f"<h3 style='color:#2563eb;'>🏷 Predicted Category: <code>{category}</code></h3>", unsafe_allow_html=True)

    with st.expander("📄 View Resume Content"):
        st.text_area("Extracted Text", text[:3000] + "...", height=300)

    # Optional: Download result
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_text = f"Filename: {uploaded_file.name}\nPredicted Category: {category}\n\nResume:\n{text}"
    st.download_button(
        label="⬇️ Download Result",
        data=result_text,
        file_name=f"classified_resume_{timestamp}.txt",
        mime="text/plain"
    )
