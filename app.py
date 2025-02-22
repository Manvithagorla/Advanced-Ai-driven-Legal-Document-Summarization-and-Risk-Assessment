import streamlit as st
import os
import pdfplumber
import docx
import io
from groq import Groq
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("API_KEY")
client = Groq(api_key=api_key)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
    return text.strip()

# Function to extract text from DOCX
def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error processing DOCX: {e}")
        return ""

# Function to summarize the document
def summarize_document(text):
    prompt = f"Summarize the following legal document, highlighting key clauses, important details, and potential risks:\n\n{text}"

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            max_completion_tokens=500,
            top_p=1,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Unable to generate summary."

# Streamlit UI Configuration
st.set_page_config(page_title="AI-Powered Legal Document Summarization", layout="wide")
st.title("📄 AI-Driven Legal Document Summarization")

# Upload file
uploaded_file = st.file_uploader("Upload a Legal Document (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    document_text = ""

    # Extract text
    with st.spinner("🔍 Extracting text..."):
        if file_extension == "pdf":
            document_text = extract_text_from_pdf(uploaded_file)
        elif file_extension == "docx":
            document_text = extract_text_from_docx(uploaded_file)
        else:
            document_text = uploaded_file.getvalue().decode("utf-8")

    if document_text:
        with st.spinner("🤖 Generating summary..."):
            summary = summarize_document(document_text)

        st.subheader("📜 Summarized Document")
        st.write(summary)

        # Prepare summary for download
        summary_file = io.BytesIO()
        summary_file.write(summary.encode())
        summary_file.seek(0)

        st.download_button("📥 Download Summary (TXT)", summary_file, file_name="Legal_Summary.txt", mime="text/plain")
