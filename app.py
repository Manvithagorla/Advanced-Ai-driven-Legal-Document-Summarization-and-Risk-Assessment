# All rights reserved.
# Copyright (c) 2025 VidzAI
# This software and associated documentation files are the property of VidzAI.
# No part of this software may be copied, modified, distributed, or used 
# without explicit permission from VidzAI.

import streamlit as st
import os
import pdfplumber
import docx
import faiss
import numpy as np
import tiktoken
import matplotlib.pyplot as plt
import requests
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
import hashlib

# Load API key
load_dotenv()
api_key = os.getenv("API_KEY")
client = Groq(api_key=api_key)

# Initialize FAISS and Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
faiss_index = faiss.IndexFlatL2(dimension)
documents = []
doc_embeddings = []

# Custom Risk Keywords with unique scoring system
RISK_KEYWORDS = {
    "penalty": 8.3, "breach": 9.1, "liability": 7.7, "compliance": 5.9,
    "sanction": 8.6, "lawsuit": 9.4, "violation": 10.0, "termination": 6.2
}

# Custom email sending with document hash
def send_email(user_email, document_summary, doc_hash):
    FORMSPREE_URL = "https://formspree.io/f/xjkybjgj"
    if not "@" in user_email or not "." in user_email.split("@")[1]:
        return "❌ Invalid email format"
    
    payload = {
        "email": user_email,
        "subject": f"Legal Document Summary - {datetime.now().strftime('%Y-%m-%d')} (Hash: {doc_hash[:8]})",
        "message": document_summary
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(FORMSPREE_URL, json=payload, headers=headers)
    return "✅ Email sent successfully!" if response.status_code == 200 else f"❌ Failed to send email. Error: {response.text}"

# Custom PDF extraction with metadata
def extract_text_from_pdf(file):
    text = f"Document Extracted on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            extracted = page.extract_text()
            if extracted:
                text += f"[Section {i}]\n{extracted.strip()}\n\n"
    return text.strip()

# Custom DOCX extraction with metadata
def extract_text_from_docx(file):
    text = f"Document Extracted on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    doc = docx.Document(file)
    for i, para in enumerate(doc.paragraphs, 1):
        if para.text.strip():
            text += f"[Clause {i}] {para.text.strip()}\n"
    return text.strip()

# Custom text splitting with overlap
def split_text(text, max_tokens=5000, overlap=200):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        start = i
        end = min(i + max_tokens, len(tokens))
        chunks.append(tokens[start:end])
    return [encoding.decode(chunk) for chunk in chunks]

# Custom summarization with Groq API and fallback
def summarize_large_document(text):
    try:
        text_chunks = split_text(text)
        summaries = []
        for i, chunk in enumerate(text_chunks, 1):
            prompt = f"Summarize section {i} of legal document:\n\n{chunk}"
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.5,
                max_completion_tokens=500
            )
            summaries.append(f"Section {i}:\n{response.choices[0].message.content.strip()}")
        return "\n\n".join(summaries)
    except Exception as e:
        st.warning(f"⚠ AI summarization failed: {str(e)}. Using fallback method.")
        lines = text.split("\n")
        summary = []
        keyword_count = 0
        for line in lines[:50]:
            if any(keyword in line.lower() for keyword in RISK_KEYWORDS):
                summary.append(f"⚠ {line.strip()}")
                keyword_count += 1
            else:
                summary.append(line.strip())
        return f"Fallback Summary (Risk Keywords Detected: {keyword_count}):\n\n{'\n'.join(summary)}"

# Custom risk assessment with severity levels
def assess_risks(text):
    risk_scores = {}
    severity_levels = {10: "Critical", 8: "High", 6: "Medium", 4: "Low"}
    for i, line in enumerate(text.split("\n"), 1):
        detected_keywords = [word for word in RISK_KEYWORDS if word in line.lower()]
        if detected_keywords:
            score = sum(RISK_KEYWORDS[word] for word in detected_keywords)
            severity = next((level for threshold, level in severity_levels.items() if score >= threshold), "Low")
            risk_scores[f"Line {i}: {', '.join(detected_keywords)} ({severity})"] = score
    return risk_scores

# Custom visualization with color coding
def plot_risk_analysis(risk_scores, title):
    if not risk_scores:
        st.write("✅ No significant risks detected!")
        return
    
    clauses = list(risk_scores.keys())
    scores = list(risk_scores.values())
    colors = ['#FF0000' if score >= 9 else '#FFA500' if score >= 7 else '#FFFF00' for score in scores]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(clauses, scores, color=colors)
    ax.set_xlabel("Risk Score")
    ax.set_title(title)
    ax.invert_yaxis()
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF0000', label='Critical (≥9)'),
        Patch(facecolor='#FFA500', label='High (7-8.9)'),
        Patch(facecolor='#FFFF00', label='Medium (≤6.9)')
    ]
    ax.legend(handles=legend_elements, title="Risk Severity")
    
    st.pyplot(fig)

# Add document to FAISS
def add_to_faiss(text):
    global faiss_index, documents, doc_embeddings
    embedding = embedding_model.encode([text])
    faiss_index.add(np.array(embedding, dtype=np.float32))
    documents.append(text)
    doc_embeddings.append(embedding)

# Custom legal chatbot with RAG
def legal_chatbot(user_question, document_text):
    global faiss_index, documents
    
    try:
        question_embedding = embedding_model.encode([user_question])
        distances, indices = faiss_index.search(np.array(question_embedding, dtype=np.float32), k=1)
        
        if indices[0][0] == -1 or not documents:
            return "I couldn't find relevant information in the document to answer your question."
        
        relevant_chunk = documents[indices[0][0]]
        doc_hash = hashlib.md5(relevant_chunk.encode()).hexdigest()[:8]
        
        prompt = (
            f"You are a legal assistant specializing in compliance. Using the document context below, "
            f"answer the user's question with reference to the document:\n\n"
            f"**Document Context (Hash: {doc_hash}):**\n{relevant_chunk}\n\n"
            f"**User Question:**\n{user_question}\n\n"
            f"Provide a concise response with document reference if applicable."
        )
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            max_completion_tokens=300
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"⚠ Chatbot error: {str(e)}. Using fallback response.")
        return simulate_legal_chatbot(user_question, document_text)

# Fallback chatbot simulation
def simulate_legal_chatbot(query, document_text):
    query_lower = query.lower()
    response = "I'm analyzing your question in the context of the document..."
    for line in document_text.split("\n"):
        if any(keyword in line.lower() for keyword in RISK_KEYWORDS) and any(word in line.lower() for word in query_lower.split()):
            response += f"\n\nRelevant finding: {line.strip()}"
            break
    else:
        response += "\n\nNo specific information found in the document related to your query."
    return response

# Streamlit UI
st.set_page_config(page_title="🔍 VidzAI Legal Compliance Analyzer", layout="wide")
st.title("📜 AI Powered Legal Document Summarizer")

# Sidebar
st.sidebar.header("📂 Document Upload")
st.sidebar.write("Note: This AI assistant is for informational purposes only and should not replace professional legal advice.")
uploaded_file = st.sidebar.file_uploader("Upload your legal document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    
    # Extract text and generate document hash
    if file_extension == "pdf":
        document_text = extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        document_text = extract_text_from_docx(uploaded_file)
    else:
        document_text = uploaded_file.getvalue().decode("utf-8")
    doc_hash = hashlib.md5(document_text.encode()).hexdigest()

    # Add to FAISS
    add_to_faiss(document_text)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📜 Document Summary", "⚠ Risk Assessment", "🤖 Legal Advisor"])

    with tab1:
        st.subheader("📜 Document Summary")
        summary = summarize_large_document(document_text)
        st.write(summary)
        
        summary_filename = f"Legal_Summary_{doc_hash[:8]}.txt"
        with open(summary_filename, "w", encoding="utf-8") as f:
            f.write(summary)
        with open(summary_filename, "rb") as f:
            st.download_button("📥 Download Summary", f, file_name=summary_filename)
        
        st.subheader("📧 Email Summary")
        user_email = st.text_input("Enter your email address:", key="email_tab1")
        if st.button("Send Summary via Email"):
            if user_email:
                email_status = send_email(user_email, summary, doc_hash)
                st.success(email_status)
            else:
                st.warning("⚠ Please enter a valid email address.")

    with tab2:
        st.subheader("⚠ Risk Assessment")
        risks = assess_risks(document_text)
        plot_risk_analysis(risks, "Risk Assessment Report")

    with tab3:
        st.subheader("🤖 Legal Advisor")
        query = st.text_input("Ask a question about your document:")
        if query:
            response = legal_chatbot(query, document_text)
            st.write("🧠 AI Response:")
            st.write(response)
