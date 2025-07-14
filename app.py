# AI-Powered Resume Screening System using NLP and BERT (with Streamlit UI)

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import spacy
import PyPDF2
import streamlit as st

# Load BERT Model
model = SentenceTransformer('all-MiniLM-L6-v2')
import spacy
import spacy.cli
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Preprocess text
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Embed text
def get_embedding(text):
    return model.encode(text, convert_to_tensor=True)

# App UI using Streamlit
st.title("📄 AI-Powered Resume Screening")
st.write("Upload a job description and multiple resumes to rank them by match quality.")

job_desc_input = st.text_area("Enter Job Description", height=200)
resume_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("🔍 Match Resumes"):
    if not job_desc_input or not resume_files:
        st.warning("Please provide both job description and resumes.")
    else:
        with st.spinner("Processing..."):
            job_text = preprocess_text(job_desc_input)
            job_embedding = get_embedding(job_text)

            results = []
            for file in resume_files:
                resume_text = extract_text_from_pdf(file)
                cleaned_resume = preprocess_text(resume_text)
                resume_embedding = get_embedding(cleaned_resume)
                similarity = util.cos_sim(job_embedding, resume_embedding).item()
                results.append((file.name, similarity))

            ranked = sorted(results, key=lambda x: x[1], reverse=True)

            st.subheader("📊 Ranked Resumes:")
            for name, score in ranked:
                st.write(f"**{name}** — Match Score: `{score:.4f}`")
