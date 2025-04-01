import streamlit as st
import fitz  # PyMuPDF
import tiktoken
from openai import OpenAI
import json
import re
import os
from dotenv import load_dotenv

# ========== CONFIGURATION ==========
load_dotenv()



GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "llama3-70b-8192"

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


# ========== TOKENIZATION ==========
def tokenize_text(text, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    return enc.encode(text)


def split_text(text, max_tokens=1500):
    words = text.split()
    chunks, chunk, current_tokens = [], [], 0

    for word in words:
        tokens = len(tokenize_text(word))
        if current_tokens + tokens > max_tokens:
            chunks.append(" ".join(chunk))
            chunk, current_tokens = [], 0
        chunk.append(word)
        current_tokens += tokens

    if chunk:
        chunks.append(" ".join(chunk))
    return chunks


# ========== PDF TEXT EXTRACTION ==========
def extract_pdf_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


# ========== SEND TO GROQ ==========
def ask_llm(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts structured data from documents."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# ========== INVOICE FIELD PARSING ==========
def extract_invoice_fields(text_chunk):
    prompt = f"""
Extract the following fields from the invoice text:
- invoice_date
- invoice_number
- amount
- due_date

Return the result in this exact JSON format:
{{
  "invoice_date": "...",
  "invoice_number": "...",
  "amount": "...",
  "due_date": "..."
}}

Invoice text:
\"\"\"
{text_chunk}
\"\"\"
"""
    response = ask_llm(prompt)
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        return json.loads(match.group()) if match else {"error": "No JSON found"}
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON"}


# ========== STREAMLIT UI ==========
st.set_page_config(page_title="Invoice Extractor", layout="centered")
st.title("📄 Invoice Field Extractor (Groq + LLaMA3)")

uploaded_file = st.file_uploader("Upload a PDF invoice", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 Extracting text from invoice..."):
        text = extract_pdf_text(uploaded_file)
        chunks = split_text(text)

    st.success(f"Extracted text split into {len(chunks)} chunk(s)")

    extracted_data = {}
    for i, chunk in enumerate(chunks):
        st.info(f"⏳ Sending chunk {i + 1} to Groq LLaMA3...")
        result = extract_invoice_fields(chunk)
        extracted_data.update(result)
        break  # Stop after first meaningful chunk (for efficiency)

    st.subheader("📋 Extracted Fields")
    st.json(extracted_data)
