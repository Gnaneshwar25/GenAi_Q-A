import streamlit as st
import pandas as pd
import zipfile
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader,
    JSONLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.schema import Document


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)
gemini = genai.GenerativeModel("gemini-1.5-flash")

SUPPORTED_TYPES = ["pdf", "txt", "docx", "csv", "xlsx", "xls", "json"]

def load_csv_with_summary(file_path):
    df = pd.read_csv(file_path)
    summary = []
    summary.append("### File Summary:\n")
    summary.append(str(df.describe(include='all')))
    summary.append("\n### Columns:\n" + ", ".join(df.columns))
    summary.append(f"\n### Total Rows: {len(df)}")

    max_rows = 1000
    row_texts = [", ".join([f"{col}: {row[col]}" for col in df.columns]) for _, row in df.head(max_rows).iterrows()]
    summary.extend(row_texts)

    return [Document(page_content="\n".join(summary))]


def load_excel_with_summary(file_path):
    df = pd.read_excel(file_path)
    summary = []
    summary.append("### File Summary:\n")
    summary.append(str(df.describe(include='all')))
    summary.append("\n### Columns:\n" + ", ".join(df.columns))
    summary.append(f"\n### Total Rows: {len(df)}")

    max_rows = 1000
    row_texts = [", ".join([f"{col}: {row[col]}" for col in df.columns]) for _, row in df.head(max_rows).iterrows()]
    summary.extend(row_texts)

    return [Document(page_content="\n".join(summary))]

def get_file_type(filename):
    ext = filename.lower().split('.')[-1]
    if ext in SUPPORTED_TYPES:
        return ext
    return None


def load_file(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_type == "txt":
        loader = TextLoader(file_path)
        return loader.load()
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
        return loader.load()
    elif file_type == "csv":
        return load_csv_with_summary(file_path)
    elif file_type in ["xlsx", "xls"]:
        return load_excel_with_summary(file_path)
    elif file_type == "json":
        loader = JSONLoader(file_path, jq_schema=".", text_content=False)
        return loader.load()
    else:
        return []

def process_zip(zip_path):
    extracted_docs = []
    with tempfile.TemporaryDirectory() as extract_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        for root, _, files in os.walk(extract_dir):
            for file in files:
                full_path = os.path.join(root, file)
                file_type = get_file_type(file)
                if file_type:
                    extracted_docs.extend(load_file(full_path, file_type))
    return extracted_docs


def embed_and_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vectordb = FAISS.from_documents(docs, embedding=embeddings)
    return vectordb

def ask_gemini(question, context):
    prompt = (
        f"Answer the following question based only on the provided context. "
        f"If the answer is not present, reply exactly: Not related to this file uploaded.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = gemini.generate_content(prompt)
    return response.text.strip()

def search_vectordb(vectordb, query, k=3):
    return vectordb.similarity_search(query, k=k)


st.title(" Gemini Ai QA ")

uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, TXT, CSV, XLSX, JSON, ZIP)", type=SUPPORTED_TYPES + ["zip"])

if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    st.success(f"File uploaded: {uploaded_file.name}")

    with st.spinner("Extracting and indexing..."):
        if file_ext == "zip":
            documents = process_zip(tmp_path)
        else:
            file_type = get_file_type(uploaded_file.name)
            documents = load_file(tmp_path, file_type)

        vectordb = embed_and_store(documents)
    os.unlink(tmp_path)

    question = st.text_input("Ask a question about this file:")
    if question:
        with st.spinner("Searching for relevant context..."):
            search_results = search_vectordb(vectordb, question, k=3)
            context = "\n\n".join([doc.page_content for doc in search_results])
        with st.spinner("Consulting Gemini..."):
            answer = ask_gemini(question, context)
        st.markdown(f"**Answer:** {answer}")
else:
    st.info("Please upload a file to begin.")