import asyncio                       
asyncio.set_event_loop(asyncio.new_event_loop())

import os
import uuid
from pathlib import Path
from typing import List

import streamlit as st
import fitz                   
import requests
import chromadb
from chromadb import PersistentClient
import google.generativeai as genai   
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
import json

from langchain import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma as LangChainChroma


# Load .env and API key

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error(
        "⚠️  No Google API key found. Create a `.env` file containing:\n"
        "`GOOGLE_API_KEY=your_gemini_api_key`\n"
        "or set the environment variable manually."
    )
    st.stop()


# Configure Gemini (embeddings + LLM)

genai.configure(api_key=GOOGLE_API_KEY)

EMBEDDINGS = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=GOOGLE_API_KEY,
)

LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  
    temperature=0.0,
    api_key=GOOGLE_API_KEY,
)


# Initialise persistent Chroma collection

CHROMA_DIR = Path("chroma_db")
CHROMA_DIR.mkdir(exist_ok=True)

client = PersistentClient(path=str(CHROMA_DIR))
COLLECTION_NAME = "medical_kb"
collection = client.get_or_create_collection(name=COLLECTION_NAME)


# Helper functions


def fetch_text_from_url(url: str) -> str:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception as exc:
        st.warning(f"Could not fetch {url}: {exc}")
        return ""

def add_source_to_collection(name: str, text: str) -> None:
    splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(text)

    ids = [f"{name}_{i}_{uuid.uuid4().hex}" for i, _ in enumerate(chunks)]
    documents = chunks
    embeddings = EMBEDDINGS.embed_documents(documents)

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=[{"source": name} for _ in ids],
        embeddings=embeddings,
    )

def preload_trusted_sources() -> None:
    if collection.count() > 0:
        return
    st.info("Populating medical knowledge base …")
    sources = {
        "WHO": "https://en.wikipedia.org/wiki/World_Health_Organization",
        "NIH": "https://en.wikipedia.org/wiki/United_States_National_Institutes_of_Health",
        "Mayo_Clinic": "https://en.wikipedia.org/wiki/Mayo_Clinic",
        "NICE": "https://en.wikipedia.org/wiki/National_Institute_for_Health_and_Care_Excellence",
        "FDA": "https://en.wikipedia.org/wiki/United_States_Food_and_Drug_Administration",
        "MedlinePlus": "https://en.wikipedia.org/wiki/MedlinePlus",
    }
    for name, url in sources.items():
        txt = fetch_text_from_url(url)
        if txt:
            add_source_to_collection(name, txt)
    st.success("✅ Medical knowledge base ready!")

def extract_text(file) -> str:
    if file.type == "application/pdf" or file.name.lower().endswith(".pdf"):
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        txt = "\n".join(page.get_text() for page in doc)
        doc.close()
        return txt
    else:
        return file.read().decode("utf-8")


# Pydantic schema for AI response

class MedicalResponse(BaseModel):
    summary: str
    confidence: str
    codes: List[str] | None = None
    limitations: List[str] | None = None
    followUpQuestions: List[str] | None = None
    disclaimer: str


# Prompt templates

SYSTEM_MESSAGE = (
    "You are a medical assistant. Do NOT give diagnoses or prescriptions."
)

USER_MESSAGE = (
    "Report: {report}\n\nContext from trusted sources:\n{context}\n\nAnswer ONLY with the following JSON structure:\n"
    "{{\n  \"summary\": \"...\",\n  \"confidence\": \"...\",\n  \"codes\": [\"ICD-10 codes if available\"],\n  \"limitations\": [\"...\"],\n  \"followUpQuestions\": [\"...\"],\n  \"disclaimer\": \"...\"\n}}"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE),
        HumanMessagePromptTemplate.from_template(USER_MESSAGE),
    ]
)


# Streamlit UI

st.set_page_config(page_title="Medical Report Summarizer (RAG)", page_icon="🩺")
st.title("🩺 Medical Report Summarizer (RAG)")

preload_trusted_sources()

uploaded_file = st.file_uploader(
    "📁 Upload a PDF or TXT report",
    type=["pdf", "txt"],
    accept_multiple_files=False,
)

if uploaded_file is None:
    st.info("💡 No file uploaded yet. Please choose a report to analyze.")
    st.stop()

# Extract and preview
report_text = extract_text(uploaded_file)
st.subheader("📄 Extracted Text (preview)")
st.text_area("Report preview", report_text[:2000], height=200, key="preview")

#  Build a Chroma retriever using LangChain's wrapper 
vectorstore = LangChainChroma(
    collection_name=COLLECTION_NAME,
    client=client,
    embedding_function=EMBEDDINGS,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
retrieved_docs = retriever.invoke(report_text)
context = "\n\n".join(doc.page_content for doc in retrieved_docs)

# Build & run prompt
chain = LLMChain(llm=LLM, prompt=PROMPT)
ai_response_raw = chain.run({"report": report_text[:1500], "context": context})

# Raw JSON
st.subheader("⇨ Raw JSON reply from Gemini")
st.code(ai_response_raw, language="json")

# Utility to extract JSON from AI response
def extract_json_from_string(text: str) -> str:
    """
    Extracts a JSON string from a given text by finding the first '{' and the last '}'.
    Assumes there is at least one JSON object in the text.
    """
    try:
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            return text[start_index : end_index + 1]
    except Exception:
        pass
    return "" # Return empty string if no valid JSON structure is found

# Parsed JSON
st.subheader("✅ Structured response")
try:
    cleaned = extract_json_from_string(ai_response_raw)
    data = json.loads(cleaned)
    ai_response_parsed = MedicalResponse.model_validate(data)
    st.json(ai_response_parsed.model_dump())
except ValidationError as err:
    st.error("❌ Could not parse the AI response as the expected JSON schema.")
    st.json(err.errors())
