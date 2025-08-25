# 🩺 Medical Report Summarizer (RAG)

A Retrieval-Augmented Generation (RAG) project for summarizing medical reports using Google Gemini, LangChain, ChromaDB, and Streamlit. This tool provides contextual summaries from trusted medical sources.

⚠️ **Disclaimer:** This tool is for **educational and research purposes only**. It does **not** provide medical advice, diagnosis, or prescriptions.

---

## 📝 Table of Contents

- [About the Project](#about-the-project)
- [What is RAG?](#what-is-rag)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [How It Works (Step by Step)](#how-it-works-step-by-step)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Example Workflow](#example-workflow)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Key Learnings](#key-learnings)
- [Why This Project?](#why-this-project)
- [Final Note](#final-note)

---

## 🌟 About the Project

This project is my **first hands-on implementation of Retrieval-Augmented Generation (RAG)** applied to the medical domain.
It demonstrates how to use **Google Gemini**, **LangChain**, **ChromaDB**, and **Streamlit** to build a system that can **summarize medical reports with context from trusted medical sources**.

---

## 📖 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that improves LLMs by combining:
1.  **Retrieval** → fetching relevant documents from a knowledge base.
2.  **Generation** → feeding both the user input + retrieved docs into an LLM to generate a contextual, accurate answer.

Instead of the AI "guessing," it **grounds its response** in trusted external data.
In this project, the RAG system retrieves from **medical sources** before generating a structured summary.

---

## 🚀 Features

- Upload a **PDF or TXT medical report**.
- Extract and preview the report text.
- Retrieve context from trusted medical sources (WHO, NIH, Mayo Clinic, FDA, NICE, MedlinePlus).
- Summarize the report with structured JSON output:
    - `summary`
    - `confidence`
    - `ICD-10 codes` (if available)
    - `limitations`
    - `followUpQuestions`
    - `disclaimer`

---

## 🛠️ Tech Stack

- **LLM:** [Google Gemini 1.5 Flash](https://ai.google/) (fast + reliable structured output)
- **Embeddings:** Google Generative AI Embeddings (`models/embedding-001`)
    - Alternatives: OpenAI, Cohere, Hugging Face embeddings
- **Vector Database:** [ChromaDB](https://www.trychroma.com/) for local persistence
    - Alternatives: Pinecone, Weaviate, Milvus, FAISS
- **Framework:** [LangChain](https://www.langchain.com/) to connect embeddings, retrievers, and LLM
- **UI:** [Streamlit](https://streamlit.io/) for an easy-to-use web app
- **Other Libraries:** PyMuPDF (`fitz`) for PDF extraction, Pydantic for schema validation

---
```markdown
## ⚙️ How It Works (Step by Step)

```mermaid
flowchart TD
    A[Upload Medical Report] --> B[Extract Text (PDF/TXT)]
    B --> C[Split into Chunks]
    C --> D[Embed with Gemini Embeddings]
    D --> E[Store in ChromaDB]
    A --> F[Retrieve Top-k Similar Docs]
    F --> G[Combine Report + Context]
    G --> H[Pass to Gemini LLM via LangChain]
    H --> I[Generate Structured JSON Summary]
    I --> J[Display Results in Streamlit]
```

### 🔑 Breakdown of Code

- **Environment Setup**
    - Loads `.env` with `GOOGLE_API_KEY`.
    - Configures Gemini API for embeddings + LLM.
- **Trusted Sources Preload**
    - Fetches text from WHO, NIH, Mayo Clinic, NICE, FDA, MedlinePlus.
    - Splits into chunks → embeds → stores in ChromaDB.
- **File Upload & Text Extraction**
    - Supports `.pdf` and `.txt`.
    - Uses PyMuPDF for PDFs.
- **Vector Store + Retriever**
    - LangChain wrapper for ChromaDB.
    - Retrieves top-5 relevant documents for a given report.
- **Prompt & LLMChain**
    - System message: “You are a medical assistant, do NOT give diagnosis.”
    - User message includes report + retrieved context.
    - Structured JSON schema enforced with Pydantic.
- **JSON Parsing & Validation**
    - Extracts JSON string from raw LLM response.
    - Validates against `MedicalResponse` schema (Pydantic).
    - Displays structured summary in Streamlit.

---

## 📂 Project Structure

```
📦 medical-rag-summarizer
 ┣ 📜 app.py                 # Main Streamlit app
 ┣ 📜 .env                   # API key config
 ┣ 📂 chroma_db              # Persistent vector DB storage
 ┣ 📜 requirements.txt       # Dependencies
 ┗ 📜 README.md              # Documentation
```

---

## ▶️ Usage

1.  **Clone Repo**
    ```bash
    git clone https://github.com/MAYOR-G/medical-rag-summarizer.git
    cd medical-rag-summarizer
    ```
2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set API Key**
    Create a `.env` file in the project root:
    ```
    GOOGLE_API_KEY=your_gemini_api_key
    ```
4.  **Run Streamlit App**
    ```bash
    streamlit run app.py
    ```
5.  **Upload a Report**
    - Upload a PDF or TXT file.
    - See extracted text preview.
    - View JSON output from Gemini.
    - Explore structured summary in Streamlit UI.

---

## 📊 Example Workflow

Upload a medical report PDF.
System extracts and chunks text.
Retrieves context from WHO, NIH, Mayo Clinic, etc.
LLM outputs JSON like:

```json
{
  "summary": "The report suggests hypertension management...",
  "confidence": "High",
  "codes": ["I10"],
  "limitations": ["Report lacks lab test details"],
  "followUpQuestions": ["Has the patient undergone ECG?"],
  "disclaimer": "This summary is for educational purposes only."
}
```

---

## ⚠️ Limitations

- Not a replacement for professional medical consultation.
- Works best on reports in English.
- JSON output sometimes needs cleaning if the LLM adds extra text.
- Retrieval depends heavily on quality of embeddings + vector database.

---

## 🔮 Future Improvements

- Add more sources (e.g., PubMed, clinical guidelines).
- Try alternative vector DBs like Pinecone.
- Experiment with GPT-4 or Claude for more detailed outputs.
- Improve UI with charts + side-by-side comparisons.
- Deploy on cloud (e.g., Streamlit Cloud, Hugging Face Spaces).

---

## 📌 Key Learnings

- RAG makes AI more factual by grounding responses in external data.
- Embeddings + retrievers are crucial for retrieval quality.
- JSON enforcement is tricky but solvable with schema validation.
- Modular design makes it easy to swap embeddings, vector DB, or LLM.

---

## ✨ Why This Project?

This was my first project on RAG, built to learn:

- How retrieval and generation work together
- How to handle embeddings and vector databases
- How to integrate Streamlit for usability
- How to enforce structured outputs with Pydantic

It helped me understand end-to-end AI pipelines and gave me hands-on practice with cutting-edge tools like Gemini + LangChain.

---

## 📢 Final Note

This project shows how RAG can be applied to sensitive domains like medicine while keeping safety in mind.
