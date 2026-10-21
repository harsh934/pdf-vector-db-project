# Context-Aware Document Intelligence System Using a Local Vector Database

A lightweight Retrieval-Augmented Generation (RAG) system built using **Chroma DB** as a local vector database management system (VDBMS) and **Google Gemini API** for semantic similarity search and context synthesis over unstructured PDF documents[cite: 2, 20].

---

## 📌 Project Architecture & Workflow
1. **Document Ingestion (`app.py`):** Loads `document.pdf` (*The 48 Laws of Power* - 476 pages), splits raw text into overlapping semantic chunks (1200 characters, 150 overlap), generates 384-dimensional vector embeddings using `all-MiniLM-L6-v2`, and persists vectors with page metadata into a local Chroma DB collection[cite: 2, 8, 9].
2. **Vector Indexing:** Chroma DB uses a Hierarchical Navigable Small World (HNSW) graph index for fast Approximate Nearest Neighbor (ANN) search[cite: 2].
3. **Interactive UI & RAG Pipeline (`gui.py`):** A Streamlit dashboard takes plain-English queries, computes vector distance to fetch top-k context passages, and passes them to the Gemini API (`gemini-3.8-flash`) to generate factual, hallucination-free answers with page citations[cite: 2, 16, 20, 22].

---

## 🛠️ Tech Stack
- **Vector Database:** Chroma DB (HNSW Indexing)[cite: 2]
- **Embedding Model:** SentenceTransformers (`all-MiniLM-L6-v2`)[cite: 2]
- **LLM Synthesis:** Google Gemini API (`gemini-3.8-flash`)[cite: 20, 22]
- **Frontend Framework:** Streamlit[cite: 16]
- **Document Processing:** PyPDF & LangChain Text Splitters[cite: 2]

---

## 📁 Repository Structure
- `app.py` - PDF parsing, chunking, embedding, and vector DB insertion script[cite: 2, 14].
- `gui.py` - Interactive Streamlit web interface with Gemini RAG pipeline[cite: 14, 20].
- `document.pdf` - Target document dataset[cite: 8].
- `dbms_project_report.pdf` - Complete DBMS academic report.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/harsh934/pdf-vector-db-project.git](https://github.com/harsh934/pdf-vector-db-project.git)
cd pdf-vector-db-project
