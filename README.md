# 🧑‍⚖️ Autonomous Legal Assistant

An AI-powered legal assistant tailored for the Indian legal ecosystem. This project automates legal document interpretation, drafting, deadline tracking, and case referencing using advanced NLP and retrieval techniques.

---

## 📝 Problem Statement

The Indian legal system generates a vast amount of unstructured legal data — court judgments, notices, contracts, etc. Manual interpretation of these documents is time-consuming and error-prone. This project aims to create an AI assistant that can **understand, summarize, and generate legal content** efficiently.

---

## 🎯 Objectives

- **Information Extraction:** Automatically extract critical legal information (parties involved, dates, obligations).
- **Document Drafting:** Generate contracts, NDAs, and legal notices using AI-based templates.
- **Deadline Tracking:** Keep track of statute of limitations and other key legal deadlines.
- **Legal Reasoning:** Provide context-aware legal advice and reference case law based on user queries.

---

## ⚙️ Features

- 🧾 Upload and process legal documents (contracts, judgments).
- 📄 Auto-generate contracts, NDAs, and legal notices.
- 🔍 Ask legal questions like *“What is IPC 420?”* and get LLM-powered answers with references.
- 🧠 Context-aware search across Indian case law.
- 📌 Track important legal dates and receive deadline reminders.

---

## 💡 Tech Stack

### 🧰 Frontend
- `Streamlit` or `Flask` for building a simple user interface.

### 🧠 Backend / AI Models
- **LLMs & APIs:**
  - GPT-4 / Mistral for natural language generation.
  - InLegalBERT / LegalBERT – fine-tuned on Indian legal texts.
  - HuggingFace Transformers for model integration.

- **NLP Libraries:**
  - SpaCy – NER for extracting names, dates, statutes.
  - NLTK – preprocessing and tokenization.
  - Scikit-learn – basic classification and clustering.

### 📚 Retrieval & Indexing
- FAISS / ChromaDB / Elasticsearch – for dense and keyword search.
- RAG (Retrieval-Augmented Generation) – to combine search + generation.

### 🧪 ML Models (Fine-Tuned)
- Contract Clause Classifier
- Legal Summarizer
- Judgment Outcome Predictor

---

## 🧭 Workflow

1. User uploads a legal document or types a query.
2. Backend retrieves relevant laws/judgments using vector search.
3. A large language model interprets and responds contextually.
4. Summaries, answers, or document drafts are generated and displayed.
5. Documents, metadata, and indexes are saved for future reference.

---

## 📥 Input Types

- Uploaded PDF/DOC of legal documents
- Form-based contract details (NDA, lease, etc.)
- Legal queries in natural language

## 📤 Output Types

- Extracted legal entities and facts
- Auto-drafted contracts or legal notices
- Summaries of uploaded content
- LLM-generated legal answers with citations

---

## 📂 Dataset Sources

- [ILDC](https://www.academia.edu/41521229/ILDC_Case_Law_Corpus)
- [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT)
- Indian Penal Code, Bare Acts, Case Law from SCC / Indian Kanoon (public scrapes)

---

## ✅ Deliverables

- Interactive legal assistant interface
- Contract generator module
- Legal Q&A with case citation
- Legal document summarizer
- Backend with RAG architecture
- Deployment-ready application (Docker/Streamlit)

---


