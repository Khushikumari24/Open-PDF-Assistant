# Open PDF Assistant 📄

A 100% free, open-source, privacy-first document chatbot. It uses local vector embeddings and a 4-bit quantized LLM (Qwen 2.5) running entirely locally, requiring **zero API keys** and ensuring enterprise data never leaves the environment.

## 🚀 Features
- **Zero API Costs:** Runs entirely on local compute (optimized for T4 GPU or standard CPUs).
- **Privacy-First:** Your PDF data is processed locally and never sent to third-party APIs.
- **Conversational Memory:** Remembers context across multiple questions using a custom RAG loop.
- **Smart Chunking:** Uses recursive text splitting for accurate, context-aware retrieval.

## 🛠️ Tech Stack
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` (4-bit quantized via `bitsandbytes`)
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **UI:** Gradio
- **Orchestration:** Manual RAG pipeline (using LangChain utilities for loading/splitting)

## 📦 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/open-pdf-assistant.git
   cd open-pdf-assistant
