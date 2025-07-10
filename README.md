# Book RAG

This project enables **question-answering on books** using **Retrieval-Augmented Generation (RAG)**. It extracts text from PDFs, stores it in a **ChromaDB** vector database, retrieves relevant context, and generates responses using an **Ollama** AI model.  
A Phi-2 (2.7b) model is used for generating responses, with some parameter modifications applied. The Modelfile is included in the repository for easy setup and replication.

## Features
- **Extracts text** from PDFs and stores them as embeddings.
- **Retrieves relevant content** based on user queries.
- **Streams AI-generated responses** for faster feedback.
- **Fully local RAG setup** (no cloud required, runs on your machine).

## Setup & Installation
### 1️⃣ Install Dependencies
Ensure you have **Python 3.8+** installed, then run:
```bash
pip install ollama chromadb sentence-transformers pymupdf
```

### 2️⃣ Ask Questions About a Book
Simply run the Q&A chatbot with your book:
```bash
python3 ask.py your_book.pdf
```
This will:
1. Load your book (automatically replacing any previously stored book)
2. Start an interactive Q&A session

> Supported file formats: PDF and TXT

## File Overview
- `store.py` → Extracts text from a PDF and stores it in **ChromaDB**.
- `ask.py` → Loads the stored text, retrieves relevant content, and queries **Ollama AI** for answers.
- `book_db` → ChromaDB database directory where the extracted book data is stored.

## Notes
- This implementation ensures **retrieval-augmented generation (RAG)** by grounding responses in stored book content.
- The **ChromaDB** database enables fast retrieval.
- The AI **only answers based on the book**, ensuring **accurate** responses without hallucinating.
- Any Ollama-compatible model can be used for response generation by changing the model name in `ask.py`.


