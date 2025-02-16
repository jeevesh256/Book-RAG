# BookRAG

This project enables **question-answering on books** using **Retrieval-Augmented Generation (RAG)**. It extracts text from PDFs, stores it in a **ChromaDB** vector database, retrieves relevant context, and generates responses using an **Ollama** AI model.
A Phi-2 (2.7B) model is used for generating responses, with some parameter modifications applied. The Modelfile is included in the repository for easy setup and replication.

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

### 2️⃣ Store a Book in the Database
Run the following command to extract text from a **PDF** and store it in ChromaDB:
```bash
python3 store.py
```
> Make sure to replace `test.pdf` in `store.py` with your book's filename.

### 3️⃣ Ask Questions About the Book
Once the book is stored, start the **Q&A chatbot**:
```bash
python3 ask.py
```
Then, enter your questions based on the book!

## File Overview
- `store.py` → Extracts text from a PDF and stores it in **ChromaDB**.
- `ask.py` → Loads the stored text, retrieves relevant content, and queries **Ollama AI** for answers.
- `book_db` → ChromaDB database directory where the extracted book data is stored.

## Updating the Book
If you want to replace the stored book with a new one:
1. Delete the existing database folder:
   ```bash
   rm -rf book_db/
   ```
2. Run `store.py` again with a new PDF.

## Notes
- This implementation ensures **retrieval-augmented generation (RAG)** by grounding responses in stored book content.
- The **ChromaDB** database enables fast retrieval.
- The AI **only answers based on the book**, ensuring **accurate** responses without hallucinating.
- Any Ollama-compatible model can be used for response generation by changing the model name in `ask.py`.


