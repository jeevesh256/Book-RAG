import fitz  # PyMuPDF for PDF processing
import chromadb
from sentence_transformers import SentenceTransformer
import os
import sys
import shutil

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    """Extracts text and splits it into paragraphs."""
    doc = fitz.open(pdf_path)
    paragraphs = []
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            paragraphs.extend(text.split("\n\n"))  # Split into paragraphs
    return paragraphs

def extract_text_from_txt(txt_path):
    """Extracts text from a text file and splits it into paragraphs."""
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read().strip()
    paragraphs = text.split("\n\n")  # Split into paragraphs
    return paragraphs

def store_book(file_path):
    """Stores book paragraphs in ChromaDB for retrieval."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)
    
    try:
        # Clear existing database if it exists
        db_path = "./book_db"
        if os.path.exists(db_path):
            try:
                shutil.rmtree(db_path, ignore_errors=True)
                print("🗑️  Cleared existing database")
            except PermissionError:
                print(f"❌ Error: Cannot delete database directory.")
                sys.exit(1)

        # Ensure directory exists with proper permissions
        os.makedirs(db_path, exist_ok=True)
        os.chmod(db_path, 0o777)  # Full permissions for testing

        # Initialize ChromaDB within function scope
        chroma_client = chromadb.PersistentClient(path=db_path)
        collection = chroma_client.get_or_create_collection(name="book")

        if file_path.endswith('.pdf'):
            print(f"📖 Extracting text from {file_path}...")
            paragraphs = extract_text_from_pdf(file_path)
        elif file_path.endswith('.txt'):
            print(f"📖 Extracting text from {file_path}...")
            paragraphs = extract_text_from_txt(file_path)
        else:
            print(f"Error: Unsupported file type '{file_path}'.")
            sys.exit(1)

        if not paragraphs:
            print("❌ Error: No text could be extracted from the file.")
            sys.exit(1)

        print(f"🔍 Generating embeddings and storing {len(paragraphs)} paragraphs...")
        for i, para in enumerate(paragraphs):
            if para.strip():
                try:
                    embedding = model.encode(para).tolist()
                    collection.add(
                        ids=[str(i)],
                        documents=[para],
                        embeddings=[embedding],
                    )
                except Exception as e:
                    print(f"❌ Error adding paragraph {i}: {str(e)}")
                    raise

        print(f"✅ Stored {len(paragraphs)} paragraphs in ChromaDB.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if os.path.exists(db_path):
            shutil.rmtree(db_path, ignore_errors=True)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 storepdf.py <filename>")
        sys.exit(1)
    store_book(sys.argv[1])
