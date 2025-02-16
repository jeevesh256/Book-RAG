import fitz  # PyMuPDF for PDF processing
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./book_db")
collection = chroma_client.get_or_create_collection(name="book")

def extract_text_from_pdf(pdf_path):
    """Extracts text and splits it into paragraphs."""
    doc = fitz.open(pdf_path)
    paragraphs = []
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            paragraphs.extend(text.split("\n\n"))  # Split into paragraphs
    return paragraphs

def store_book(pdf_path):
    """Stores book paragraphs in ChromaDB for retrieval."""
    print(f"📖 Extracting text from {pdf_path}...")
    paragraphs = extract_text_from_pdf(pdf_path)

    print(f"🔍 Generating embeddings and storing {len(paragraphs)} paragraphs...")
    for i, para in enumerate(paragraphs):
        embedding = model.encode(para).tolist()  # Convert embedding to list
        collection.add(
            ids=[str(i)],
            documents=[para],
            embeddings=[embedding],
        )

    print(f"✅ Stored {len(paragraphs)} paragraphs in ChromaDB.")

if __name__ == "__main__":
    store_book("test.pdf")
