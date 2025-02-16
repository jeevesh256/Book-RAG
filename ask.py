import ollama
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load ChromaDB
chroma_client = chromadb.PersistentClient(path="./book_db")
collection = chroma_client.get_or_create_collection(name="book")

def retrieve_and_ask(question):
    """Finds relevant text from the book and streams AI response."""
    question_embedding = model.encode(question).tolist()

    # Retrieve top 3 relevant paragraphs
    results = collection.query(query_embeddings=[question_embedding], n_results=3)
    retrieved_text = "\n\n".join(results["documents"][0])  # Combine retrieved paragraphs

    # Create prompt
    prompt = f"Based only on the book, provide a concise answer:\n{retrieved_text}\n\nQuestion: {question}\nAnswer:"

    # Stream response from Ollama
    response_stream = ollama.chat(model="jiffy", messages=[{"role": "user", "content": prompt}], stream=True)

    # Print response token-by-token
    for chunk in response_stream:
        print(chunk['message']['content'], end="", flush=True)

if __name__ == "__main__":
    while True:
        query = input("\n🔍 Ask something about the book (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        print("\nJiffy: ")
        retrieve_and_ask(query)
        print()  # Add a newline after response
