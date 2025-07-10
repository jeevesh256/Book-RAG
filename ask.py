import sys
import argparse
import os
import ollama
import chromadb
import signal
from sentence_transformers import SentenceTransformer
from storepdf import store_book

def main():
    # Add response generation state flag
    generating_response = False
    
    def signal_handler(signum, frame):
        nonlocal generating_response
        if generating_response:
            # First Ctrl+C: Stop response generation
            generating_response = False
            print("\n\n[Response generation stopped]")
        else:
            # Second Ctrl+C: Exit program
            print("\nGoodbye!")
            sys.exit(0)
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Parse arguments
    parser = argparse.ArgumentParser(description='Ask questions about a book.')
    parser.add_argument('filename', help='The PDF or TXT file to query')
    args = parser.parse_args()

    try:
        # Store/update the book in database
        print(f"📚 Loading book: {args.filename}")
        store_book(args.filename)
        
        # Load models and database
        model = SentenceTransformer("all-MiniLM-L6-v2")
        chroma_client = chromadb.PersistentClient(path="./book_db")
        collection = chroma_client.get_or_create_collection(name="book")

        def retrieve_and_ask(question):
            """Finds relevant text from the book and streams AI response."""
            nonlocal generating_response
            try:
                question_embedding = model.encode(question).tolist()
                # Get collection count
                collection_count = collection.count()
                n_results = min(3, collection_count) if collection_count > 0 else 1
                
                results = collection.query(
                    query_embeddings=[question_embedding], 
                    n_results=n_results
                )
                retrieved_text = "\n\n".join(results["documents"][0])

                prompt = f"Based only on the book, provide a concise answer:\n{retrieved_text}\n\nQuestion: {question}\nAnswer:"
                response_stream = ollama.chat(model="phi", messages=[{"role": "user", "content": prompt}], stream=True)

                generating_response = True
                for chunk in response_stream:
                    if not generating_response:
                        break
                    print(chunk['message']['content'], end="", flush=True)
                generating_response = False
                
            except Exception as e:
                generating_response = False
                print(f"\n❌ Error in retrieve_and_ask: {str(e)}")

        print("\nReady! You can now ask questions about the book.")
        while True:
            try:
                query = input("\n🔍 Ask something about the book (or type 'exit' to quit): ")
                if query.lower() == 'exit':
                    break
                print("\nResponse: ")
                retrieve_and_ask(query)
                print()
            except (EOFError, KeyboardInterrupt):
                break
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
