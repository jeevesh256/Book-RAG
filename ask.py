def main():
    try:
        import os
        import ollama
        import chromadb
        from sentence_transformers import SentenceTransformer

        # Check if database exists and has content
        if not os.path.exists("./book_db") or not os.listdir("./book_db"):
            print("Error: No book database found. Please store a book first using storepdf.py")
            return

        # Load embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load ChromaDB
        chroma_client = chromadb.PersistentClient(path="./book_db")
        collection = chroma_client.get_or_create_collection(name="book")

        def retrieve_and_ask(question):
            """Finds relevant text from the book and streams AI response."""
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
                response_stream = ollama.chat(model="jiffy", messages=[{"role": "user", "content": prompt}], stream=True)

                for chunk in response_stream:
                    print(chunk['message']['content'], end="", flush=True)
            except:
                pass

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
    except (KeyboardInterrupt, ImportError, Exception):
        pass
    finally:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
