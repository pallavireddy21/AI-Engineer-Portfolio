from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# ── 1. Load embedding model + ChromaDB (same ones used in ingest.py) ──────────
def load_resources(collection_name="documents"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name="attention_paper")
    return model, collection

# ── 2. Retrieve the most relevant chunks for a question ───────────────────────
def retrieve(question, model, collection, top_k=3):
    # Embed the question using the same model used during ingestion
    # This puts the question in the same vector space as our chunks
    question_embedding = model.encode([question])[0]

    # Search ChromaDB for the top_k most similar chunks
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=top_k
    )
    # results["documents"][0] is a list of the matching chunk texts
    return results["documents"][0]

# ── 3. Generate an answer using Ollama + retrieved context ────────────────────
def generate(question, chunks):
    # Join retrieved chunks into one block of context
    context = "\n\n---\n\n".join(chunks)

    # This is the core RAG prompt — ground the LLM in only what we retrieved
    prompt = f"""You are an expert on the paper "Attention Is All You Need".
Answer the question using ONLY the context provided below.
If the answer isn't in the context, say "I don't have enough context to answer that."

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# ── 4. Main loop — ask questions until you type 'exit' ────────────────────────
def main():
    print("🔍 Loading resources...")
    model, collection = load_resources()
    print("✅ Ready! Ask anything about the Attention paper.")
    print("Type 'exit' to quit\n")

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        chunks = retrieve(question, model, collection)
        print("\n📚 Retrieved chunks:", len(chunks))
        answer = generate(question, chunks)
        print(f"\n🤖 Answer: {answer}\n")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()