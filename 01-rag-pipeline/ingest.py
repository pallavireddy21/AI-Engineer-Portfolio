import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# ── 1. Load the PDF and extract raw text ──────────────────────────────────────
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# ── 2. Split text into overlapping chunks ─────────────────────────────────────
# Why chunks? LLMs have a context limit — you can't feed a whole paper at once.
# Why overlap? So sentences at chunk boundaries don't lose context.
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # move forward but keep last 50 words
    return chunks

# ── 3. Embed chunks and store in ChromaDB ─────────────────────────────────────
def ingest(pdf_path):
    print("📄 Loading PDF...")
    text = load_pdf(pdf_path)
    print(f"✅ Extracted {len(text)} characters")

    print("✂️  Chunking text...")
    chunks = chunk_text(text)
    print(f"✅ Created {len(chunks)} chunks")

    print("🔢 Loading embedding model...")
    # This model runs locally — no API call, no cost
    # It converts text → a list of 384 numbers that capture meaning
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("🧮 Embedding chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"✅ Created {len(embeddings)} embeddings of size {len(embeddings[0])}")

    print("💾 Storing in ChromaDB...")
    # ChromaDB stores locally in a folder called chroma_db/
    client = chromadb.PersistentClient(path="chroma_db")
    
    # A collection is like a table — holds our vectors + the original text
    collection = client.get_or_create_collection(name="attention_paper")

    # Add everything to the collection
    collection.add(
        documents=chunks,                                    # original text
        embeddings=embeddings.tolist(),                      # vectors
        ids=[f"chunk_{i}" for i in range(len(chunks))]      # unique IDs
    )
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
    print("🎉 Ingestion complete!")

# ── 4. Run it ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ingest("data/attention_is_all_you_need.pdf")