# RAG Pipeline — Chat with Research Papers

A retrieval-augmented generation (RAG) system built from scratch that lets you ask questions about any PDF document using a fully local stack — no API keys, no cloud, no cost.

## What it does
- Extracts and chunks text from a PDF
- Embeds chunks using sentence-transformers locally
- Stores vectors in ChromaDB
- Retrieves the most relevant chunks for any question
- Generates grounded answers using Llama3.2 via Ollama

## Concepts Demonstrated
- Embedding and vector similarity search
- Chunking strategies with overlap
- Retrieval-Augmented Generation (RAG) pattern
- Local LLM inference with Ollama
- Vector database (ChromaDB)

## Stack
- PDF Parsing: pypdf
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector DB: ChromaDB
- LLM: Llama3.2 via Ollama

## Setup

1. Install Ollama and pull the model
ollama pull llama3.2
ollama serve

2. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

3. Add your PDF to the data/ folder

4. Run ingestion
python3 ingest.py

5. Ask questions
python3 query.py

## Example
You: How does multi-head attention work?
Answer: Multi-head attention works by linearly projecting the queries,
keys, and values h times with different learned projections...

## Structure
01-rag-pipeline/
  data/              source PDFs
  ingest.py          chunk + embed + store
  query.py           retrieve + generate
  requirements.txt
  README.md
