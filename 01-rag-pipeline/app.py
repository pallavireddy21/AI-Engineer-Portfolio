import streamlit as st
import os
import tempfile
from ingest import ingest
from query import load_resources, retrieve, generate

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 RAG Pipeline")
st.caption("Upload any PDF and ask questions about it using local AI")

# ── Session state ─────────────────────────────────────────────────────────────
# Streamlit reruns the whole script on every interaction
# session_state persists values across reruns — like a memory for the app
if "model" not in st.session_state:
    st.session_state.model = None
if "collection" not in st.session_state:
    st.session_state.collection = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar — PDF Upload ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        # Use filename (without extension) as collection name
        collection_name = os.path.splitext(uploaded_file.name)[0].lower().replace(" ", "_")

        if st.button("⚡ Ingest Document", use_container_width=True):
            with st.spinner("Ingesting document..."):
                # Save uploaded file to a temp location
                # Streamlit gives us bytes — we need a real file path for pypdf
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                # Run ingestion pipeline
                ingest(tmp_path, collection_name=collection_name)

                # Load resources into session state so query tab can use them
                model, collection = load_resources(collection_name)
                st.session_state.model = model
                st.session_state.collection = collection
                st.session_state.collection_name = collection_name
                st.session_state.chat_history = []

                # Clean up temp file
                os.unlink(tmp_path)

            st.success(f"✅ Ready! Ask questions about {uploaded_file.name}")

    # Show currently loaded document
    if st.session_state.collection_name:
        st.info(f"📚 Loaded: {st.session_state.collection_name}")

# ── Main — Chat Interface ──────────────────────────────────────────────────────
if st.session_state.model is None:
    st.info("👈 Upload a PDF and click Ingest Document to get started")
else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    question = st.chat_input("Ask anything about your document...")

    if question:
        # Show user message
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chunks = retrieve(question, st.session_state.model, st.session_state.collection)
                answer = generate(question, chunks)
            st.write(answer)

            # Show retrieved chunks in expander
            with st.expander("📚 Retrieved chunks"):
                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.caption(chunk[:300] + "...")
                    st.divider()

        st.session_state.chat_history.append({"role": "assistant", "content": answer})