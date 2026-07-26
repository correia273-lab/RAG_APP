import tempfile
import os

import streamlit as st

# ── CHANGED ──────────────────────────────────────────────────────────────────
# Removed: process_urls  (URL-based entry point)
# Added:   process_documents  (file-based entry point)
# generate_answer is completely unchanged.
from rag import generate_answer, process_documents
# ─────────────────────────────────────────────────────────────────────────────

# Set page configuration
# ── CHANGED ── Title/icon updated to reflect document (not URL) focus
st.set_page_config(page_title="Smart Document Answer Bot", page_icon="📄", layout="centered")

# App Title
# ── CHANGED ── Heading and subtitle updated to reflect document ingestion
st.title("📄 Your Smart Document Answer Bot")
st.markdown("Ask questions based on content from uploaded PDF, DOCX, or TXT files.")

# ── CHANGED ──────────────────────────────────────────────────────────────────
# Removed: three st.sidebar.text_input() widgets for URL 1/2/3
# Added:   st.sidebar.file_uploader() accepting multiple files of supported types
#
# Why: Instead of typing URLs, users now drag-and-drop or browse for local
# files.  `accept_multiple_files=True` mirrors the previous ability to supply
# up to (and beyond) three sources.  The uploader returns a list of
# UploadedFile objects whose binary content we write to a temp directory so
# the LangChain loaders (which expect file-system paths) can read them.
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("📥 Upload Documents to Process")
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Placeholder for status messages
status_placeholder = st.empty()

# ── CHANGED ──────────────────────────────────────────────────────────────────
# Removed: 'Process URLs' button logic
# Added:   'Process Documents' button logic
#
# Key difference: UploadedFile objects live in memory; LangChain loaders need
# real paths on disk.  We write each uploaded file to a NamedTemporaryFile,
# collect those paths, pass them to process_documents(), then clean up.
# Everything inside the `for status in ...` loop is structurally identical to
# the original — we still iterate the generator and display status messages.
# ─────────────────────────────────────────────────────────────────────────────
if st.sidebar.button("🚀 Process Documents"):
    if not uploaded_files:
        status_placeholder.error("⚠️ Please upload at least one PDF, DOCX, or TXT file.")
    else:
        # Save uploaded files to a temporary directory so loaders can access them
        tmp_dir = tempfile.mkdtemp()
        tmp_paths = []

        for uploaded_file in uploaded_files:
            # Preserve the original file extension so the loader picks the right strategy
            suffix = os.path.splitext(uploaded_file.name)[1]
            tmp_file_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_file_path, "wb") as f:
                f.write(uploaded_file.read())
            tmp_paths.append(tmp_file_path)

        # Generator-based progress display — identical pattern to the original
        for status in process_documents(tmp_paths):
            status_placeholder.info(status)

        # Clean up temp files after ingestion is complete
        for path in tmp_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
# ─────────────────────────────────────────────────────────────────────────────

# Divider
st.markdown("---")

# ── UNCHANGED from here on ───────────────────────────────────────────────────
# Main Input - Question
st.subheader("💬 Ask a Question")
query = st.text_input("Type your question here and press Enter:")

# Show answer if query is submitted
if query:
    try:
        answer, sources = generate_answer(query)
        st.success("✅ Answer Generated!")

        st.markdown("### 🧠 Answer")
        st.write(answer)

        if sources:
            st.markdown("### 📚 Sources")
            for source in sources.strip().split("\n"):
                if source.strip():
                    st.markdown(f"- {source}")
    except RuntimeError:
        st.error("⚠️ You must process the documents first before asking a question.")
# ─────────────────────────────────────────────────────────────────────────────
