# Pages/01_Upload.py

import sys
import os
import tempfile
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Preprocessing.document_processor import CVProcessor
from Preprocessing.vector_store import CVVectorStore


def app():
    # --- Global Page Styles ---
    st.markdown("""
    <style>
        .main-title {
            color: #017691;
            font-size: 38px;
            font-weight: bold;
            text-align: center;
            margin: 20px 0 10px;
        }
        .section-box {
            color: #333;
            padding: 16px;
            border: 1px solid #ccc;
            border-radius: 10px;
            background-color: #ffffff;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 22px;
            color: #017691;
            font-weight: 600;
            margin-bottom: 12px;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- Title ---
    st.markdown('<div class="main-title">📂 Upload CVs</div>', unsafe_allow_html=True)

    st.markdown("""
    <p style="text-align:center; font-size:18px; color:#333; max-width: 700px; margin: auto; line-height: 1.6;">
    Upload candidates' CVs here to process and prepare for analysis.  
    The system will extract and clean the content to help you match top candidates faster.
    </p>
    """, unsafe_allow_html=True)

    # --- Initialize Processor ---
    processor = CVProcessor(
        chunk_size=1000,
        chunk_overlap=200,
        single_chunk=True,
        save_txt=True,
        txt_output_dir="data/txt_cvs"
    )

    # --- Upload Section ---
    uploaded_files = st.file_uploader(
        "**Select PDF CV files to upload**",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple PDF CVs at once."
    )

    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} CV(s)")

        if st.button("🚀 Process CVs"):
            all_chunks = []
            processed_files = []

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                try:
                    chunks = processor.process_cv(tmp_path)
                    all_chunks.extend(chunks)
                    processed_files.append(uploaded_file.name)
                except Exception as e:
                    st.error(f"❌ Failed to process {uploaded_file.name}: {e}")
                finally:
                    os.remove(tmp_path)

            if all_chunks:
                vector_store = CVVectorStore(reset_store=True)  # clean ONCE only
                vector_store.add_cvs(all_chunks)
                st.session_state["vector_store"] = vector_store
                st.success("✅ CVs processed and stored in the Chroma vector database!")
            else:
                st.warning("⚠️ No valid CV chunks found to process.")
