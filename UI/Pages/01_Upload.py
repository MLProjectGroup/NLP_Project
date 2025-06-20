# pages/01_Upload.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import tempfile
from Preprocessing.document_processor import CVProcessor

def app():
    # --- Global Page Styles ---
    st.markdown(f"""
    <style>
        .main-title {{
            color: #017691;
            font-size: 38px;
            font-weight: bold;
            text-align: center;
            margin: 20px 0 10px;
        }}
        .section-box {{
            color: #333;
            padding: 16px;
            border: 1px solid #ccc;
            border-radius: 10px;
            background-color: #ffffff;
            margin-top: 20px;
            margin-bottom: 20px;
        }}
        .section-title {{
            font-size: 22px;
            color: #017691;
            font-weight: 600;
            margin-bottom: 12px;
        }}
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
    st.markdown('<div class="section-box">', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "**Select PDF CV files to upload**",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple PDF CVs at once."
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")  # Spacer

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

            st.success("Done ✅")
