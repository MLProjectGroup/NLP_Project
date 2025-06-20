# pages/01_Upload.py

import streamlit as st
import os
import tempfile
from document_processor import CVProcessor

def app():
    # --- Title ---
    st.markdown('<div class="main-title">📂 Upload CVs</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align:center; font-size:18px; color:#444;">
    Upload candidates' CVs here to process and prepare for analysis!
    </p>
    """, unsafe_allow_html=True)

    # --- Initialize Processor ---
    processor = CVProcessor(
        chunk_size=1000,
        chunk_overlap=200,
        single_chunk=True,         # You can change to False if needed
        save_txt=True,
        txt_output_dir="data/txt_cvs"
    )

    # --- File Uploader ---
    uploaded_files = st.file_uploader(
        "Upload PDF CV files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple PDF CVs at once."
    )

    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} CV(s)")

        # --- Process Button ---
        if st.button("🚀 Process CVs"):
            st.info("Processing... please wait ⏳")

            all_chunks = []
            processed_files = []

            # --- Save files temporarily and process
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # --- Process CV
                try:
                    chunks = processor.process_cv(tmp_path)
                    all_chunks.extend(chunks)
                    processed_files.append(uploaded_file.name)
                except Exception as e:
                    st.error(f"❌ Failed to process {uploaded_file.name}: {e}")
                finally:
                    os.remove(tmp_path)

            st.success(f"✅ Processed {len(processed_files)} CV(s) successfully!")

            # --- Show processed file names
            st.markdown("### Processed Files:")
            for fname in processed_files:
                st.write(f"📄 {fname}")

            # --- Show saved .txt files info
            txt_files_info = processor.get_txt_files_info()
            st.markdown("### Saved TXT Files:")
            for info in txt_files_info:
                st.write(f"📄 {info['filename']} — {round(info['size_bytes'] / 1024, 2)} KB")

    else:
        st.info("No CVs uploaded yet. Please upload PDF files to get started.")

    # --- Sidebar Tips ---
    with st.sidebar:
        st.markdown("### Tips for Better Matching:")
        st.markdown("""
        - Upload PDF format CVs
        - Make sure text is extractable (not scanned images)
        - Upload updated CVs
        - Multiple CVs improve matching quality
        """)
