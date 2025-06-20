# ===== 📂 pages/01_Upload.py =====

def app():
    import os
    import sys
    import tempfile
    import streamlit as st

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

    from Preprocessing.document_processor import CVProcessor

    # --- Global Styles ---
    st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
            color: #333;
        }
        .main-title {
            color: #017691;
            font-size: 38px;
            font-weight: bold;
            text-align: center;
            margin: 20px 0 10px;
        }
        .section-box {
            background-color: #ffffff;
            padding: 16px;
            border-radius: 10px;
            border: 1px solid #ccc;
            margin-top: 20px;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- Title ---
    st.markdown('<div class="main-title">📂 Upload CVs</div>', unsafe_allow_html=True)

    st.markdown("""
    <p style="text-align:center; font-size:18px; color:#333;">
    Upload candidates' CVs here to process and prepare for analysis.  
    The system will extract and clean the content to help you match top candidates faster.
    </p>
    """, unsafe_allow_html=True)

    # --- Initialize CVProcessor ---
    processor = CVProcessor(
        chunk_size=1000,
        chunk_overlap=200,
        single_chunk=True,
        save_txt=True,
        txt_output_dir="data/txt_cvs"
    )

    # --- Upload UI ---
    st.markdown('<div class="section-box">', unsafe_allow_html=True)

    st.markdown("""
    <p style='font-size:20px; color:#017691; font-weight:600;'>📎 Select PDF CV files to upload</p>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        label="",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple PDF CVs at once."
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} CV(s)")

        if st.button("🚀 Process CVs"):

            if "all_cv_chunks" not in st.session_state:
                st.session_state.all_cv_chunks = []

            for uploaded_file in uploaded_files:
                suffix = os.path.splitext(uploaded_file.name)[1]

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                try:
                    chunks = processor.process_cv(tmp_path)

                    # 🟢 Store chunks in all_cv_chunks session state
                    st.session_state.all_cv_chunks.extend(chunks)

                    st.success(f"✅ Processed and saved {uploaded_file.name} ({len(chunks)} chunks)")

                except Exception as e:
                    st.error(f"❌ Failed to process {uploaded_file.name}: {e}")

                finally:
                    os.remove(tmp_path)

            st.info(f"Total chunks stored: {len(st.session_state.all_cv_chunks)}")


