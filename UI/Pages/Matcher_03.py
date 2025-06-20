# pages/03_Matcher.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
from dotenv import load_dotenv
from Preprocessing.vector_store import CVVectorStore
from RAG.rag_engine import EnhancedRAGEngine
from RAG.job_matcher import EnhancedJobMatcher

load_dotenv()

# --- Theme Colors ---
PRIMARY_COLOR = "#017691"
ACCENT_COLOR = "#FF9F1C"
BACKGROUND_COLOR = "#f5f7fa"

# --- Initialize Vector Store & Engine ---
@st.cache_resource
def load_matcher_engine():
    vector_store = CVVectorStore()
    rag_engine = EnhancedRAGEngine(vector_store)
    job_matcher = EnhancedJobMatcher(vector_store, rag_engine)
    return job_matcher

job_matcher = load_matcher_engine()

# --- App Function ---
def app():
    # --- Page Styles ---
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {BACKGROUND_COLOR};
            font-family: 'Poppins', sans-serif;
        }}
        .main-title {{
            color: {PRIMARY_COLOR};
            font-size: 38px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #444;
            font-size: 16px;
            margin-bottom: 20px;
        }}
        .stButton>button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border-radius: 10px;
            font-weight: bold;
            border: none;
            padding: 0.5rem 1.2rem;
        }}
        .stButton>button:hover {{
            background-color: #015566;
            transform: scale(1.03);
        }}
    </style>
    """, unsafe_allow_html=True)

    # --- Title ---
    st.markdown('<div class="main-title">🚀 Smart Job Matcher</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Paste your job description and discover your top candidates!</div>', unsafe_allow_html=True)

    # --- Session State ---
    if "match_results" not in st.session_state:
        st.session_state.match_results = None
    if "job_description_input" not in st.session_state:
        st.session_state.job_description_input = ""

    # --- Input Form ---
    with st.form("matcher_form"):
        job_description = st.text_area(
            "📝 Job Description:",
            value=st.session_state.job_description_input,
            height=300
        )
        submitted = st.form_submit_button("🚀 Match Candidates")

    # --- Handle Submit ---
    if submitted and job_description.strip():
        with st.spinner("🔍 Analyzing job description and finding top candidates..."):
            st.session_state.job_description_input = job_description
            results = job_matcher.match_job_to_cvs(job_description, top_k=5, explain=True)
            formatted_results = job_matcher.format_results(results, show_snippets=True)
            st.session_state.match_results = formatted_results

    # --- Display Results ---
    if st.session_state.match_results:
        st.markdown("## 🎯 Match Results")
        st.markdown(st.session_state.match_results, unsafe_allow_html=True)

    # --- Clear Button ---
    if st.session_state.match_results:
        st.markdown("---")
        if st.button("🗑️ Clear Results"):
            st.session_state.match_results = None
            st.session_state.job_description_input = ""
            st.experimental_rerun()
