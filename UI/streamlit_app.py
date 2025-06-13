__import__('pysqlite3')
import sys
import os

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import os
from pathlib import Path
from Preprocessing.document_processor import CVProcessor
from Preprocessing.vector_store import CVVectorStore
from RAG.rag_engine import EnhancedRAGEngine
from RAG.job_matcher import EnhancedJobMatcher
from RAG.cv_summarizer import CVSummarizer

st.set_page_config(page_title="Smart Recruiter Assistant", layout="wide")
st.title("🤖 Smart Recruiter Assistant")
st.write("Upload CVs, ask questions, and match candidates to jobs using AI.")

# Session state for uploaded CVs
if "uploaded_cvs" not in st.session_state:
    st.session_state.uploaded_cvs = []
    st.session_state.cv_texts = []
    st.session_state.candidate_names = []

# Initialize components
processor = CVProcessor(single_chunk=True)
vector_store = CVVectorStore()
rag_engine = EnhancedRAGEngine(vector_store)
matcher = EnhancedJobMatcher(vector_store, rag_engine)
summarizer = CVSummarizer()

# Upload and Analyze CVs
st.subheader("📁 Upload CVs")
uploaded_files = st.file_uploader("Upload multiple CVs (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("🔍 Upload and Analyze"):
    if uploaded_files:
        save_dir = "data/cvs"
        os.makedirs(save_dir, exist_ok=True)
        file_paths = []
        for file in uploaded_files:
            path = os.path.join(save_dir, file.name.replace(" ", "_"))
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            file_paths.append(path)

        chunks = processor.process_multiple_cvs(file_paths)
        vector_store.add_cvs(chunks)

        for path in file_paths:
            st.session_state.uploaded_cvs.append(path)
            st.session_state.cv_texts.append(Path(path).read_text(encoding="utf-8", errors="ignore"))
            st.session_state.candidate_names.append(Path(path).stem)

        st.success(f"{len(file_paths)} CV(s) processed successfully.")
    else:
        st.warning("Please upload at least one file.")

# Tabs for features
tab1, tab2, tab3, tab4 = st.tabs(["❓ Ask Questions", "🎯 Job Matching", "📝 CV Summarizer", "💼 Job Recommender"])

with tab1:
    st.subheader("Ask questions about candidates")
    query = st.text_input("e.g. Who has experience with NLP?")
    if st.button("Ask"):
        if st.session_state.uploaded_cvs:
            answer = rag_engine.query(query)
            st.text_area("Answer", answer, height=250)
        else:
            st.warning("Please upload and analyze CVs first.")

with tab2:
    st.subheader("Match candidates to a job description")
    job_desc = st.text_area("Paste job description here")
    if st.button("Match Candidates"):
        if st.session_state.uploaded_cvs:
            results = matcher.match_job_to_cvs(job_desc, top_k=5)
            formatted = matcher.format_results(results, show_snippets=True)
            st.text_area("Results", formatted, height=400)
        else:
            st.warning("Please upload and analyze CVs first.")

with tab3:
    st.subheader("CV Summaries")
    if st.button("Summarize CVs"):
        if st.session_state.uploaded_cvs:
            summaries = summarizer.summarize_multiple_cvs([
                {"candidate_name": name, "cv_text": text}
                for name, text in zip(st.session_state.candidate_names, st.session_state.cv_texts)
            ])
            for name, summary in summaries.items():
                st.markdown(f"**📄 {name}**")
                st.success(summary)
        else:
            st.warning("Please upload and analyze CVs first.")

with tab4:
    st.subheader("Job Recommendations")
    if st.button("Recommend Jobs"):
        if st.session_state.uploaded_cvs:
            job_examples = [
                {"title": "Data Scientist", "description": "Python, ML, statistics, SQL"},
                {"title": "Frontend Developer", "description": "React, JavaScript, CSS, HTML"},
                {"title": "AI Research Intern", "description": "NLP, LLMs, deep learning"},
            ]
            for name, text in zip(st.session_state.candidate_names, st.session_state.cv_texts):
                st.markdown(f"### 📄 {name}")
                recommendations = matcher.batch_match_multiple_jobs(job_examples, top_k=1)
                for job, result in recommendations.items():
                    top_candidate = result['top_candidates'][0]
                    st.markdown(f"- **💼 {job}** → Score: {top_candidate['similarity_score']:.2f}")
                    st.markdown(f"  🤖 {top_candidate['explanation']}")
                    st.markdown("---")
        else:
            st.warning("Please upload and analyze CVs first.")
