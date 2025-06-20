import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(_file_), '..')))
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(_file_), '..')))

import streamlit as st
from pathlib import Path
import random
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

from Preprocessing.document_processor import CVProcessor
from Preprocessing.vector_store import CVVectorStore
from RAG.rag_engine import EnhancedRAGEngine
from RAG.job_matcher import EnhancedJobMatcher
from RAG.cv_summarizer import CVSummarizer
from RAG.job_recommender import JobRecommender
from RAG.hr_question_generator import HRQuestionGenerator

# Initialize modules
processor = CVProcessor(single_chunk=True)
vector_store = CVVectorStore()
rag_engine = EnhancedRAGEngine(vector_store, max_candidates_per_query=15)
job_matcher = EnhancedJobMatcher(vector_store, rag_engine)
summarizer = CVSummarizer()
job_recommender = JobRecommender()
hr_question_generator = HRQuestionGenerator()

st.set_page_config(page_title="Smart Recruiter Assistant", layout="wide")
st.title("Smart Recruiter Assistant")
st.write("Upload CVs, analyze them, ask queries, match jobs, and generate HR questions.")

if "uploaded_cvs" not in st.session_state:
    st.session_state.uploaded_cvs = {}

st.subheader("Upload CVs")
uploaded_files = st.file_uploader("Upload CVs (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("Process CVs"):
    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join("uploaded_files", file.name)
            os.makedirs("uploaded_files", exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            chunks = processor.process_cv(file_path)
            name = os.path.splitext(file.name)[0]
            if chunks:
                st.session_state.uploaded_cvs[name] = chunks[0].page_content
                for chunk in chunks:
                    chunk.metadata["candidate_name"] = name
                    chunk.metadata["source_file"] = file.name
                vector_store.add_cvs(chunks)
        st.success(f"{len(st.session_state.uploaded_cvs)} CV(s) processed.")
    else:
        st.warning("Please upload CV files first.")

# TABS
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Ask", "Match", "Summarize", "Recommend", "HR Questions"])

with tab0:
    st.subheader("Overview of Job Matches")
    job_counts = Counter()
    if st.session_state.uploaded_cvs:
        recommendations = job_recommender.get_top_candidates_for_jobs(st.session_state.uploaded_cvs, top_k=1)
        for job, candidates in recommendations.items():
            job_counts[job] = len(candidates)

        if job_counts:
            df = pd.DataFrame(job_counts.items(), columns=["Job Title", "Count"])
            st.bar_chart(df.set_index("Job Title"))
        else:
            st.info("No job recommendations available yet. Process CVs first.")
    else:
        st.warning("Upload and process CVs to view overview.")

with tab1:
    st.subheader("Ask a question")
    query = st.text_input("Enter your query")
    if st.button("Ask"):
        if st.session_state.uploaded_cvs:
            top_text, all_relevant, top_names = rag_engine.find_top_candidates(query, top_k=5)
            st.text_area("Top Candidates", top_text)
            st.text_area("All Relevant", all_relevant)
        else:
            st.warning("Upload and process CVs first.")

with tab2:
    st.subheader("Match job description")
    job_desc = st.text_area("Paste job description")
    if st.button("Match"):
        if st.session_state.uploaded_cvs:
            results = job_matcher.match_job_to_cvs(job_desc, top_k=5, explain=True)
            for res in job_matcher.format_results(results).split("\n"):
                st.markdown(res)
        else:
            st.warning("Upload and process CVs first.")

with tab3:
    st.subheader("Summarize CVs")
    if st.button("Summarize"):
        for name, content in st.session_state.uploaded_cvs.items():
            summary = summarizer.summarize_cv(content, name)
            st.markdown(f"**{name}**")
            st.success(summary)

with tab4:
    st.subheader("Job Recommendations")
    if st.button("Recommend Jobs"):
        recommendations = job_recommender.get_top_candidates_for_jobs(st.session_state.uploaded_cvs, top_k=3)
        for job, candidates in recommendations.items():
            st.markdown(f"### {job}")
            for candidate, score, reason in candidates:
                st.markdown(f"**Candidate:** {candidate}  ")
                st.markdown(f"**Score:** {score:.2f}  ")
                st.markdown(f"**Reason:** {reason}")
                st.markdown("---")

with tab5:
    st.subheader("HR Interview Questions")
    if st.button("Generate Questions"):
        if st.session_state.uploaded_cvs:
            top_names = list(st.session_state.uploaded_cvs.keys())[:5]
            questions = hr_question_generator.generate_questions_for_top_candidates(st.session_state.uploaded_cvs, top_names)
            for name in top_names:
                st.markdown(f"### {name}")
                for sec, qs in questions[name].items():
                    st.markdown(f"**{sec}**")
                    for q in qs:
                        st.markdown(f"- {q}")
        else:
            st.warning("Upload and process CVs first.")
