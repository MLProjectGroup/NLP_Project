import streamlit as st
from dotenv import load_dotenv
from RAG.job_matcher import EnhancedJobMatcher
from Preprocessing.vector_store import CVVectorStore  # fallback

load_dotenv()

def app():
    st.markdown('<div class="main-title">📌 Match CVs to Job Description</div>', unsafe_allow_html=True)

    vector_store = CVVectorStore(reset_store=False)  # do NOT clean again
    candidates = vector_store.get_all_candidates()

    if not candidates:
        st.warning("⚠️ No CVs found in the database. Please upload and process CVs first.")
        return

    st.success(f"✅ Found {len(candidates)} candidate(s) in the database.")

    job_description = st.text_area("📄 Paste Job Description Here:", height=250)

    if st.button("🔍 Match Candidates"):
        if job_description.strip():
            matcher = EnhancedJobMatcher(vector_store)
            result = matcher.match_job_to_cvs(job_description, top_k=5, explain=True)
            st.subheader("🎯 Top Matches:")
            st.code(matcher.format_results(result, show_snippets=True), language="markdown")
        else:
            st.warning("⚠️ Please enter a job description.")
