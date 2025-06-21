import sys
import os
import streamlit as st
from pathlib import Path
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Preprocessing.document_processor import CVProcessor
from Preprocessing.vector_store import CVVectorStore
from RAG.rag_engine import EnhancedRAGEngine
from RAG.job_matcher import EnhancedJobMatcher
from RAG.cv_summarizer import CVSummarizer
from RAG.job_recommender import JobRecommender
from RAG.hr_question_generator import HRQuestionGenerator

# --- App Config ---
st.set_page_config(
    page_title="Smart Recruiter Assistant 🤖",
    layout="wide",
    page_icon="https://raw.githubusercontent.com/MLProjectGroup/NLP_Project/main/UI/assets/hr_man.png"
)
    

# --- Theme Colors ---
theme = {
    "primary": "#017691",       # Blue
    "secondary": "#FF9F1C",     # Orange
    "accent": "#e0e0e0",
    "background": "#dce3e4",
    "text": "#222222",          # Black
}

# --- Google Fonts ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# --- Daily Tips ---
daily_tips = [
    "Always personalize your hiring message!",
    "Look beyond keywords, consider potential.",
    "Soft skills matter as much as experience.",
    "Diversity is a strength in hiring!",
    "Hiring is like dating... look for culture fit!",
]

# --- CSS Styles ---
st.markdown(f"""
<style>
body, .stApp {{
    background-color: {theme['background']};
    font-family: 'Poppins', sans-serif;
    color: {theme['text']};
}}
.header {{
    background-color: {theme['primary']};
    padding: 15px;
    color: white;
    font-weight: bold;
    font-size: 26px;
    top: 0;
    width: 100%;
    z-index: 1000;
    display: flex;
    justify-content: center;
    align-items: center;
}}

h1, h2, h3, .main-title {{
    color: {theme['primary']};
    font-weight: 700;
}}

p, label, .stText, .stMarkdown {{
    color: {theme['text']};
}}

.stButton > button {{
    background-color: {theme['primary']} !important;
    color: white !important;
    font-weight: 600;
}}

.stFileUploader > div > div {{
    background-color: white !important;
    border: 2px dashed #ccc !important;
    border-radius: 12px !important;
    padding: 20px !important;
}}

.centered-image {{
    text-align: center;
    margin: 30px 0;
}}

.centered-image img {{
    width: 400px;
    border-radius: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    transition: transform 0.3s ease;
}}

.centered-image img:hover {{
    transform: scale(1.05);
}}

.quote {{
    color: {theme['primary']};
    font-style: italic;
    font-weight: 600;
    font-size: 20px;
    text-align: center;
    margin: 20px 0 40px 0;
}}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown(f'<div class="header">🤖 Smart Recruiter Assistant</div>', unsafe_allow_html=True)

# --- Home Section ---
st.markdown('<div class="centered-image">', unsafe_allow_html=True)
st.image("https://raw.githubusercontent.com/MLProjectGroup/NLP_Project/main/UI/assets/hr_man.png", width=400)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<p class="quote">💡 Daily Tip: {random.choice(daily_tips)}</p>', unsafe_allow_html=True)


# --- Main App Logic ---

processor = CVProcessor(single_chunk=True)
vector_store = CVVectorStore(reset_store=True)
rag_engine = EnhancedRAGEngine(vector_store)
job_matcher = EnhancedJobMatcher(vector_store, rag_engine)
summarizer = CVSummarizer()
job_recommender = JobRecommender()
hr_question_generator = HRQuestionGenerator()

# Upload CVs
uploaded_files = st.file_uploader(
    "Upload Candidate CVs", type=["pdf", "docx", "doc", "txt"], accept_multiple_files=True)

candidate_cv_map = {}
all_documents = []
all_candidates = []

if uploaded_files:
    os.makedirs("temp", exist_ok=True)
    for uploaded_file in uploaded_files:
        content_path = f"temp/{uploaded_file.name}"
        with open(content_path, "wb") as f:
            f.write(uploaded_file.read())

        chunks = processor.process_cv(content_path)
        candidate_name = Path(uploaded_file.name).stem
        normalized_name = candidate_name.replace("_", " ").title()

        if chunks:
            candidate_cv_map[normalized_name] = chunks[0].page_content
            all_candidates.append(normalized_name)
            for chunk in chunks:
                chunk.metadata["candidate_name"] = normalized_name
                chunk.metadata["source_file"] = uploaded_file.name
            all_documents.extend(chunks)

    vector_store.add_cvs(all_documents)
    st.success(f"✅ Uploaded and processed {len(all_candidates)} CVs.")

# Query box
st.markdown("---")
query = st.text_input("🔎 Enter Custom Query (e.g., Who has React experience?)")

if st.button("Run Query") and query.strip():
    try:
        top_text = rag_engine.find_top_candidates(query, top_k=5)
        all_relevant = rag_engine.get_all_candidates_for_skill(query)
        st.subheader("🎯 Top Candidates")
        st.code(top_text[0], language="markdown")

        st.subheader("📌 All Relevant Candidates")
        st.text(all_relevant)
    except Exception as e:
        st.error(f"❌ Failed to process query: {e}")

# Job matching
st.markdown("---")
st.subheader("📋 Match Job Description to Candidates")
job_description = st.text_area("Paste Job Description")

if st.button("Match Candidates to Job") and job_description.strip():
    try:
        results = job_matcher.match_job_to_cvs(job_description, top_k=5, explain=True)
        formatted = job_matcher.format_results(results, show_snippets=True)
        st.subheader("🧠 Matching Results")
        st.code(formatted, language="markdown")
    except Exception as e:
        st.error(f"❌ Job matching failed: {e}")

# Generate HR Questions
st.markdown("---")
st.subheader("💬 Generate HR Questions")
if st.button("Generate HR Questions"):
    try:
        hr_qs = hr_question_generator.generate_questions_for_top_candidates(candidate_cv_map, all_candidates[:5])
        for name, sections in hr_qs.items():
            st.markdown(f"**{name}**")
            for sec, qs in sections.items():
                st.markdown(f"*{sec}*:")
                for q in qs:
                    st.markdown(f"- {q}")
    except Exception as e:
        st.error(f"❌ HR question generation failed: {e}")

# Job Recommendation
st.markdown("---")
st.subheader("🚀 AI Job Recommendations")
if st.button("Get Job Recommendations"):
    try:
        recommendations = job_recommender.get_top_candidates_for_jobs(candidate_cv_map, top_k=5)
        for job, names in recommendations.items():
            st.markdown(f"### {job}")
            for i, name in enumerate(names, 1):
                st.markdown(f"{i}. {name}")
    except Exception as e:
        st.error(f"❌ Job recommendation failed: {e}")
