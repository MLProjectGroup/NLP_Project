import streamlit as st
import os
from pathlib import Path
import logging
import re
from collections import defaultdict
import time # Import time for potential delays if needed
import sys
import os
import streamlit as st
from pathlib import Path
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Adjust sys.path if necessary for local module imports in a deployed environment
import sys
# If running on Streamlit Cloud, pysqlite3 might be needed
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass # Use default sqlite3 if pysqlite3 is not available

# Append parent directory to sys.path to allow imports from 'Preprocessing', 'RAG'
# This assumes 'app.py' is in the root and 'Preprocessing', 'RAG' are subdirectories.
# Adjust this path if your directory structure is different.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir) # If app.py is in the root, its parent is the root
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import your modules
try:
    from Preprocessing.document_processor import CVProcessor
    from Preprocessing.vector_store import CVVectorStore
    from RAG.rag_engine import EnhancedRAGEngine
    from RAG.job_matcher import EnhancedJobMatcher
    from RAG.cv_summarizer import CVSummarizer
    from RAG.job_recommender import JobRecommender
    from RAG.hr_question_generator import HRQuestionGenerator
except ImportError as e:
    st.error(f"Error importing modules: {e}. Please ensure your project structure and module names are correct.")
    st.stop()


# Configure logging for Streamlit (optional, but good for debugging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Helper Functions (adapted from your main.py) ---
def extract_skill_from_query(query: str) -> str:
    """Extract the main skill or requirement from a query"""
    query_clean = query.lower().replace('?', '').strip()
    patterns = [
        r'who has (.+?)(?:\?|$)', r'who knows (.+?)(?:\?|$)',
        r'candidates with (.+?)(?:\?|$)', r'experience in (.+?)(?:\?|$)',
        r'experienced in (.+?)(?:\?|$)', r'skills in (.+?)(?:\?|$)',
        r'knowledge of (.+?)(?:\?|$)', r'proficient in (.+?)(?:\?|$)',
        r'expertise in (.+?)(?:\?|$)', r'background in (.+?)(?:\?|$)',
        r'qualified in (.+?)(?:\?|$)', r'certified in (.+?)(?:\?|$)',
        r'degree in (.+?)(?:\?|$)', r'studied (.+?)(?:\?|$)',
        r'worked with (.+?)(?:\?|$)', r'familiar with (.+?)(?:\?|$)'
    ]
    for pattern in patterns:
        match = re.search(pattern, query_clean)
        if match:
            skill = match.group(1).strip()
            return skill.replace(' experience', '').replace(' skills', '')
    stop_words = ['who', 'what', 'which', 'where', 'when', 'how', 'has', 'have', 'is', 'are',
                  'the', 'a', 'an', 'candidates', 'candidate', 'people', 'person']
    words = query_clean.split()
    key_words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(key_words) if key_words else "relevant experience"

def normalize_candidate_name(name: str) -> str:
    """Normalize candidate names to avoid duplicates"""
    if not name:
        return "Unknown Candidate"
    name = re.sub(r'(_CV|_cv|Cv|_Resume|Resume)$', '', name, flags=re.IGNORECASE)
    name = name.replace("_", " ").replace("-", " ")
    return " ".join(name.split()).strip().title()

def process_custom_query_streamlit(rag_engine_instance: EnhancedRAGEngine, query: str):
    """Process any custom query and return formatted results with candidate names for Streamlit"""
    logger.info(f"Processing custom query: {query}")
    skill_or_requirement = extract_skill_from_query(query)
    logger.info(f"Extracted skill/requirement: {skill_or_requirement}")

    top_candidates_text, top_candidate_names = rag_engine_instance.find_top_candidates(query, top_k=5)
    all_relevant = rag_engine_instance.get_all_candidates_for_skill(skill_or_requirement)
    return top_candidates_text, all_relevant, top_candidate_names

# --- Streamlit App Setup ---
st.set_page_config(page_title="Smart Recruiter Assistant", layout="wide")
st.title("🤖 Smart Recruiter Assistant")
st.write("Upload CVs, ask questions, and match candidates to jobs using AI.")

# Initialize session state variables
if "uploaded_cv_paths" not in st.session_state:
    st.session_state.uploaded_cv_paths = []
if "candidate_cv_map" not in st.session_state:
    st.session_state.candidate_cv_map = {}
if "vector_store_initialized" not in st.session_state:
    st.session_state.vector_store_initialized = False

# Instantiate core components only once
@st.cache_resource
def init_components():
    """Initializes and caches the heavy components like models and vector stores."""
    processor = CVProcessor(single_chunk=True)
    vector_store = CVVectorStore()
    rag_engine = EnhancedRAGEngine(vector_store, max_candidates_per_query=15)
    job_matcher = EnhancedJobMatcher(vector_store, rag_engine)
    summarizer = CVSummarizer()
    job_recommender = JobRecommender()
    hr_question_generator = HRQuestionGenerator()
    return processor, vector_store, rag_engine, job_matcher, summarizer, job_recommender, hr_question_generator

processor, vector_store, rag_engine, job_matcher, summarizer, job_recommender, hr_question_generator = init_components()

# Define a default job list (from your template)
job_list = [
    {"title": "AI Research Intern", "description": "Deep learning, NLP, PyTorch, and academic research experience preferred."},
    {"title": "Computer Vision Engineer", "description": "Build image recognition using CNNs, OpenCV, and Python."},
    {"title": "Data Analyst (Tableau)", "description": "Strong Tableau, SQL, Excel, statistics background needed."},
    {"title": "NLP Engineer", "description": "Build language models, transformers, and pipelines for text classification."},
    {"title": "Machine Learning Engineer", "description": "Develop end-to-end ML pipelines using Python, Scikit-learn, TensorFlow."},
    {"title": "Frontend Developer", "description": "React, TypeScript, Tailwind, and UI testing experience required."},
    {"title": "Backend Developer", "description": "Develop APIs using Django or FastAPI, PostgreSQL, Redis."},
    {"title": "Data Scientist", "description": "Data modeling, predictive analytics, A/B testing, and feature engineering."},
    {"title": "Generative AI Engineer", "description": "Work with LLMs, prompt engineering, RAG pipelines, and vector DBs."},
    {"title": "Cloud DevOps Engineer", "description": "Set up CI/CD, Docker, Kubernetes, and deploy ML systems on AWS."},
    {"title": "Cybersecurity Analyst", "description": "Monitor systems for threats, use SIEM tools, and perform vulnerability analysis."},
    {"title": "Business Intelligence Developer", "description": "Build dashboards using Power BI, DAX, and data warehousing skills."},
    {"title": "AI Product Manager", "description": "Define product roadmaps for ML systems and coordinate with cross-functional teams."},
    {"title": "Mobile App Developer", "description": "Develop Flutter or Android apps and integrate REST APIs."},
    {"title": "Software QA Engineer", "description": "Write unit tests, automate UI testing, and use tools like Selenium."},
    {"title": "Bioinformatics Researcher", "description": "Analyze DNA data using ML, sequence modeling, and biomedical NLP."},
    {"title": "Computer Science Instructor", "description": "Teach algorithms, data structures, Python, and ML to students."},
    {"title": "AI for Healthcare Specialist", "description": "Apply ML to medical data, detect anomalies, and support diagnosis."},
    {"title": "Robotics Engineer", "description": "Design intelligent robotic systems using sensor fusion and control theory."},
    {"title": "Speech Recognition Scientist", "description": "Build models for audio transcription, speech-to-text using transformers."}
]

# --- CV Upload and Processing Section ---
st.subheader("📁 Upload CVs")
uploaded_files = st.file_uploader("Upload multiple CVs (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if st.button("🔍 Upload and Analyze CVs"):
    if uploaded_files:
        temp_cv_dir = "uploaded_cvs_temp"
        os.makedirs(temp_cv_dir, exist_ok=True)
        
        st.session_state.uploaded_cv_paths = []
        st.session_state.candidate_cv_map = {}
        all_documents = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, file in enumerate(uploaded_files):
            file_path = os.path.join(temp_cv_dir, file.name.replace(" ", "_"))
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            st.session_state.uploaded_cv_paths.append(file_path)
            
            status_text.text(f"Processing CV: {file.name} ({i+1}/{len(uploaded_files)})")
            try:
                chunks = processor.process_cv(file_path)
                candidate_name = Path(file_path).stem
                normalized_name = normalize_candidate_name(candidate_name)

                if chunks:
                    st.session_state.candidate_cv_map[normalized_name] = chunks[0].page_content
                    for chunk in chunks:
                        chunk.metadata["candidate_name"] = normalized_name
                        chunk.metadata["source_file"] = file_path
                    all_documents.extend(chunks)
                else:
                    st.warning(f"Could not extract content from {file.name}.")
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if all_documents:
            with st.spinner("Embedding CVs..."):
                vector_store.add_cvs(all_documents)
            st.session_state.vector_store_initialized = True
            st.success(f"{len(st.session_state.uploaded_cv_paths)} CV(s) processed and embedded successfully.")
        else:
            st.warning("No valid CVs were processed for embedding.")
        
        progress_bar.empty()
        status_text.empty()
    else:
        st.warning("Please upload at least one file.")

# --- Tabs for different functionalities ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["❓ Ask Questions", "🎯 Job Matching", "📝 CV Summarizer", "💼 Job Recommender", "🗣️ HR Interview Questions"])

with tab1:
    st.subheader("Ask questions about candidates")
    query = st.text_input("Example: who is experienced with Deep Learning?")
    if st.button("Ask Question"):
        if st.session_state.vector_store_initialized and st.session_state.uploaded_cv_paths:
            with st.spinner("Searching for answers..."):
                top_candidates_text, all_relevant, top_query_candidates = process_custom_query_streamlit(rag_engine, query)
            
            st.markdown("### Top Candidates Ranking:")
            if top_candidates_text:
                st.write(top_candidates_text)
            else:
                st.info("No top candidates found for this query.")
            
            st.markdown("### All Relevant Candidates:")
            if all_relevant:
                st.write(all_relevant)
            else:
                st.info("No other relevant candidates found for this query.")
            
            st.session_state.top_query_candidates_for_hr = top_query_candidates # Store for HR questions tab
        else:
            st.warning("Please upload and analyze CVs first to ask questions.")

with tab2:
    st.subheader("Match candidates to a job description")
    job_desc = st.text_area("Paste job description here:", height=200, 
                            value="""
Job Title: Full Stack Developer
Key Responsibilities:
- Develop and maintain web applications using JavaScript, React, and Node.js
- Design and implement responsive UI with HTML and CSS
- Manage databases using MongoDB
- Containerize applications with Docker
- Deploy applications on AWS cloud infrastructure

Required Skills (Weighted):
JavaScript (10), React (10), Node.js (10), HTML (8), CSS (8), MongoDB (8)

Preferred Skills (Weighted):
TypeScript (7), Angular (7), Vue.js (7), Docker (6), AWS (6)

Experience Requirements:
- Minimum 3 years professional experience in full-stack development
- Portfolio of deployed web applications
""")
    if st.button("Match Candidates to Job"):
        if st.session_state.vector_store_initialized and st.session_state.uploaded_cv_paths:
            if job_desc.strip():
                with st.spinner("Matching candidates..."):
                    try:
                        job_match_result = job_matcher.match_job_to_cvs(job_desc, top_k=5, explain=True)
                        if job_match_result:
                            st.markdown("### Matching Results:")
                            for candidate_name, match_info in job_match_result.items():
                                st.markdown(f"#### 📄 Candidate: `{candidate_name}`")
                                st.markdown(f"**💡 Score:** `{match_info['total_score']:.2f}`")
                                st.markdown(f"**Relevant Sections:**")
                                for section, score in match_info['matched_sections'].items():
                                    st.write(f"- **{section}**: Score {score:.2f}")
                                st.markdown(f"**🤖 Explanation:** {match_info['explanation']}")
                                st.markdown("---")
                        else:
                            st.info("No candidates matched the job description with a significant score.")
                    except Exception as e:
                        st.error(f"Error during job matching: {e}")
            else:
                st.warning("Please paste a job description.")
        else:
            st.warning("Please upload and analyze CVs first to match candidates.")

with tab3:
    st.subheader("CV Summaries")
    if st.button("Generate All CV Summaries"):
        if st.session_state.uploaded_cv_paths and st.session_state.candidate_cv_map:
            st.markdown("### All Candidate Summaries:")
            for i, (candidate_name, cv_content) in enumerate(st.session_state.candidate_cv_map.items()):
                st.markdown(f"#### 📄 Candidate: `{candidate_name}`")
                if cv_content:
                    try:
                        with st.spinner(f"Summarizing {candidate_name}..."):
                            cv_summary = summarizer.summarize_cv(cv_content, candidate_name)
                        st.success(cv_summary)
                    except Exception as e:
                        st.error(f"Error summarizing CV for {candidate_name}: {e}")
                else:
                    st.warning(f"No content found for {candidate_name}.")
                st.markdown("---")
        else:
            st.warning("Please upload and analyze CVs first to generate summaries.")

with tab4:
    st.subheader("Job Recommendations for Candidates")
    if st.button("Get Job Recommendations"):
        if st.session_state.uploaded_cv_paths and st.session_state.candidate_cv_map:
            with st.spinner("Generating job recommendations..."):
                try:
                    # job_recommender.get_top_candidates_for_jobs works by finding best candidate for each job
                    # We want to find best jobs for each candidate.
                    # This requires iterating through candidates and recommending jobs from job_list
                    
                    st.markdown("### Job Recommendations per Candidate:")
                    for candidate_name, cv_content in st.session_state.candidate_cv_map.items():
                        if cv_content:
                            st.markdown(f"#### 📄 Candidate: `{candidate_name}`")
                            ranked_jobs = job_recommender.recommend_jobs_for_candidate(cv_content, job_list, top_k=3, return_output=True)
                            if ranked_jobs:
                                for job, score, reason in ranked_jobs:
                                    st.markdown(f"**💼 {job['title']}** (Score: {score:.2f})")
                                    st.markdown(f"**Reason:** {reason}")
                                st.markdown("---")
                            else:
                                st.info(f"No specific job recommendations for {candidate_name} based on the provided list.")
                        else:
                            st.warning(f"No content found for {candidate_name} to recommend jobs.")
                except Exception as e:
                    st.error(f"Error during job recommendations: {e}")
        else:
            st.warning("Please upload and analyze CVs first to get job recommendations.")

with tab5:
    st.subheader("Generate HR Interview Questions")
    if "top_query_candidates_for_hr" in st.session_state and st.session_state.top_query_candidates_for_hr:
        st.info(f"Questions will be generated for candidates identified in the 'Ask Questions' tab: {', '.join(st.session_state.top_query_candidates_for_hr)}")
        
        if st.button("Generate HR Questions"):
            if st.session_state.vector_store_initialized and st.session_state.candidate_cv_map:
                matched_candidates_for_questions = []
                for candidate_from_query in st.session_state.top_query_candidates_for_hr:
                    if candidate_from_query in st.session_state.candidate_cv_map:
                        matched_candidates_for_questions.append(candidate_from_query)
                    else:
                        # Fallback for case-insensitive matching if needed
                        found = False
                        for stored_candidate in st.session_state.candidate_cv_map.keys():
                            if candidate_from_query.lower() == stored_candidate.lower():
                                matched_candidates_for_questions.append(stored_candidate)
                                found = True
                                break
                        if not found:
                            st.warning(f"Could not find CV content for '{candidate_from_query}' to generate questions.")

                if matched_candidates_for_questions:
                    with st.spinner("Generating HR interview questions..."):
                        try:
                            hr_questions = hr_question_generator.generate_questions_for_top_candidates(
                                st.session_state.candidate_cv_map,
                                matched_candidates_for_questions
                            )
                            st.markdown("### Generated HR Questions:")
                            for candidate, questions_dict in hr_questions.items():
                                st.markdown(f"#### 📄 Candidate: `{candidate}`")
                                if questions_dict:
                                    for category, q_list in questions_dict.items():
                                        st.markdown(f"**{category}:**")
                                        for q in q_list:
                                            st.write(f"- {q}")
                                else:
                                    st.info(f"No specific questions generated for {candidate}.")
                                st.markdown("---")
                        except Exception as e:
                            st.error(f"Error generating HR questions: {e}")
                else:
                    st.warning("No matching candidates found from the query results to generate HR questions.")
            else:
                st.warning("Please upload and analyze CVs first, and then run a query in the 'Ask Questions' tab.")
    else:
        st.info("Run a query in the 'Ask Questions' tab first to identify candidates for HR question generation.")

st.markdown("---")
st.markdown("### About")
st.markdown("This Smart Recruiter Assistant leverages various AI capabilities to streamline the recruitment process. "
            "It uses **document processing** for CV extraction, **vector databases** for efficient search, "
            "**Retrieval-Augmented Generation (RAG)** for answering questions, "
            "**LLMs** for summarization, job matching, and question generation.")

st.sidebar.markdown("### Instructions")
st.sidebar.markdown(
    """
    1.  **Upload CVs**: Click 'Browse files' to upload PDF or DOCX CVs. Then click 'Upload and Analyze CVs'.
    2.  **Ask Questions**: Go to the 'Ask Questions' tab, type a query (e.g., "who is experienced with Python?"), and click 'Ask Question'.
    3.  **Job Matching**: In the 'Job Matching' tab, paste a job description and click 'Match Candidates to Job'.
    4.  **CV Summarizer**: In the 'CV Summarizer' tab, click 'Generate All CV Summaries' to get concise summaries.
    5.  **Job Recommender**: In the 'Job Recommender' tab, click 'Get Job Recommendations' to see relevant jobs for uploaded candidates.
    6.  **HR Interview Questions**: After asking a query in the 'Ask Questions' tab, go to this tab to generate HR questions for the top candidates.
    """
)
