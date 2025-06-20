import os
import sys
from pathlib import Path
import re
import time
import logging
from typing import List, Dict, Tuple
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from Preprocessing.document_processor import CVProcessor
from Preprocessing.vector_store import CVVectorStore
from RAG.rag_engine import EnhancedRAGEngine
from RAG.job_matcher import EnhancedJobMatcher
from RAG.cv_summarizer import CVSummarizer
from RAG.job_recommender import JobRecommender
from RAG.hr_question_generator import HRQuestionGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if "uploaded_cvs" not in st.session_state:
    st.session_state.uploaded_cvs = {}
if "candidate_cv_map" not in st.session_state:
    st.session_state.candidate_cv_map = {}
if "all_documents" not in st.session_state:
    st.session_state.all_documents = []
if "vector_store_initialized" not in st.session_state:
    st.session_state.vector_store_initialized = False
if "processing_status" not in st.session_state:
    st.session_state.processing_status = ""

# Initialize modules
@st.cache_resource
def init_modules():
    processor = CVProcessor(single_chunk=True)
    vector_store = CVVectorStore()
    rag_engine = EnhancedRAGEngine(vector_store, max_candidates_per_query=15)
    job_matcher = EnhancedJobMatcher(vector_store, rag_engine)
    summarizer = CVSummarizer()
    job_recommender = JobRecommender()
    hr_question_generator = HRQuestionGenerator()
    
    return processor, vector_store, rag_engine, job_matcher, summarizer, job_recommender, hr_question_generator

def normalize_candidate_name(name: str) -> str:
    """Normalize candidate names to avoid duplicates - from main.py"""
    if not name:
        return "Unknown Candidate"
    
    # Remove common suffixes and normalize
    name = re.sub(r'(_CV|_cv|Cv|_Resume|Resume)$', '', name, flags=re.IGNORECASE)
    name = name.replace("_", " ").replace("-", " ")
    
    # Clean up extra spaces and title case
    name = " ".join(name.split()).strip().title()
    
    return name

def extract_skill_from_query(query: str) -> str:
    """Extract the main skill or requirement from a query - from main.py"""
    query_clean = query.lower().replace('?', '').strip()
    
    patterns = [
        r'who has (.+?)(?:\?|$)',
        r'who knows (.+?)(?:\?|$)',
        r'candidates with (.+?)(?:\?|$)',
        r'experience in (.+?)(?:\?|$)',
        r'experienced in (.+?)(?:\?|$)',
        r'skills in (.+?)(?:\?|$)',
        r'knowledge of (.+?)(?:\?|$)',
        r'proficient in (.+?)(?:\?|$)',
        r'expertise in (.+?)(?:\?|$)',
        r'background in (.+?)(?:\?|$)',
        r'qualified in (.+?)(?:\?|$)',
        r'certified in (.+?)(?:\?|$)',
        r'degree in (.+?)(?:\?|$)',
        r'studied (.+?)(?:\?|$)',
        r'worked with (.+?)(?:\?|$)',
        r'familiar with (.+?)(?:\?|$)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query_clean)
        if match:
            skill = match.group(1).strip()
            skill = skill.replace(' experience', '').replace(' skills', '')
            return skill
    
    stop_words = ['who', 'what', 'which', 'where', 'when', 'how', 'has', 'have', 'is', 'are', 
                  'the', 'a', 'an', 'candidates', 'candidate', 'people', 'person']
    
    words = query_clean.split()
    key_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    if key_words:
        return ' '.join(key_words)
    
    return "relevant experience"

# Streamlit App Configuration
st.set_page_config(
    page_title="Smart Recruiter Assistant", 
    layout="wide",
    page_icon="🤖"
)

st.title("🤖 Smart Recruiter Assistant")
st.markdown("Upload CVs, analyze them, ask queries, match jobs, and generate HR questions.")

# Initialize modules
processor, vector_store, rag_engine, job_matcher, summarizer, job_recommender, hr_question_generator = init_modules()

# Sidebar for CV Upload
with st.sidebar:
    st.header("📁 CV Upload & Processing")
    
    uploaded_files = st.file_uploader(
        "Upload CVs (PDF/DOCX)", 
        type=["pdf", "docx"], 
        accept_multiple_files=True,
        help="Upload multiple CV files to get started"
    )
    
    if st.button("🔄 Process CVs", type="primary"):
        if uploaded_files:
            with st.spinner("Processing CVs..."):
                try:
                    # Clear previous state
                    st.session_state.candidate_cv_map = {}
                    st.session_state.all_documents = []
                    
                    # Create upload directory
                    upload_dir = "uploaded_files"
                    os.makedirs(upload_dir, exist_ok=True)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, file in enumerate(uploaded_files):
                        status_text.text(f"Processing {file.name}...")
                        
                        # Save uploaded file
                        file_path = os.path.join(upload_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        # Process CV
                        try:
                            chunks = processor.process_cv(file_path)
                            candidate_name = Path(file.name).stem
                            normalized_name = normalize_candidate_name(candidate_name)
                            
                            if chunks:
                                # Store CV content
                                st.session_state.candidate_cv_map[normalized_name] = chunks[0].page_content
                                
                                # Update metadata
                                for chunk in chunks:
                                    chunk.metadata["candidate_name"] = normalized_name
                                    chunk.metadata["source_file"] = file.name
                                
                                st.session_state.all_documents.extend(chunks)
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Add to vector store
                    if st.session_state.all_documents:
                        status_text.text("Adding to vector store...")
                        vector_store.add_cvs(st.session_state.all_documents)
                        st.session_state.vector_store_initialized = True
                        
                        st.session_state.processing_status = f"✅ Successfully processed {len(st.session_state.candidate_cv_map)} CVs"
                        st.success(st.session_state.processing_status)
                    else:
                        st.error("No valid CV content found")
                        
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
        else:
            st.warning("Please upload CV files first")
    
    # Display current status
    if st.session_state.processing_status:
        st.info(st.session_state.processing_status)
    
    if st.session_state.candidate_cv_map:
        st.success(f"📊 {len(st.session_state.candidate_cv_map)} candidates loaded")
        
        with st.expander("View Candidate Names"):
            for i, name in enumerate(st.session_state.candidate_cv_map.keys(), 1):
                st.write(f"{i}. {name}")

# Main Tabs
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", 
    "❓ Ask", 
    "🎯 Match", 
    "📝 Summarize", 
    "💼 Recommend", 
    "🤖 HR Questions", 
    "🗃 Debug"
])

# TAB 0: OVERVIEW
with tab0:
    st.header("📊 Overview Dashboard")
    
    if not st.session_state.candidate_cv_map:
        st.info("👆 Upload and process CVs to view the overview dashboard")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Candidates", len(st.session_state.candidate_cv_map))
        
        with col2:
            st.metric("Documents Processed", len(st.session_state.all_documents))
        
        with col3:
            if st.session_state.vector_store_initialized:
                st.metric("Vector Store", "✅ Ready")
            else:
                st.metric("Vector Store", "❌ Not Ready")
        
        # Quick job recommendation overview
        st.subheader("🎯 Quick Job Match Overview")
        
        if st.button("Generate Quick Job Overview"):
            with st.spinner("Analyzing job matches..."):
                try:
                    # Get job recommendations for overview
                    quick_recommendations = job_recommender.get_best_jobs_for_candidates(
                        st.session_state.candidate_cv_map, top_k=1
                    )
                    
                    # Count jobs
                    job_counts = Counter()
                    for candidate, jobs in quick_recommendations.items():
                        if jobs:
                            top_job = jobs[0][0]  # Get job title from first recommendation
                            job_counts[top_job] += 1
                    
                    if job_counts:
                        # Create DataFrame for visualization
                        df = pd.DataFrame(list(job_counts.items()), columns=["Job Title", "Count"])
                        
                        # Create bar chart
                        fig = px.bar(df, x="Job Title", y="Count", 
                                   title="Distribution of Top Job Matches",
                                   color="Count")
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show top matches
                        st.subheader("🏆 Top Job Categories")
                        for job, count in job_counts.most_common():
                            st.write(f"**{job}**: {count} candidates")
                    else:
                        st.warning("No job matches found")
                        
                except Exception as e:
                    st.error(f"Error generating overview: {str(e)}")

# TAB 1: ASK (Custom Query)
with tab1:
    st.header("❓ Ask Questions")
    
    if not st.session_state.vector_store_initialized:
        st.warning("⚠️ Please upload and process CVs first")
    else:
        st.markdown("Ask questions about candidates, skills, experience, or any specific requirements.")
        
        # Query input
        query = st.text_input(
            "Enter your query:", 
            placeholder="e.g., Who has experience in Python and machine learning?",
            help="Ask about specific skills, experience, education, or any requirements"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            top_k = st.selectbox("Top candidates to show:", [3, 5, 7, 10], index=1)
        
        if st.button("🔍 Search", type="primary") and query.strip():
            with st.spinner(f"Searching for top {top_k} candidates..."):
                try:
                    # Extract skill from query (from main.py logic)
                    skill_or_requirement = extract_skill_from_query(query)
                    
                    # Get top candidates - returns both text and names
                    top_candidates_text, top_candidate_names = rag_engine.find_top_candidates(query, top_k=top_k)
                    
                    # Get all relevant candidates for the extracted skill
                    all_relevant = rag_engine.get_all_candidates_for_skill(skill_or_requirement)
                    
                    # Display results
                    st.subheader(f"🎯 Top {top_k} Candidates")
                    if top_candidate_names:
                        # Show candidate names as tags
                        st.write("**Found candidates:**")
                        cols = st.columns(min(len(top_candidate_names), 5))
                        for i, name in enumerate(top_candidate_names):
                            with cols[i % 5]:
                                st.success(name)
                    
                    # Show detailed ranking
                    with st.expander("📊 Detailed Ranking", expanded=True):
                        st.code(top_candidates_text, language="markdown")
                    
                    st.subheader("📋 All Relevant Candidates")
                    with st.expander("View All Relevant Candidates"):
                        st.text(all_relevant)
                    
                    # Store results in session state for HR questions
                    st.session_state.last_query_candidates = top_candidate_names
                    st.session_state.last_query = query
                    
                except Exception as e:
                    st.error(f"❌ Search failed: {str(e)}")
                    logger.error(f"Query processing failed: {e}")

# TAB 2: MATCH (Job Matching)
with tab2:
    st.header("🎯 Job Matching")
    
    if not st.session_state.vector_store_initialized:
        st.warning("⚠️ Please upload and process CVs first")
    else:
        st.markdown("Paste a job description to find the best matching candidates.")
        
        # Job details input
        col1, col2 = st.columns([1, 2])
        with col1:
            job_title = st.text_input("Job Title:", placeholder="e.g., Full Stack Developer")
        with col2:
            top_k = st.selectbox("Number of candidates:", [3, 5, 7, 10], index=1)
        
        job_description = st.text_area(
            "Job Description:",
            height=200,
            placeholder="""Full Stack Developer needed for web application development.
Required: JavaScript, React, Node.js, HTML, CSS, MongoDB
Preferred: TypeScript, Angular, Vue.js, Docker, AWS
3+ years of full-stack development experience required.""",
            help="Paste the complete job description including requirements and responsibilities"
        )
        
        if st.button("🎯 Match Candidates", type="primary") and job_description.strip():
            with st.spinner("Finding best matching candidates..."):
                try:
                    # Enhanced job description (from main.py logic)
                    enhanced_job_description = f"""
Job Title: {job_title}
Key Responsibilities and Requirements:
{job_description.strip()}

Skills Assessment:
Analyze candidates for both technical skills and experience level.
Consider cultural fit and communication abilities.
Evaluate portfolio and project experience.
"""
                    
                    # Perform job matching
                    job_match_result = job_matcher.match_job_to_cvs(
                        enhanced_job_description, 
                        top_k=top_k, 
                        explain=True
                    )
                    
                    # Format results
                    formatted_results = job_matcher.format_results(job_match_result, show_snippets=False)
                    
                    # Display results
                    st.subheader(f"🏆 Top {top_k} Matching Candidates")
                    
                    if job_title:
                        st.markdown(f"**Job:** {job_title}")
                    
                    # Show job description in expander
                    with st.expander("📋 Job Description"):
                        st.text(job_description)
                    
                    # Display matching results
                    st.markdown("### 🎯 Matching Results")
                    
                    # Parse and display results in a more structured way
                    if formatted_results:
                        lines = formatted_results.split('\n')
                        current_candidate = ""
                        
                        for line in lines:
                            line = line.strip()
                            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                                # New candidate
                                if '-' in line:
                                    parts = line.split('-', 1)
                                    rank_name = parts[0].strip()
                                    score_info = parts[1].strip() if len(parts) > 1 else ""
                                    
                                    st.markdown(f"**{rank_name}** - {score_info}")
                                    current_candidate = rank_name
                                else:
                                    st.markdown(f"**{line}**")
                            elif line.startswith('Score:'):
                                st.markdown(f"*{line}*")
                            elif line.startswith('Reason:') or line.startswith('Match:'):
                                st.markdown(f"📝 {line}")
                            elif line and not line.startswith('='):
                                st.markdown(line)
                    else:
                        st.warning("No matching results found")
                    
                except Exception as e:
                    st.error(f"❌ Job matching failed: {str(e)}")
                    logger.error(f"Job matching failed: {e}")

# TAB 3: SUMMARIZE
with tab3:
    st.header("📝 CV Summaries")
    
    if not st.session_state.candidate_cv_map:
        st.warning("⚠️ Please upload and process CVs first")
    else:
        st.markdown(f"Generate AI-powered summaries for all {len(st.session_state.candidate_cv_map)} candidates.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            summarize_all = st.button("📄 Summarize All CVs", type="primary")
        with col2:
            selected_candidate = st.selectbox(
                "Or select specific candidate:",
                [""] + list(st.session_state.candidate_cv_map.keys())
            )
        
        # Summarize specific candidate
        if selected_candidate and st.button("📝 Summarize Selected"):
            with st.spinner(f"Generating summary for {selected_candidate}..."):
                try:
                    cv_content = st.session_state.candidate_cv_map[selected_candidate]
                    summary = summarizer.summarize_cv(cv_content, selected_candidate)
                    
                    st.subheader(f"📄 Summary: {selected_candidate}")
                    st.success(summary)
                    
                except Exception as e:
                    st.error(f"❌ Summary generation failed for {selected_candidate}: {str(e)}")
        
        # Summarize all candidates
        if summarize_all:
            with st.spinner("Generating summaries for all candidates..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                summaries = {}
                total_candidates = len(st.session_state.candidate_cv_map)
                
                for i, (candidate_name, cv_content) in enumerate(st.session_state.candidate_cv_map.items()):
                    status_text.text(f"Summarizing {candidate_name}...")
                    
                    try:
                        summary = summarizer.summarize_cv(cv_content, candidate_name)
                        summaries[candidate_name] = summary
                    except Exception as e:
                        summaries[candidate_name] = f"⚠️ Error generating summary: {str(e)}"
                        logger.error(f"Summary generation failed for {candidate_name}: {e}")
                    
                    progress_bar.progress((i + 1) / total_candidates)
                
                status_text.text("✅ All summaries generated!")
                
                # Display summaries
                st.subheader("📚 All CV Summaries")
                
                for i, (candidate_name, summary) in enumerate(summaries.items(), 1):
                    with st.expander(f"{i}. {candidate_name}", expanded=False):
                        if summary.startswith("⚠️"):
                            st.error(summary)
                        else:
                            st.success(summary)

# TAB 4: RECOMMEND (Job Recommendations)
with tab4:
    st.header("💼 AI Job Recommendations")
    
    if not st.session_state.candidate_cv_map:
        st.warning("⚠️ Please upload and process CVs first")
    else:
        st.markdown("Get AI-powered job recommendations for each candidate based on their skills and experience.")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            top_k = st.selectbox("Jobs per candidate:", [1, 2, 3], index=1)
        with col2:
            show_explanations = st.checkbox("Show detailed explanations", value=True)
        
        if st.button("🚀 Generate Job Recommendations", type="primary"):
            with st.spinner("Analyzing candidates and generating job recommendations..."):
                try:
                    # Get job recommendations using the correct method from main.py
                    job_recommendations = job_recommender.get_best_jobs_for_candidates(
                        st.session_state.candidate_cv_map, 
                        top_k=top_k
                    )
                    
                    st.subheader("🎯 Job Recommendations")
                    st.markdown(f"**Total Candidates:** {len(job_recommendations)}")
                    
                    # Display recommendations for each candidate
                    for i, (candidate_name, jobs) in enumerate(job_recommendations.items(), 1):
                        with st.expander(f"{i}. {candidate_name}", expanded=True):
                            
                            if jobs:
                                st.markdown("**Recommended Positions:**")
                                
                                for j, (job_title, score, explanation) in enumerate(jobs, 1):
                                    percentage = int(score * 100)
                                    
                                    # Create columns for job info
                                    job_col1, job_col2 = st.columns([3, 1])
                                    
                                    with job_col1:
                                        st.markdown(f"**{j}. {job_title}**")
                                    with job_col2:
                                        # Create colored score badge
                                        if percentage >= 80:
                                            st.success(f"{percentage}%")
                                        elif percentage >= 60:
                                            st.warning(f"{percentage}%")
                                        else:
                                            st.info(f"{percentage}%")
                                    
                                    if show_explanations:
                                        st.markdown(f"💡 **Why this fits:** {explanation}")
                                    
                                    st.markdown("---")
                            else:
                                st.warning("No job recommendations available for this candidate")
                    
                    # Summary statistics
                    st.subheader("📊 Recommendation Summary")
                    
                    # Count job types
                    job_counts = Counter()
                    for candidate, jobs in job_recommendations.items():
                        for job_title, score, explanation in jobs:
                            job_counts[job_title] += 1
                    
                    if job_counts:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Most Recommended Jobs:**")
                            for job, count in job_counts.most_common(5):
                                st.write(f"• **{job}**: {count} candidates")
                        
                        with col2:
                            # Create a simple chart
                            df = pd.DataFrame(list(job_counts.items()), columns=["Job", "Count"])
                            st.bar_chart(df.set_index("Job"))
                    
                except Exception as e:
                    st.error(f"❌ Job recommendation failed: {str(e)}")
                    logger.error(f"Job recommendation failed: {e}")

# TAB 5: HR QUESTIONS
with tab5:
    st.header("🤖 HR Interview Questions")
    
    if not st.session_state.candidate_cv_map:
        st.warning("⚠️ Please upload and process CVs first")
    else:
        st.markdown("Generate tailored HR interview questions for candidates.")
        
        # Option 1: Generate for query results
        if hasattr(st.session_state, 'last_query_candidates') and st.session_state.last_query_candidates:
            st.subheader("🎯 Generate for Last Query Results")
            st.info(f"Last query: '{st.session_state.last_query}'")
            st.write(f"**Top candidates found:** {', '.join(st.session_state.last_query_candidates)}")
            
            if st.button("🤖 Generate HR Questions for Query Results", type="primary"):
                with st.spinner("Generating HR questions for top candidates..."):
                    try:
                        # Match candidates from query with CV map (from main.py logic)
                        matched_candidates = []
                        for candidate in st.session_state.last_query_candidates:
                            if candidate in st.session_state.candidate_cv_map:
                                matched_candidates.append(candidate)
                            else:
                                # Try case-insensitive match
                                for cv_candidate in st.session_state.candidate_cv_map.keys():
                                    if candidate.lower() == cv_candidate.lower():
                                        matched_candidates.append(cv_candidate)
                                        break
                        
                        if matched_candidates:
                            # Generate questions using the correct method
                            hr_questions = hr_question_generator.generate_questions_for_top_candidates(
                                st.session_state.candidate_cv_map, 
                                matched_candidates
                            )
                            
                            # Display questions
                            st.success(f"✅ Generated questions for {len(matched_candidates)} candidates")
                            
                            for candidate in matched_candidates:
                                if candidate in hr_questions:
                                    with st.expander(f"🎤 {candidate}", expanded=True):
                                        
                                        for section, questions in hr_questions[candidate].items():
                                            if questions:
                                                st.markdown(f"**{section}**")
                                                for i, question in enumerate(questions, 1):
                                                    st.markdown(f"{i}. {question}")
                                                st.markdown("---")
                                else:
                                    st.warning(f"No questions generated for {candidate}")
                        else:
                            st.error("Could not match query candidates with CV database")
                    
                    except Exception as e:
                        st.error(f"❌ HR question generation failed: {str(e)}")
        
        st.markdown("---")
        
        # Option 2: Select specific candidates
        st.subheader("👥 Generate for Selected Candidates")
        
        selected_candidates = st.multiselect(
            "Select candidates for HR questions:",
            list(st.session_state.candidate_cv_map.keys()),
            max_selections=5,
            help="Select up to 5 candidates"
        )
        
        if selected_candidates and st.button("🤖 Generate HR Questions for Selected"):
            with st.spinner(f"Generating HR questions for {len(selected_candidates)} candidates..."):
                try:
                    # Generate questions for selected candidates
                    hr_questions = hr_question_generator.generate_questions_for_top_candidates(
                        st.session_state.candidate_cv_map, 
                        selected_candidates
                    )
                    
                    st.success(f"✅ Generated questions for {len(selected_candidates)} candidates")
                    
                    # Display questions
                    for candidate in selected_candidates:
                        if candidate in hr_questions:
                            with st.expander(f"🎤 {candidate}", expanded=True):
                                total_questions = sum(len(questions) for questions in hr_questions[candidate].values())
                                st.info(f"Total questions: {total_questions}")
                                
                                for section, questions in hr_questions[candidate].items():
                                    if questions:
                                        st.markdown(f"**{section}**")
                                        for i, question in enumerate(questions, 1):
                                            st.markdown(f"{i}. {question}")
                                        st.markdown("---")
                        else:
                            st.warning(f"No questions generated for {candidate}")
                
                except Exception as e:
                    st.error(f"❌ HR question generation failed: {str(e)}")

# TAB 6: DEBUG
with tab6:
    st.header("🗃 Debug Information")
    
    st.subheader("📊 Session State")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Status:**")
        st.write(f"• Candidates loaded: {len(st.session_state.candidate_cv_map)}")
        st.write(f"• Documents processed: {len(st.session_state.all_documents)}")
        st.write(f"• Vector store initialized: {st.session_state.vector_store_initialized}")
        
        if hasattr(st.session_state, 'last_query'):
            st.write(f"• Last query: {st.session_state.last_query}")
            st.write(f"• Query candidates: {len(getattr(st.session_state, 'last_query_candidates', []))}")
    
    with col2:
        st.markdown("**Memory Usage:**")
        st.write(f"• Session state keys: {len(st.session_state.keys())}")
        if st.session_state.candidate_cv_map:
            avg_cv_length = sum(len(cv) for cv in st.session_state.candidate_cv_map.values()) // len(st.session_state.candidate_cv_map)
            st.write(f"• Average CV length: {avg_cv_length} chars")
    
    # Candidate details
    if st.session_state.candidate_cv_map:
        st.subheader("👥 Candidate Details")
