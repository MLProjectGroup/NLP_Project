# main.py
from Preprocessing.document_processor import CVProcessor
from Preprocessing.vector_store import CVVectorStore
from RAG.rag_engine import EnhancedRAGEngine
from RAG.job_matcher import EnhancedJobMatcher
from RAG.cv_summarizer import CVSummarizer
from RAG.job_recommender import JobRecommender
from RAG.hr_question_generator import HRQuestionGenerator
import os
from pathlib import Path
import logging
import re

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directories
CV_DIR = "data/cvs"
OUTPUT_DIR = "results"
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summaries")

# Helpers
def save_to_txt(content: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Saved to {path}")

def normalize_name(name: str) -> str:
    name = re.sub(r'(_CV|_cv|Cv|_Resume|Resume)$', '', name, flags=re.IGNORECASE)
    return " ".join(name.replace('_', ' ').replace('-', ' ').split()).title()

# Main
if __name__ == "__main__":
    print("\n🚀 Starting Smart Recruiter Assistant")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    processor = CVProcessor(single_chunk=True)
    vector_store = CVVectorStore()
    rag = EnhancedRAGEngine(vector_store, max_candidates_per_query=15)
    matcher = EnhancedJobMatcher(vector_store, rag)
    summarizer = CVSummarizer()
    recommender = JobRecommender()
    hrgen = HRQuestionGenerator()

    if not os.path.exists(CV_DIR):
        logger.error(f"CV directory not found: {CV_DIR}")
        exit(1)

    cv_files = [f for f in Path(CV_DIR).glob("*.*") if f.suffix.lower() in ['.pdf', '.docx', '.doc', '.txt']]
    if not cv_files:
        logger.error("No CV files found.")
        exit(1)

    candidate_cv = {}
    all_docs = []
    all_names = []

    for file in cv_files:
        try:
            chunks = processor.process_cv(str(file))
            name = normalize_name(file.stem)
            if chunks:
                candidate_cv[name] = chunks[0].page_content
                all_names.append(name)
                for c in chunks:
                    c.metadata.update({"candidate_name": name, "source_file": str(file)})
                all_docs.extend(chunks)
        except Exception as e:
            logger.error(f"Failed processing {file}: {e}")

    vector_store.add_cvs(all_docs)

    # Generate Summaries
    master_summary = f"CV SUMMARIES FOR ALL CANDIDATES\n{'='*60}\n\nTotal: {len(all_names)}\n\n"
    for i, name in enumerate(all_names, 1):
        content = candidate_cv[name]
        try:
            summary = summarizer.summarize_cv(content, name)
            summary_file = os.path.join(SUMMARY_DIR, f"summary_{re.sub(r'[^\w]', '_', name)}.txt")
            save_to_txt(summary, summary_file)
            master_summary += f"{i}. {name}\n{'-'*40}\n{summary}\n\n"
        except Exception as e:
            msg = f"⚠️ Error summarizing {name}: {e}"
            logger.error(msg)
            master_summary += f"{i}. {name}\n{'-'*40}\n{msg}\n\n"
    save_to_txt(master_summary, os.path.join(OUTPUT_DIR, "all_candidate_summaries.txt"))

    # AI Recommendations
    try:
        recs = recommender.get_top_candidates_for_jobs(candidate_cv, top_k=5)
        recommender.save_recommendations_to_file(recs, os.path.join(OUTPUT_DIR, "ai_job_recommendations.txt"))
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")

    # Custom Query
    print("\nCUSTOM QUERY")
    query = input("Enter your query: ").strip()
    top_query_candidates = []

    if query:
        try:
            skill = rag.extract_skill_from_query(query)
            top_txt, all_rel, top_names = rag.find_top_candidates(query, top_k=5)
            top_query_candidates = top_names
            q_file = os.path.join(OUTPUT_DIR, "query_results.txt")
            result = f"Query: {query}\n{'='*60}\n\nTOP 5:\n{'-'*40}\n{top_txt}\n\nALL MATCHES:\n{'-'*40}\n{all_rel}"
            save_to_txt(result, q_file)
        except Exception as e:
            logger.error(f"Query error: {e}")

    # HR Questions
    if top_query_candidates:
        matched = [n for n in top_query_candidates if n in candidate_cv]
        try:
            questions = hrgen.generate_questions_for_top_candidates(candidate_cv, matched)
            hr_file = os.path.join(OUTPUT_DIR, "hr_interview_questions.txt")
            hrgen.save_questions_to_file(questions, hr_file)
        except Exception as e:
            logger.error(f"HR Q error: {e}")

    # Job Matching
    try:
        job_title = "Full Stack Developer"
        job_desc = """
        Full Stack Developer needed. Must know JavaScript, React, Node.js, HTML, CSS, MongoDB.
        Preferred: TypeScript, Angular, Vue.js, Docker, AWS. 3+ years experience.
        """
        enhanced = """
        Job Title: Full Stack Developer
        Required Skills: JavaScript (10), React (10), Node.js (10), HTML (8), CSS (8), MongoDB (8)
        Preferred Skills: TypeScript (7), Angular (7), Vue.js (7), Docker (6), AWS (6)
        Experience: 3+ years full-stack development
        """
        match = matcher.match_job_to_cvs(enhanced, top_k=5, explain=True)
        match_text = matcher.format_results(match, show_snippets=False)
        result_text = f"Job Title: {job_title}\n{'='*60}\n\nJob Description:\n{job_desc.strip()}\n\n{'='*60}\n\nMATCHES:\n\n{match_text}"
        save_to_txt(result_text, os.path.join(OUTPUT_DIR, "job_match_results.txt"))
    except Exception as e:
        logger.error(f"Job matching failed: {e}")

    print("\n✅ All done. Check the 'results' folder.")
