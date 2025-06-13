# main.py
from Preprocessing.document_processor import CVProcessor
from Preprocessing.vector_store import CVVectorStore
from RAG.rag_engine import EnhancedRAGEngine
from RAG.job_matcher import EnhancedJobMatcher
from RAG.cv_summarizer import CVSummarizer
import os
from pathlib import Path
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_to_txt(content: str, filename: str):
    """Save content to a text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"Saved results to {filename}")

def format_query_results(top_candidates: str, all_relevant: str) -> str:
    """Format query results in the requested style"""
    output = "Query: Who has work experience?\n"
    output += "=" * 60 + "\n\n"
    output += "TOP 5 CANDIDATES RANKING:\n"
    output += "-" * 40 + "\n"
    output += top_candidates + "\n\n"
    output += "=" * 40 + "\n"
    output += "ALL RELEVANT CANDIDATES:\n"
    output += "-" * 40 + "\n"
    output += all_relevant
    return output

def format_job_match_results(job_title: str, job_description: str, formatted_results: str) -> str:
    """Format job matching results in the requested style"""
    output = f"Job Title: {job_title}\n"
    output += "=" * 60 + "\n\n"
    output += "Job Description:\n\n"
    output += job_description.strip() + "\n\n"
    output += "=" * 60 + "\n\n"
    output += "MATCHING RESULTS:\n\n"
    output += formatted_results
    return output

def normalize_candidate_name(name: str) -> str:
    """Normalize candidate names to avoid duplicates"""
    if not name:
        return "Unknown Candidate"
    
    # Remove common suffixes and normalize
    name = re.sub(r'(_CV|_cv|Cv|_Resume|Resume)$', '', name, flags=re.IGNORECASE)
    name = name.replace("_", " ").replace("-", " ")
    
    # Clean up extra spaces and title case
    name = " ".join(name.split()).strip().title()
    
    return name

def main():
    print("🚀 Starting Smart Recruiter Assistant")
    
    # Initialize components
    processor = CVProcessor(single_chunk=True)
    vector_store = CVVectorStore()
    rag_engine = EnhancedRAGEngine(vector_store, max_candidates_per_query=15)
    job_matcher = EnhancedJobMatcher(vector_store, rag_engine)
    summarizer = CVSummarizer()
    
    # Setup output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get CV files
    cv_directory = "data/cvs"
    if not os.path.exists(cv_directory):
        logger.error(f"CV directory not found: {cv_directory}")
        return
    
    cv_files = [
        os.path.join(cv_directory, f)
        for f in os.listdir(cv_directory)
        if f.lower().endswith(('.pdf', '.docx', '.doc', '.txt'))
    ]
    
    if not cv_files:
        logger.error("No CV files found in the directory")
        return
    
    # Process all CVs and store candidate content
    candidate_cv_map = {}
    all_documents = []
    all_candidates = []
    
    for cv_file in cv_files:
        try:
            chunks = processor.process_cv(cv_file)
            candidate_name = Path(cv_file).stem
            normalized_name = normalize_candidate_name(candidate_name)
            
            if chunks:
                # Store CV content for later summarization
                candidate_cv_map[normalized_name] = chunks[0].page_content
                all_candidates.append(normalized_name)
                
                # Update metadata with normalized name
                for chunk in chunks:
                    chunk.metadata["candidate_name"] = normalized_name
                    chunk.metadata["source_file"] = cv_file
                
                all_documents.extend(chunks)
        except Exception as e:
            logger.error(f"Error processing {cv_file}: {e}")
    
    # Add documents to vector store
    vector_store.add_cvs(all_documents)
    
    # 1. Generate summaries for ALL candidates
    summary_dir = os.path.join(output_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Create a master summary file that lists all summaries
    master_summary_content = "CV SUMMARIES FOR ALL CANDIDATES\n"
    master_summary_content += "=" * 60 + "\n\n"
    master_summary_content += f"Total Candidates: {len(all_candidates)}\n\n"
    
    for i, candidate_name in enumerate(all_candidates, 1):
        cv_content = candidate_cv_map.get(candidate_name, "")
        
        if cv_content:
            try:
                # Get formatted summary with line breaks
                cv_summary = summarizer.summarize_cv(cv_content, candidate_name)
                
                # Format summary content for individual file
                summary_content = f"CV Summary for {candidate_name}\n"
                summary_content += "=" * 60 + "\n\n"
                summary_content += cv_summary
                
                # Save each summary to separate file
                safe_name = re.sub(r'[^\w]', '_', candidate_name)
                summary_file = os.path.join(summary_dir, f"summary_{safe_name}.txt")
                save_to_txt(summary_content, summary_file)
                
                # Add to master summary
                master_summary_content += f"{i}. {candidate_name}\n"
                master_summary_content += "-" * 40 + "\n"
                master_summary_content += cv_summary + "\n\n"
                
            except Exception as e:
                error_msg = f"⚠️ Error generating summary for {candidate_name}: {str(e)}"
                logger.error(error_msg)
                master_summary_content += f"{i}. {candidate_name}\n"
                master_summary_content += "-" * 40 + "\n"
                master_summary_content += error_msg + "\n\n"
        else:
            error_msg = f"⚠️ No CV content found for {candidate_name}"
            logger.warning(error_msg)
            master_summary_content += f"{i}. {candidate_name}\n"
            master_summary_content += "-" * 40 + "\n"
            master_summary_content += error_msg + "\n\n"
    
    # Save master summary file
    master_summary_file = os.path.join(output_dir, "all_candidate_summaries.txt")
    save_to_txt(master_summary_content, master_summary_file)
    
    # 2. Query-based candidate search
    query = "Who has work experience?"
    top_candidates = rag_engine.find_top_candidates(query, top_k=5)
    all_relevant = rag_engine.get_all_candidates_for_skill("work experience")
    
    # Format and save query results
    query_results = format_query_results(top_candidates, all_relevant)
    save_to_txt(query_results, os.path.join(output_dir, "query_results.txt"))
    
    # 3. Job description matching
    job_title = "Full Stack Developer"
    job_description = """
        Full Stack Developer needed for web application development.
        Required: JavaScript, React, Node.js, HTML, CSS, MongoDB
        Preferred: TypeScript, Angular, Vue.js, Docker, AWS
        3+ years of full-stack development experience required.
    """
    
    # Enhanced job matching with skill weighting and detailed responsibilities
    enhanced_job_description = """
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
    """
    
    # Perform job matching with enhanced description
    job_match_result = job_matcher.match_job_to_cvs(enhanced_job_description, top_k=5, explain=True)
    formatted_job_results = job_matcher.format_results(job_match_result, show_snippets=False)
    
    # Format and save job matching results
    job_match_results = format_job_match_results(job_title, job_description, formatted_job_results)
    save_to_txt(job_match_results, os.path.join(output_dir, "job_match_results.txt"))
    
    print("✅ Processing complete. All results saved to 'results' directory")

if __name__ == "__main__":
    main()
