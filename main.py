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
import time
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_skill_from_query(query: str) -> str:
    """Extract the main skill or requirement from a query"""
    # Remove question marks and convert to lowercase
    query_clean = query.lower().replace('?', '').strip()
    
    # Common patterns to identify skills/requirements
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
    
    # Try each pattern
    for pattern in patterns:
        match = re.search(pattern, query_clean)
        if match:
            skill = match.group(1).strip()
            # Clean up common words
            skill = skill.replace(' experience', '').replace(' skills', '')
            return skill
    
    # If no pattern matches, try to extract key terms
    # Remove common question words
    stop_words = ['who', 'what', 'which', 'where', 'when', 'how', 'has', 'have', 'is', 'are', 
                  'the', 'a', 'an', 'candidates', 'candidate', 'people', 'person']
    
    words = query_clean.split()
    key_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    if key_words:
        return ' '.join(key_words)
    
    # Default fallback
    return "relevant experience"

def save_to_txt(content: str, filename: str):
    """Save content to a text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"Saved results to {filename}")

def format_query_results(query: str, top_candidates: str, all_relevant: str) -> str:
    """Format query results in the requested style"""
    output = f"Query: {query}\n"
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

def process_custom_query(rag_engine: EnhancedRAGEngine, query: str):
    """Process any custom query and return formatted results with candidate names"""
    logger.info(f"Processing custom query: {query}")
    
    # Extract skill/requirement from query
    skill_or_requirement = extract_skill_from_query(query)
    logger.info(f"Extracted skill/requirement: {skill_or_requirement}")
    
    # Get top candidates - now returns both text and names
    top_candidates_text, top_candidate_names = rag_engine.find_top_candidates(query, top_k=5)
    
    # Get all relevant candidates for the extracted skill
    all_relevant = rag_engine.get_all_candidates_for_skill(skill_or_requirement)
    
    return top_candidates_text, all_relevant, top_candidate_names

def main():
    print("🚀 Starting Smart Recruiter Assistant")
    
    # Initialize components
    processor = CVProcessor(single_chunk=True)
    vector_store = CVVectorStore()
    rag_engine = EnhancedRAGEngine(vector_store, max_candidates_per_query=15)
    job_matcher = EnhancedJobMatcher(vector_store, rag_engine)
    summarizer = CVSummarizer()
    job_recommender = JobRecommender()
    hr_question_generator = HRQuestionGenerator()
    
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
    
    # 2. AI Job recommendations
    logger.info("Starting AI job recommendations...")
    try:
        job_recommendations = job_recommender.get_top_candidates_for_jobs(candidate_cv_map, top_k=5)
        job_recommender.save_recommendations_to_file(
            job_recommendations,
            os.path.join(output_dir, "ai_job_recommendations.txt")
        )
    except Exception as e:
        logger.error(f"Job recommendation failed: {e}")
        with open(os.path.join(output_dir, "ai_job_recommendations.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Job recommendation failed: {str(e)}")
    
    # 3. Single custom query
    print("\n" + "="*60)
    print("CUSTOM QUERY SEARCH")
    print("="*60)
    
    query = input("\nEnter your query: ").strip()
    
    top_query_candidates = []  # Store top candidates from query
    
    if query:
        logger.info(f"Processing custom query: {query}")
        try:
            # Process query - now returns candidate names directly
            top_candidates_text, all_relevant, top_query_candidates = process_custom_query(rag_engine, query)
            
            query_results = format_query_results(query, top_candidates_text, all_relevant)
            save_to_txt(query_results, os.path.join(output_dir, "query_results.txt"))
            print(f"\n✅ Query processed successfully!")
            print(f"📄 Results saved to: {os.path.join(output_dir, 'query_results.txt')}")
            
            if top_query_candidates:
                print(f"\n📋 Top {len(top_query_candidates)} candidates identified: {', '.join(top_query_candidates)}")
            
            # Show preview of top candidates
                        # Show preview of top candidates
            print("\n--- Preview of Top Candidates ---")
            preview_lines = top_candidates_text.split('\n')[:10]
            for line in preview_lines:
                if line.strip():
                    print(line)
            if len(top_candidates_text.split('\n')) > 10:
                print("...")
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            error_msg = f"Query processing failed: {str(e)}"
            with open(os.path.join(output_dir, "query_results.txt"), 'w', encoding='utf-8') as f:
                f.write(error_msg)
            print(f"\n❌ {error_msg}")
    else:
        print("\n⚠️ No query entered. Skipping query processing.")
    
    # 4. Generate HR Questions for Top 5 Query Results
    if top_query_candidates:
        print("\n" + "="*60)
        print("HR QUESTION GENERATION")
        print("="*60)
        print(f"\n🎯 Generating HR interview questions for top {len(top_query_candidates)} candidates from query results...")
        
        try:
            # Debug: Check if candidates exist in CV map
            logger.info(f"Candidates to generate questions for: {top_query_candidates}")
            logger.info(f"Available candidates in CV map: {list(candidate_cv_map.keys())[:10]}...")
            
            # Ensure candidate names match exactly
            matched_candidates = []
            for candidate in top_query_candidates:
                # Try exact match first
                if candidate in candidate_cv_map:
                    matched_candidates.append(candidate)
                else:
                    # Try case-insensitive match
                    for cv_candidate in candidate_cv_map.keys():
                        if candidate.lower() == cv_candidate.lower():
                            matched_candidates.append(cv_candidate)
                            break
            
            if matched_candidates:
                logger.info(f"Matched candidates for HR questions: {matched_candidates}")
                
                # Generate questions for the matched candidates
                hr_questions = hr_question_generator.generate_questions_for_top_candidates(
                    candidate_cv_map, 
                    matched_candidates
                )
                
                # Save HR questions to file
                hr_questions_file = os.path.join(output_dir, "hr_interview_questions.txt")
                hr_question_generator.save_questions_to_file(hr_questions, hr_questions_file)
                
                print(f"\n✅ HR questions generated successfully!")
                print(f"📄 Questions saved to: {hr_questions_file}")
                
                # Show summary of generated questions
                print("\n--- HR Questions Summary ---")
                for candidate in matched_candidates[:5]:
                    if candidate in hr_questions:
                        total_questions = sum(len(questions) for questions in hr_questions[candidate].values())
                        print(f"• {candidate}: {total_questions} questions generated")
            else:
                print("\n⚠️ Could not match query candidates with CV database.")
                logger.warning(f"No matches found. Query candidates: {top_query_candidates}")
                logger.warning(f"CV candidates sample: {list(candidate_cv_map.keys())[:5]}")
            
        except Exception as e:
            logger.error(f"HR question generation failed: {e}")
            print(f"\n❌ HR question generation failed: {str(e)}")
    else:
        print("\n⚠️ No candidates identified from query. Skipping HR question generation.")
    
    # 5. Job description matching
    logger.info("Processing job matching...")
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
    
    try:
        # Perform job matching with enhanced description
        job_match_result = job_matcher.match_job_to_cvs(enhanced_job_description, top_k=5, explain=True)
        formatted_job_results = job_matcher.format_results(job_match_result, show_snippets=False)
        
        # Format and save job matching results
        job_match_results = format_job_match_results(job_title, job_description, formatted_job_results)
        save_to_txt(job_match_results, os.path.join(output_dir, "job_match_results.txt"))
    except Exception as e:
        logger.error(f"Job matching failed: {e}")
        with open(os.path.join(output_dir, "job_match_results.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Job matching failed: {str(e)}")
    
    # 6. Final Summary
    print("\n" + "="*60)
    print("✅ PROCESSING COMPLETE")
    print("="*60)
    print(f"\n📁 All results saved to: {os.path.abspath(output_dir)}")
    print("\nGenerated Files:")
    print("1. all_candidate_summaries.txt - Summaries for all candidates")
    print("2. ai_job_recommendations.txt - AI-powered job recommendations") 
    print("3. query_results.txt - Results from your custom query")
    if top_query_candidates:
        print("4. hr_interview_questions.txt - HR questions for top 5 candidates from query")
    print(f"{5 if top_query_candidates else 4}. job_match_results.txt - Job matching results")
    
    print("\n📊 Summary:")
    print(f"• Total candidates processed: {len(all_candidates)}")
    if top_query_candidates:
        print(f"• Top candidates from query: {len(top_query_candidates)}")
        if 'matched_candidates' in locals() and matched_candidates:
            print(f"• HR questions generated for: {len(matched_candidates)} candidates")

if __name__ == "__main__":
    main()
