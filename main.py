# main.py - Enhanced with better query processing and diverse candidate selection
from Preprocessing.document_processor import CVProcessor
from Preprocessing.vector_store import CVVectorStore
from RAG.rag_engine import EnhancedRAGEngine
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_all_candidates_from_directory(cv_directory):
    """Get all candidate names from CV files in directory"""
    candidate_names = []
    if os.path.exists(cv_directory):
        for filename in os.listdir(cv_directory):
            if filename.lower().endswith(('.pdf', '.docx', '.doc', '.txt')):
                candidate_name = Path(filename).stem.replace('_CV', '').replace('_cv', '').replace('_', ' ').replace('-', ' ').title()
                candidate_names.append(candidate_name)
    return candidate_names

def extract_skill_from_query(query: str) -> str:
    """Extract skill/topic from query like 'Who has experience in ... ?'"""
    prefix = "Who has experience in "
    if query.startswith(prefix):
        return query[len(prefix):].rstrip("?").strip()
    return query

def main():
    print("🚀 Smart Recruiter Assistant - Enhanced Version")
   
    # Initialize components
    processor = CVProcessor(single_chunk=True)
    vector_store = CVVectorStore()
   
    # Get CV files
    cv_directory = "D:/0_ITI_9_month/A15 NLP/Smart Recruiter Assistant/data/cvs"
    if not os.path.exists(cv_directory):
        return
   
    cv_files = [
        os.path.join(cv_directory, f)
        for f in os.listdir(cv_directory)
        if f.lower().endswith(('.pdf', '.docx', '.doc', '.txt'))
    ]
    if not cv_files:
        return
   
    # Process all CVs
    all_documents = []
    candidate_names = []
   
    for cv_file in cv_files:
        try:
            chunks = processor.process_cv(cv_file)
            candidate_name = Path(cv_file).stem.replace('_CV', '').replace('_cv', '').replace('_', ' ').replace('-', ' ').title()
            candidate_names.append(candidate_name)
            for chunk in chunks:
                chunk.metadata.update({
                    "candidate_name": candidate_name,
                    "file_path": cv_file,
                    "file_type": Path(cv_file).suffix,
                    "document_type": "complete_cv"
                })
            all_documents.extend(chunks)
        except Exception:
            continue
   
    if not all_documents:
        return
   
    # Add documents to vector store
    vector_store.add_cvs(all_documents)
   
    # Initialize Enhanced RAG engine
    rag_engine = EnhancedRAGEngine(vector_store, max_candidates_per_query=15)
   
    # SINGLE QUERY
    query = "Who has work experience?"
    top_candidates = rag_engine.find_top_candidates(query, top_k=5)

    # Extract skill automatically from query
    skill = extract_skill_from_query(query)
    all_relevant = rag_engine.get_all_candidates_for_skill(skill)
    
    # Save results
    filename = "query_results.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Query: {query}\n")
        f.write("="*60 + "\n\n")
        f.write("TOP 5 CANDIDATES RANKING:\n")
        f.write("-" * 40 + "\n")
        f.write(top_candidates)
        f.write("\n\n" + "="*40 + "\n")
        f.write("ALL RELEVANT CANDIDATES:\n")
        f.write("-" * 40 + "\n")
        f.write(all_relevant)
    
    print(f"\n💾 Results saved to '{filename}'")

if __name__ == "__main__":
    main()
