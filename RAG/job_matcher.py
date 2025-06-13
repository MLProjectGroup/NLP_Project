# job_matcher.py
"""Enhanced Job Matcher for Smart Recruiter Assistant"""

from collections import defaultdict
from typing import List, Dict, Tuple, Any
import re
import logging
from langchain.schema import Document
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class EnhancedJobMatcher:
    """Enhanced job matcher with better candidate diversity and analysis"""
    
    def __init__(self, vector_store, rag_engine=None):
        """
        Initialize JobMatcher with existing vector store and optional RAG engine
        
        Args:
            vector_store: CVVectorStore instance
            rag_engine: EnhancedRAGEngine instance (optional)
        """
        self.vector_store = vector_store
        self.rag_engine = rag_engine
        
        # Initialize Gemini for explanations
        self.llm = ChatGroq(
            model=os.getenv("Groq_model", "deepseek-r1-distill-llama-70b"),
            google_api_key=os.getenv("Groq_API_KEY"),
            temperature=0,
            convert_system_message_to_human=True
        )
        
        # Enhanced stopwords for better keyword extraction
        self.stopwords = {
            "and", "the", "are", "you", "for", "our", "with", "your", "but", "not", "has",
            "was", "this", "that", "they", "will", "who", "all", "can", "have", "from",
            "preferred", "experience", "skilled", "seeking", "engineer", "we", "job", "role",
            "to", "in", "of", "a", "an", "as", "at", "by", "on", "or", "is", "it", "be",
            "must", "should", "would", "could", "may", "might", "able", "work", "working",
            "years", "year", "minimum", "required", "requirements", "responsibilities"
        }
    
    def extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text with enhanced filtering"""
        text = text.lower()
        
        # Extract technical terms, skills, and important phrases
        # Look for multi-word technical terms first
        technical_patterns = [
            r'\b(?:machine learning|data science|artificial intelligence|deep learning)\b',
            r'\b(?:react js|node js|angular js|vue js)\b',
            r'\b(?:python|java|javascript|typescript|c\+\+|c#|php|ruby|go|rust)\b',
            r'\b(?:aws|azure|gcp|docker|kubernetes|terraform)\b',
            r'\b(?:sql|mysql|postgresql|mongodb|redis|elasticsearch)\b',
            r'\b(?:git|github|gitlab|jenkins|ci/cd)\b'
        ]
        
        keywords = set()
        
        # Extract multi-word technical terms
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            keywords.update(matches)
        
        # Extract single words (3+ characters)
        single_words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        keywords.update(kw for kw in single_words if kw not in self.stopwords)
        
        # Filter out common non-technical words
        filtered_keywords = set()
        for kw in keywords:
            if len(kw) >= 3 and not kw.isdigit():
                filtered_keywords.add(kw)
        
        return filtered_keywords
    
    def calculate_similarity_score(self, job_keywords: set, cv_keywords: set) -> float:
        """Calculate advanced similarity score between job and CV"""
        if not job_keywords:
            return 0.0
        
        # Basic keyword overlap
        common_keywords = job_keywords.intersection(cv_keywords)
        basic_score = len(common_keywords) / len(job_keywords)
        
        # Bonus for high-value technical keywords
        technical_keywords = {
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node', 
            'aws', 'azure', 'docker', 'kubernetes', 'machine learning', 'ai',
            'data science', 'sql', 'mongodb', 'git', 'jenkins', 'terraform'
        }
        
        technical_matches = common_keywords.intersection(technical_keywords)
        technical_bonus = len(technical_matches) * 0.1  # 10% bonus per technical match
        
        return min(basic_score + technical_bonus, 1.0)
    
    def explain_match_with_llm(self, job_description: str, candidate_snippet: str, candidate_name: str) -> str:
        """Use LLM to explain why a candidate matches the job description"""
        prompt = f"""You are an expert recruiter analyzing candidate CVs for job matches.

JOB DESCRIPTION:
{job_description}

CANDIDATE: {candidate_name}
CANDIDATE CV EXCERPT:
\"\"\"{candidate_snippet}\"\"\"

Provide a concise 2-3 sentence analysis covering:
1. Key matching qualifications/skills
2. Relevant experience alignment
3. Overall fit assessment

Focus on specific technical skills, years of experience, and role-relevant achievements.
"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM explanation error for {candidate_name}: {e}")
            return f"Analysis unavailable: {str(e)}"

    def normalize_candidate_name(self, name: str) -> str:
        """Normalize candidate names to avoid duplicates"""
        if not name:
            return "Unknown Candidate"
        
        # Remove common suffixes and normalize
        name = name.replace("_CV", "").replace("_cv", "").replace("Cv", "")
        name = name.replace("_", " ").replace("-", " ")
        
        # Clean up extra spaces and title case
        name = " ".join(name.split()).strip().title()
        
        return name

    def get_all_relevant_candidates_detailed(self, job_description: str) -> str:
        """Get detailed information about all relevant candidates using RAG engine"""
        if not self.rag_engine:
            return "RAG engine not available for detailed candidate analysis."
        
        try:
            # Use RAG engine to get all relevant candidates
            all_relevant = self.rag_engine.get_all_candidates_for_skill("work experience")
            return all_relevant
        except Exception as e:
            logger.error(f"Error getting all relevant candidates: {e}")
            return f"Error retrieving detailed candidate information: {str(e)}"
    
    def match_job_to_cvs(self, job_description: str, top_k: int = 5, explain: bool = True) -> Dict[str, Any]:
        """
        Match a job description to top K CVs with enhanced scoring and fixed duplicates
        
        Args:
            job_description: The job description text
            top_k: Number of top candidates to return (default 5)
            explain: Whether to generate LLM explanations
            
        Returns:
            Dictionary with matching results
        """
        logger.info(f"Matching job description to top {top_k} CVs")
        
        # Get 500 documents initially as requested
        initial_results = self.vector_store.similarity_search(job_description, k=500)
        
        # Extract keywords from job description
        jd_keywords = self.extract_keywords(job_description)
        logger.info(f"Extracted {len(jd_keywords)} keywords from job description: {list(jd_keywords)[:10]}")
        
        # Group results by normalized candidate name to eliminate duplicates
        candidate_matches = defaultdict(lambda: {
            'documents': [],
            'best_document': None,
            'matched_keywords': set(),
            'relevance_score': 0.0,
            'similarity_score': 0.0,
            'original_name': None
        })
        
        # Process each result
        for doc in initial_results:
            original_name = doc.metadata.get('candidate_name', 'Unknown Candidate')
            normalized_name = self.normalize_candidate_name(original_name)
            
            # Calculate keyword matches
            doc_keywords = self.extract_keywords(doc.page_content)
            matched_keywords = jd_keywords.intersection(doc_keywords)
            similarity_score = self.calculate_similarity_score(jd_keywords, doc_keywords)
            
            # Only include candidates with some relevance
            if similarity_score > 0.05:  # Minimum 5% relevance
                candidate_matches[normalized_name]['documents'].append(doc)
                candidate_matches[normalized_name]['matched_keywords'].update(matched_keywords)
                candidate_matches[normalized_name]['original_name'] = original_name
                
                # Keep the most relevant document for this candidate
                if similarity_score > candidate_matches[normalized_name]['similarity_score']:
                    candidate_matches[normalized_name]['best_document'] = doc
                    candidate_matches[normalized_name]['similarity_score'] = similarity_score
        
        # Calculate final scores and create results
        final_matches = []
        
        for normalized_name, match_data in candidate_matches.items():
            if not match_data['best_document']:
                continue
            
            # Get the best snippet for analysis
            best_snippet = self._find_best_snippet(
                match_data['best_document'], 
                match_data['matched_keywords']
            )
            
            # Generate LLM explanation if requested
            explanation = ""
            if explain:
                explanation = self.explain_match_with_llm(
                    job_description, 
                    best_snippet, 
                    normalized_name
                )
            
            final_matches.append({
                'candidate_name': normalized_name,
                'original_name': match_data['original_name'],
                'similarity_score': match_data['similarity_score'],
                'matched_keywords': sorted(list(match_data['matched_keywords'])),
                'best_snippet': best_snippet,
                'explanation': explanation,
                'num_matches': len(match_data['matched_keywords']),
                'metadata': match_data['best_document'].metadata
            })
        
        # Sort by similarity score (primary) and number of matches (secondary)
        final_matches.sort(key=lambda x: (x['similarity_score'], x['num_matches']), reverse=True)
        
        # Always take top 5 for main results
        top_5_matches = final_matches[:5]
        
        # Get detailed information about all relevant candidates
        all_relevant_detailed = self.get_all_relevant_candidates_detailed(job_description)
        
        return {
            'job_description': job_description,
            'extracted_keywords': sorted(list(jd_keywords)),
            'total_keywords': len(jd_keywords),
            'top_candidates': top_5_matches,
            'all_candidates': final_matches,  # All candidates for reference
            'total_candidates_analyzed': len(candidate_matches),
            'candidates_with_relevance': len([m for m in final_matches if m['similarity_score'] > 0.1]),
            'all_relevant_detailed': all_relevant_detailed
        }
    
    def _find_best_snippet(self, document: Document, matched_keywords: set, max_length: int = 400) -> str:
        """Find the most relevant snippet from candidate document"""
        content = document.page_content
        
        # Skip header information (first 200 characters often contain contact info)
        if len(content) > 200:
            main_content = content[200:]
        else:
            main_content = content
        
        # Find sections with highest keyword density
        sentences = main_content.split('.')
        best_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            sentence_lower = sentence.lower()
            keyword_count = sum(1 for kw in matched_keywords if kw in sentence_lower)
            
            if keyword_count > 0:
                best_sentences.append((sentence, keyword_count))
        
        # Sort by keyword density and take top sentences
        best_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Combine top sentences into snippet
        snippet = ""
        for sentence, _ in best_sentences[:3]:  # Top 3 relevant sentences
            if len(snippet + sentence) < max_length:
                snippet += sentence + ". "
            else:
                break
        
        # If no good sentences found, use beginning of content
        if not snippet.strip():
            snippet = main_content[:max_length]
        
        return snippet.strip()
    
    def format_results(self, results: Dict[str, Any], show_snippets: bool = False) -> str:
        """Format matching results for display with ALL RELEVANT CANDIDATES section"""
        output = []
        output.append(f"\n🎯 Job Match Analysis Results")
        output.append(f"{'='*50}")
        output.append(f"📋 Job Keywords Extracted: {results['total_keywords']}")
        output.append(f"🔍 Top Keywords: {', '.join(results['extracted_keywords'][:10])}")
        output.append(f"👥 Total Candidates Analyzed: {results['total_candidates_analyzed']}")
        output.append(f"✅ Candidates with Good Relevance: {results['candidates_with_relevance']}")
        output.append(f"🏆 Top 5 Matches:")
        output.append("="*50 + "\n")
        
        # Show top 5 candidates
        for i, candidate in enumerate(results['top_candidates'], 1):
            output.append(f"{i}. 👤 **{candidate['candidate_name']}**")
            output.append(f"   📊 Similarity Score: {candidate['similarity_score']:.1%}")
            output.append(f"   🎯 Keyword Matches ({candidate['num_matches']}): {', '.join(candidate['matched_keywords'][:8])}")
            
            if show_snippets and candidate['best_snippet']:
                output.append(f"   📄 Relevant Experience: {candidate['best_snippet'][:150]}...")
            
            if candidate['explanation']:
                output.append(f"   🤖 Analysis: {candidate['explanation']}")
            
            output.append("")
        
        # Add ALL RELEVANT CANDIDATES section
        output.append("="*40)
        output.append("ALL RELEVANT CANDIDATES:")
        output.append("-"*40)
        
        if results.get('all_relevant_detailed'):
            output.append(results['all_relevant_detailed'])
        else:
            output.append("Detailed candidate information not available.")
        
        return "\n".join(output)
    
    def batch_match_multiple_jobs(self, job_descriptions: List[Dict[str, str]], top_k: int = 5) -> Dict[str, Any]:
        """Match multiple job descriptions and return consolidated results"""
        all_results = {}
        
        for job_info in job_descriptions:
            job_title = job_info.get('title', 'Untitled Job')
            job_desc = job_info.get('description', '')
            
            logger.info(f"Processing job: {job_title}")
            results = self.match_job_to_cvs(job_desc, top_k, explain=False)  # Skip explanations for batch
            all_results[job_title] = results
        
        return all_results
    
    def find_candidates_by_skills(self, required_skills: List[str], preferred_skills: List[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """Find candidates based on required and preferred skills"""
        # Create synthetic job description
        job_desc = f"Looking for candidates with the following required skills: {', '.join(required_skills)}"
        
        if preferred_skills:
            job_desc += f". Preferred additional skills: {', '.join(preferred_skills)}"
        
        return self.match_job_to_cvs(job_desc, top_k, explain=True)
    
    def get_skill_gap_analysis(self, job_description: str, top_k: int = 5) -> Dict[str, Any]:
        """Analyze skill gaps between job requirements and top candidates"""
        results = self.match_job_to_cvs(job_description, top_k, explain=False)
        
        job_keywords = set(results['extracted_keywords'])
        
        skill_gap_analysis = []
        for candidate in results['top_candidates']:
            candidate_keywords = set(candidate['matched_keywords'])
            missing_skills = job_keywords - candidate_keywords
            
            skill_gap_analysis.append({
                'candidate_name': candidate['candidate_name'],
                'matched_skills': list(candidate_keywords),
                'missing_skills': list(missing_skills),
                'skill_coverage': len(candidate_keywords) / len(job_keywords) if job_keywords else 0
            })
        
        return {
            'job_requirements': list(job_keywords),
            'candidates_analysis': skill_gap_analysis,
            'overall_results': results
        }