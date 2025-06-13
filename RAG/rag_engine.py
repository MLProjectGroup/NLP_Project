# rag_engine.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain.schema import Document
from pydantic import Field
from typing import List, Dict, Any, Tuple
import os
from dotenv import load_dotenv
import logging
import re
from collections import defaultdict

load_dotenv()
logger = logging.getLogger(__name__)

class DiverseCVRetriever(BaseRetriever):
    """Custom retriever that ensures diverse candidate selection"""
    
    vector_store: Any = Field(description="The vector store instance")
    search_kwargs: Dict[str, Any] = Field(default_factory=lambda: {"k": 500}, description="Search parameters")
    max_candidates: int = Field(default=10, description="Maximum number of unique candidates to retrieve")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """Retrieve documents ensuring diverse candidate selection"""
        # Get a large number of documents initially
        all_docs = self.vector_store.similarity_search(query, k=self.search_kwargs.get("k", 500))

        # Group documents by candidate
        candidate_docs = defaultdict(list)
        for doc in all_docs:
            candidate_name = doc.metadata.get('candidate_name', 'Unknown Candidate')
            candidate_docs[candidate_name].append(doc)
        
        # Select the best document for each candidate (first one, as they're sorted by relevance)
        diverse_docs = []
        for candidate_name, docs in candidate_docs.items():
            if len(diverse_docs) >= self.max_candidates:
                break
            
            # Take the most relevant document for this candidate
            best_doc = docs[0]
            
            # Format with clear candidate identification
            formatted_content = f"""
=== CANDIDATE: {candidate_name.upper()} ===
{best_doc.page_content}
=== END OF {candidate_name.upper()}'S CV ===
"""
            
            formatted_doc = Document(
                page_content=formatted_content,
                metadata=best_doc.metadata
            )
            diverse_docs.append(formatted_doc)
        
        logger.info(f"Retrieved {len(diverse_docs)} unique candidates for query: {query}")
        return diverse_docs

class EnhancedRAGEngine:
    def __init__(self, vector_store, max_candidates_per_query=10):
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-pro"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            convert_system_message_to_human=True
        )
        
        self.vector_store = vector_store
        self.max_candidates_per_query = max_candidates_per_query
        
        # Create diverse retriever
        self.diverse_retriever = DiverseCVRetriever(
            vector_store=vector_store.vectorstore,
            search_kwargs={"k": 500},  # Get many documents initially
            max_candidates=max_candidates_per_query
        )
        
        # Enhanced ranking prompt
        self.ranking_prompt = PromptTemplate(
            template="""You are an intelligent recruitment assistant that ranks candidates based on their relevance to a specific query.

CRITICAL INSTRUCTIONS:
1. Each candidate's CV is clearly separated with "=== CANDIDATE: [NAME] ===" headers
2. Analyze each unique candidate separately
3. Rate each candidate's relevance to the query on a scale of 1-10
4. Provide specific reasons for the rating based on their actual qualifications
5. Return ONLY the top 5 most relevant candidates
6. DO NOT repeat any candidate name - each candidate should appear only once
7. If fewer than 5 candidates are relevant, return only the relevant ones

Context from Different Candidates' CVs:
{context}

Query: {question}

Please analyze each unique candidate and rank them. Return your response in this EXACT format:

RANKING RESULTS:
1. [Candidate Name] - Score: [X/10]
   Relevance: [Specific explanation based on their CV content]

2. [Candidate Name] - Score: [X/10]
   Relevance: [Specific explanation based on their CV content]

3. [Candidate Name] - Score: [X/10]
   Relevance: [Specific explanation based on their CV content]

4. [Candidate Name] - Score: [X/10]
   Relevance: [Specific explanation based on their CV content]

5. [Candidate Name] - Score: [X/10]
   Relevance: [Specific explanation based on their CV content]

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Standard QA prompt
        self.qa_prompt = PromptTemplate(
            template="""You are an intelligent recruitment assistant analyzing CVs from different candidates.

CRITICAL INSTRUCTIONS:
1. Each CV section below is clearly marked with "=== CANDIDATE: [NAME] ===" headers
2. Information within each section belongs ONLY to that specific candidate
3. Do NOT mix information between candidates
4. Always specify which candidate has which experience
5. When listing candidates, ensure each person is mentioned only once

Context from Different Candidates' CVs:
{context}

Question: {question}

Analyze each candidate's CV separately and provide accurate information about WHO has what experience.

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Create chains with diverse retriever
        self.ranking_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.diverse_retriever,
            chain_type_kwargs={
                "prompt": self.ranking_prompt,
                "verbose": False
            }
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.diverse_retriever,
            chain_type_kwargs={
                "prompt": self.qa_prompt,
                "verbose": False
            }
        )
    
    def find_top_candidates(self, query: str, top_k: int = 5) -> str:
        """Find and rank the top K candidates based on the query"""
        try:
            logger.info(f"Finding top {top_k} candidates for query: {query}")
            
            # Get relevant documents first to see what we're working with
            docs = self.diverse_retriever._get_relevant_documents(query)
            candidate_names = []
            for doc in docs:
                candidate_name = doc.metadata.get('candidate_name', 'Unknown')
                if candidate_name not in candidate_names:
                    candidate_names.append(candidate_name)
            
            logger.info(f"Found {len(candidate_names)} unique candidates: {candidate_names}")
            
            # Use ranking chain to get structured response
            response = self.ranking_chain.run(query)
            
            # Parse and clean the response to ensure no duplicates
            cleaned_response = self._clean_ranking_response(response, top_k)
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error finding top candidates: {e}")
            return f"I encountered an error while ranking candidates: {str(e)}"
    
    def _clean_ranking_response(self, response: str, top_k: int) -> str:
        """Clean the ranking response to remove duplicates and limit results"""
        lines = response.split('\n')
        seen_candidates = set()
        cleaned_lines = []
        candidate_count = 0
        
        for line in lines:
            # Check if this is a ranking line
            match = re.match(r'^(\d+)\.\s*([^-]+)', line.strip())
            if match:
                candidate_name = match.group(2).strip()
                
                # Skip if we've already seen this candidate
                if candidate_name.upper() in seen_candidates:
                    continue
                
                # Add to seen candidates
                seen_candidates.add(candidate_name.upper())
                candidate_count += 1
                
                # Update the numbering
                cleaned_line = f"{candidate_count}. {candidate_name}" + line[line.find('-'):]
                cleaned_lines.append(cleaned_line)
                
                # Stop if we've reached the limit
                if candidate_count >= top_k:
                    break
            else:
                # Include non-ranking lines (like headers, relevance explanations)
                if line.strip() and not line.startswith('Answer:'):
                    cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def query(self, question: str) -> str:
        """Process a general question and return an answer"""
        try:
            logger.info(f"Processing query: {question}")
            response = self.qa_chain.run(question)
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def get_all_candidates_for_skill(self, skill: str) -> str:
        """Get all candidates who have a specific skill"""
        query = f"List ALL candidates who have {skill}. For each candidate, provide their name and relevant details about their {skill} experience."
        return self.query(query)
    
    def get_candidate_summary_with_ranking(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        """Get both ranking and detailed candidate information"""
        try:
            # Get top candidates
            ranking_result = self.find_top_candidates(query, top_k)
            
            # Extract candidate names from ranking
            candidate_names = []
            for line in ranking_result.split('\n'):
                match = re.match(r'^\d+\.\s*([^-]+)', line.strip())
                if match:
                    candidate_names.append(match.group(1).strip())
            
            # Get detailed info for each top candidate
            detailed_info = []
            for name in candidate_names[:top_k]:
                try:
                    info = self.get_candidate_info(name)
                    detailed_info.append({
                        'name': name,
                        'details': info
                    })
                except Exception as e:
                    logger.warning(f"Could not get details for {name}: {e}")
            
            return ranking_result, detailed_info
            
        except Exception as e:
            logger.error(f"Error in candidate summary with ranking: {e}")
            return f"Error: {str(e)}", []
    
    def get_candidate_info(self, candidate_name: str) -> str:
        """Get detailed information about a specific candidate"""
        query = f"Tell me everything about {candidate_name} including their education, experience, skills, and contact information"
        return self.query(query)
    
    def compare_candidates(self, skill_or_experience: str) -> str:
        """Compare all candidates for a specific skill or experience"""
        query = f"Which candidates have {skill_or_experience}? List each candidate separately with their relevant experience."
        return self.query(query)
    
    def get_candidates_with_education(self, education_type: str) -> str:
        """Get all candidates with specific education background"""
        query = f"List ALL candidates who have {education_type} education. For each candidate, provide their name and educational background details."
        return self.query(query)