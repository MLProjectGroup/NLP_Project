# job_recommender.py
"""Match each candidate to top AI jobs using Google Gemini for explanations and embeddings"""
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
import numpy as np
import logging
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

# Modern AI job roles with Egyptian market focus
AI_JOB_LIST = [
    {
        "title": "AI Engineer",
        "description": "Design and implement AI solutions using modern frameworks like LangChain. "
                       "Develop Agentic AI systems and CrewAI multi-agent solutions. "
                       "Experience with LangGraph for workflow orchestration. "
                       "Knowledge of cloud platforms like AWS or Azure. Egyptian market experience preferred."
    },
    {
        "title": "Data Scientist",
        "description": "Extract insights from complex datasets using statistical analysis and machine learning. "
                       "Proficiency in Python, SQL, and data visualization tools. Experience with predictive modeling "
                       "and A/B testing. Experience with Egyptian datasets and business contexts."
    },
    {
        "title": "ML Engineer",
        "description": "Build and deploy machine learning models at scale. Experience with ML frameworks like "
                       "Scikit-learn, TensorFlow, or PyTorch. Knowledge of MLOps practices. "
                       "Familiarity with Egyptian tech ecosystem and deployment requirements."
    },
    {
        "title": "MLOps Engineer",
        "description": "Implement and maintain ML pipelines and infrastructure. Experience with containerization (Docker), "
                       "orchestration (Kubernetes), and CI/CD pipelines. Knowledge of monitoring and logging for ML systems. "
                       "Experience with Egyptian cloud infrastructure providers."
    },
    {
        "title": "Computer Vision Engineer",
        "description": "Develop computer vision applications and systems. Experience with OpenCV, deep learning for "
                       "image processing, and frameworks like TensorFlow/PyTorch. Knowledge of image recognition, "
                       "object detection, and video analysis techniques. Experience with Egyptian industrial applications."
    },
    {
        "title": "NLP Engineer",
        "description": "Build natural language processing systems and applications. Experience with transformer models "
                       "(BERT, GPT), text classification, and language generation. Proficiency in Python and NLP libraries "
                       "like spaCy, NLTK, or Hugging Face Transformers. Arabic language processing experience preferred."
    },
    {
        "title": "Generative AI Engineer",
        "description": "Develop and deploy generative AI models for various applications. Experience with large language models, "
                       "prompt engineering, and RAG systems. Knowledge of diffusion models, GANs, and other generative techniques. "
                       "Familiarity with vector databases and retrieval systems. Experience with LangChain and CrewAI."
    },
    {
        "title": "Agentic AI Specialist",
        "description": "Design and implement autonomous AI agents. Expertise in LangChain for agent development, CrewAI for multi-agent collaboration, "
                       "and LangGraph for complex workflow orchestration. Experience with self-improving AI systems. "
                       "Knowledge of Egyptian AI regulations and ethical considerations."
    }
]

def cosine_similarity(vec1, vec2) -> float:
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class JobRecommender:
    """Recommend top candidates for AI roles using Gemini for explanations and embeddings"""
    
    def __init__(self):
        # Initialize Google Gemini embeddings
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize Gemini for explanations
        self.llm = ChatGroq(
            model=os.getenv("Groq_model2", "meta-llama/llama-4-scout-17b-16e-instruct"),
            groq_api_key=os.getenv("Groq_API_KEY"),
            temperature=0.0,
        )
        
        self.job_list = AI_JOB_LIST
    
    def explain_match(self, cv_text: str, job_title: str, job_description: str) -> str:
        """Use Gemini to explain why a candidate is a good fit for a job"""
        prompt = [
            SystemMessage(content="You are an AI career advisor specializing in the Egyptian tech market. "
                                  "Analyze how well a candidate fits a job role and provide a concise 2-3 sentence explanation."),
            HumanMessage(content=f"""
Candidate CV Excerpt:
\"\"\"{cv_text[:3000]}\"\"\"

Job Title: {job_title}
Job Description:
\"\"\"{job_description}\"\"\"

Focus your analysis on:
1. Matching technical skills (mention specific frameworks like LangChain, CrewAI, or LangGraph if relevant)
2. Relevant experience for Egyptian companies like VOIS or GizaSystems
3. Cultural fit for Egyptian workplace
4. Any gaps that might affect performance

Provide your analysis in 2-3 concise sentences.
""")
        ]
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return f"⚠️ Explanation unavailable: {str(e)}"
    
    def get_top_candidates_for_jobs(self, candidate_cv_map: Dict[str, str], top_k: int = 5) -> Dict[str, List[Tuple[str, float, str]]]:
        """
        Get top candidates for each job role with Gemini explanations
        
        Args:
            candidate_cv_map: Dictionary mapping candidate names to their CV text
            top_k: Number of top candidates to return per job
            
        Returns:
            Dictionary mapping job titles to lists of (candidate_name, similarity_score, explanation)
        """
        # Precompute job embeddings
        job_embeddings = {}
        for job in self.job_list:
            job_text = f"{job['title']}: {job['description']}"
            job_embeddings[job['title']] = self.embedding_model.embed_query(job_text)
        
        # Calculate similarity for each candidate against each job
        job_candidates = {job['title']: [] for job in self.job_list}
        
        for candidate_name, cv_text in candidate_cv_map.items():
            cv_embedding = self.embedding_model.embed_query(cv_text)
            
            for job in self.job_list:
                job_title = job['title']
                job_embedding = job_embeddings[job_title]
                similarity = cosine_similarity(cv_embedding, job_embedding)
                job_candidates[job_title].append((candidate_name, similarity, cv_text))
        
        # Get top candidates with explanations
        final_recommendations = {}
        
        for job_title, candidates in job_candidates.items():
            # Sort candidates by similarity
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = candidates[:top_k]
            
            # Get Gemini explanations for top candidates
            candidates_with_explanations = []
            for candidate_name, score, cv_text in top_candidates:
                explanation = self.explain_match(cv_text, job_title, self.get_job_description(job_title))
                candidates_with_explanations.append((candidate_name, score, explanation))
                time.sleep(1)  # Avoid rate limiting
            
            final_recommendations[job_title] = candidates_with_explanations
        
        return final_recommendations
    
    def get_job_description(self, job_title: str) -> str:
        """Get job description by title"""
        for job in self.job_list:
            if job['title'] == job_title:
                return job['description']
        return ""
    
    def save_recommendations_to_file(self, recommendations: Dict[str, List[Tuple[str, float, str]]], filename: str) -> None:
        """
        Save job recommendations to a text file with explanations
        
        Args:
            recommendations: Dictionary of job recommendations with explanations
            filename: Output file path
        """
        content = "TOP CANDIDATES FOR AI ROLES (EGYPTIAN MARKET FOCUS)\n"
        content += "=" * 80 + "\n\n"
        
        for job_title, candidates in recommendations.items():
            content += f"Job Title: {job_title}\n"
            content += "-" * 60 + "\n"
            content += f"Top 5 Candidates:\n"
            
            for i, (candidate, score, explanation) in enumerate(candidates, 1):
                content += f"  {i}. {candidate} (Score: {score:.2f})\n"
                content += f"     Explanation: {explanation}\n\n"
            
            content += "\n"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved job recommendations to {filename}")