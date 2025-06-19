# hr_question_generator.py
import os
import re
import logging
from typing import Dict, List
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class HRQuestionGenerator:
    def __init__(self):
        """Initialize HR Question Generator with Groq AI"""
        # Configure Groq LLM
        self.llm = ChatGroq(
            model=os.getenv("Groq_model2", "llama3-8b-8192"),
            groq_api_key=os.getenv("Groq_API_KEY"),
            temperature=0.3,  # Slightly higher for more creative questions
        )
        
        # Question categories and templates
        self.question_categories = {
            "technical": "Technical Skills & Experience",
            "behavioral": "Behavioral & Soft Skills", 
            "experience": "Work Experience & Projects",
            "situational": "Situational & Problem-Solving",
            "motivation": "Motivation & Career Goals"
        }
    
    def extract_candidate_info(self, cv_content: str, candidate_name: str) -> Dict:
        """Extract key information from CV content"""
        info = {
            "name": candidate_name,
            "technical_skills": [],
            "experience": [],
            "projects": [],
            "education": [],
            "achievements": [],
            "languages": []
        }
        
        cv_lower = cv_content.lower()
        
        # Extract technical skills (common programming languages and technologies)
        tech_keywords = [
            "python", "java", "javascript", "react", "node.js", "html", "css", 
            "mongodb", "sql", "docker", "aws", "machine learning", "deep learning",
            "data analysis", "ai", "artificial intelligence", "c++", "c#", "kotlin",
            "angular", "vue.js", "typescript", "tensorflow", "opencv", "pandas",
            "scikit-learn", "git", "linux", "mysql", "postgresql", "firebase"
        ]
        
        for skill in tech_keywords:
            if skill in cv_lower:
                info["technical_skills"].append(skill.title())
        
        # Extract experience keywords
        experience_keywords = [
            "internship", "intern", "work experience", "job", "position", 
            "role", "freelance", "volunteer", "teaching assistant", "instructor"
        ]
        
        for exp in experience_keywords:
            if exp in cv_lower:
                info["experience"].append(exp.title())
        
        # Extract project-related keywords
        if "project" in cv_lower:
            info["projects"].append("Multiple Projects Mentioned")
        
        return info
    
    def generate_questions_with_ai(self, candidate_info: Dict, cv_content: str) -> Dict[str, List[str]]:
        """Generate personalized HR questions using Groq AI"""
        
        prompt = f"""
        You are an experienced HR professional conducting interviews for technical positions. 
        Based on the following candidate's CV information, generate 3-4 relevant interview questions for each category.
        
        Candidate: {candidate_info['name']}
        Technical Skills: {', '.join(candidate_info['technical_skills'][:10])}  # Limit to avoid long prompts
        Experience: {', '.join(candidate_info['experience'])}
        
        CV Content Summary (first 1000 characters):
        {cv_content[:1000]}...
        
        Please generate specific, relevant questions for each category. Make questions personalized to this candidate's background.
        
        Categories:
        1. Technical Skills & Experience
        2. Behavioral & Soft Skills
        3. Work Experience & Projects  
        4. Situational & Problem-Solving
        5. Motivation & Career Goals
        
        Format your response as:
        **Technical Skills & Experience:**
        - Question 1
        - Question 2
        - Question 3
        
        **Behavioral & Soft Skills:**
        - Question 1
        - Question 2
        - Question 3
        
        **Work Experience & Projects:**
        - Question 1
        - Question 2
        - Question 3
        
        **Situational & Problem-Solving:**
        - Question 1
        - Question 2
        - Question 3
        
        **Motivation & Career Goals:**
        - Question 1
        - Question 2
        - Question 3
        """
        
        try:
            # Create message for Groq
            message = HumanMessage(content=prompt)
            
            # Generate response using Groq
            response = self.llm.invoke([message])
            
            # Extract the text content from the response
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            return self._parse_ai_response(response_text)
        except Exception as e:
            logger.error(f"Error generating AI questions for {candidate_info['name']}: {e}")
            return self._generate_fallback_questions(candidate_info)
    
    def _parse_ai_response(self, response_text: str) -> Dict[str, List[str]]:
        """Parse AI response and extract questions by category"""
        questions = {
            "Technical Skills & Experience": [],
            "Behavioral & Soft Skills": [],
            "Work Experience & Projects": [],
            "Situational & Problem-Solving": [],
            "Motivation & Career Goals": []
        }
        
        current_category = None
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a category header
            for category in questions.keys():
                if category.lower() in line.lower() and ('**' in line or ':' in line):
                    current_category = category
                    break
            
            # Check if line is a question (starts with - or number)
            if current_category and (line.startswith('-') or line.startswith('•') or 
                                   re.match(r'^\d+\.', line)):
                question = line.lstrip('-•').lstrip('0123456789.').strip()
                if question and len(question) > 10:  # Valid question
                    questions[current_category].append(question)
        
        # Ensure each category has at least some questions
        for category in questions:
            if len(questions[category]) == 0:
                # Add fallback questions for empty categories
                questions[category] = self._get_category_fallback_questions(category)[:3]
        
        return questions
    
    def _get_category_fallback_questions(self, category: str) -> List[str]:
        """Get fallback questions for a specific category"""
        fallback_questions = {
            "Technical Skills & Experience": [
                "Can you walk me through your most significant technical project?",
                "What programming languages are you most proficient in and why?",
                "How do you approach learning new technologies?"
            ],
            "Behavioral & Soft Skills": [
                "Describe a situation where you had to work with a difficult team member.",
                "How do you handle constructive criticism?",
                "Tell me about a time you showed leadership skills."
            ],
            "Work Experience & Projects": [
                "What has been your most challenging project to date?",
                "How do you prioritize tasks when working on multiple projects?",
                "Describe your role in a successful team project."
            ],
            "Situational & Problem-Solving": [
                "How would you handle a situation where you missed a deadline?",
                "Describe your approach to debugging complex issues.",
                "What would you do if you disagreed with your manager's technical decision?"
            ],
            "Motivation & Career Goals": [
                "What attracted you to this field?",
                "Where do you see your career in 5 years?",
                "What type of work environment helps you thrive?"
            ]
        }
        
        return fallback_questions.get(category, ["Tell me more about your experience."])
    
    def _generate_fallback_questions(self, candidate_info: Dict) -> Dict[str, List[str]]:
        """Generate fallback questions if AI fails"""
        name = candidate_info['name']
        skills = candidate_info['technical_skills']
        
        return {
            "Technical Skills & Experience": [
                f"Can you walk me through your experience with {skills[0] if skills else 'programming'}?",
                f"What technical challenges have you faced in your projects, {name}?",
                "How do you stay updated with the latest technology trends?"
            ],
            "Behavioral & Soft Skills": [
                "Describe a time when you had to work in a team. What was your role?",
                "How do you handle tight deadlines and pressure?",
                "Tell me about a time you had to learn something new quickly."
            ],
            "Work Experience & Projects": [
                "Tell me about your most challenging project and how you overcame obstacles.",
                "What was your role in your recent internship/work experience?",
                "How do you approach problem-solving in your projects?"
            ],
            "Situational & Problem-Solving": [
                "How would you debug a program that's not working as expected?",
                "If you disagreed with a team member's approach, how would you handle it?",
                "What would you do if you couldn't meet a project deadline?"
            ],
            "Motivation & Career Goals": [
                "What motivates you to work in this field?",
                "Where do you see yourself in 3-5 years?",
                "Why are you interested in this position?"
            ]
        }
    
    def format_questions_for_candidate(self, candidate_name: str, questions: Dict[str, List[str]]) -> str:
        """Format questions for a single candidate"""
        output = f"HR INTERVIEW QUESTIONS FOR {candidate_name.upper()}\n"
        output += "=" * 80 + "\n\n"
        
        for category, question_list in questions.items():
            output += f"{category.upper()}:\n"
            output += "-" * 40 + "\n"
            
            for i, question in enumerate(question_list, 1):
                output += f"{i}. {question}\n"
            
            output += "\n"
        
        output += "=" * 80 + "\n\n"
        return output
    
    def generate_questions_for_top_candidates(self, candidate_cv_map: Dict[str, str], top_candidates: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """Generate questions for top 5 candidates"""
        all_questions = {}
        
        logger.info(f"Generating questions for candidates: {top_candidates}")
        
        for candidate_name in top_candidates[:5]:  # Top 5 candidates
            if candidate_name in candidate_cv_map:
                cv_content = candidate_cv_map[candidate_name]
                
                print(f"🤖 Generating HR questions for: {candidate_name}")
                
                # Extract candidate information
                candidate_info = self.extract_candidate_info(cv_content, candidate_name)
                
                # Generate questions using AI
                questions = self.generate_questions_with_ai(candidate_info, cv_content)
                
                all_questions[candidate_name] = questions
                
                print(f"✅ Generated {sum(len(q) for q in questions.values())} questions for {candidate_name}")
            else:
                logger.warning(f"No CV content found for candidate: {candidate_name}")
        
        return all_questions
    
    def save_questions_to_file(self, all_questions: Dict[str, Dict[str, List[str]]], filename: str):
        """Save all generated questions to a file without date"""
        output_content = "HR INTERVIEW QUESTIONS FOR TOP 5 CANDIDATES\n"
        output_content += "=" * 100 + "\n\n"
        output_content += f"Total Candidates: {len(all_questions)}\n\n"
        output_content += "=" * 100 + "\n\n"
        
        if len(all_questions) == 0:
            output_content += "No candidates were provided for question generation.\n"
            output_content += "Please ensure the query results contain valid candidate rankings.\n"
        else:
            for i, (candidate_name, questions) in enumerate(all_questions.items(), 1):
                output_content += f"CANDIDATE {i}: {candidate_name.upper()}\n"
                output_content += "=" * 80 + "\n\n"
                
                for category, question_list in questions.items():
                    output_content += f"{category.upper()}:\n"
                    output_content += "-"
                    output_content += "-" * 50 + "\n"
                    
                    for j, question in enumerate(question_list, 1):
                        output_content += f"{j}. {question}\n"
                    
                    output_content += "\n"
                
                output_content += "=" * 80 + "\n\n"
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        logger.info(f"HR questions saved to {filename}")