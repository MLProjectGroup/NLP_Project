# cv_summarizer.py - Enhanced CV Summarization with Line Formatting
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Optional
import time
import json
from datetime import datetime
import re

load_dotenv()
logger = logging.getLogger(__name__)

class CVSummarizer:
    """
    Enhanced CV Summarizer using Groq API
    Generates professional 3-4 line summaries of candidate CVs with proper formatting
    """
    
    def __init__(self, temperature=0.1, max_retries=3):
        """
        Initialize CV Summarizer
        
        Args:
            temperature: Controls randomness in generation (0.1 for more focused responses)
            max_retries: Number of retry attempts for failed API calls
        """
        self.max_retries = max_retries
        
        try:
            # Initialize LangChain Groq client
            self.llm = ChatGroq(
                model=os.getenv("Groq_model2", "meta-llama/llama-4-scout-17b-16e-instruct2"),
                groq_api_key=os.getenv("Groq_API_KEY"),
                temperature=0.0,
            )
            logger.info("✅ CV Summarizer initialized successfully with Groq API")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize CV Summarizer: {e}")
            raise
    
    def _create_summary_prompt(self, cv_text: str, candidate_name: str = None) -> List:
        """
        Create structured prompt for CV summarization
        
        Args:
            cv_text: The CV content to summarize
            candidate_name: Optional candidate name for personalization
            
        Returns:
            List of messages for the LLM
        """
        system_prompt = """You are an expert HR assistant specializing in CV analysis and candidate summarization.
Your task is to create concise, professional summaries that help recruiters quickly understand a candidate's profile.

SUMMARIZATION GUIDELINES:
- Generate exactly 3-4 lines (not more, not less)
- Each line should be a complete sentence
- Include key technical and soft skills
- Mention most recent or relevant roles/projects
- Include years of experience if mentioned
- Add education background briefly
- Use professional, clear language
- Focus on achievements and capabilities
- Avoid redundant information
- Ensure each line is properly capitalized and punctuated

FORMAT: Return only the summary text with line breaks between each point."""
        
        candidate_info = f" for {candidate_name}" if candidate_name else ""
        
        human_prompt = f"""Please create a professional CV summary{candidate_info}.

CV Content:
'''
{cv_text}
'''

Generate a concise 3-4 line professional summary that captures the candidate's key qualifications, experience, and skills. 
Format each point on a separate line with proper capitalization and punctuation."""
        
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
    
    def summarize_cv(self, cv_text: str, candidate_name: str = None) -> str:
        """
        Summarize a single CV
        
        Args:
            cv_text: The CV content to summarize
            candidate_name: Optional candidate name
            
        Returns:
            Professional CV summary (3-4 lines)
        """
        if not cv_text or not cv_text.strip():
            return "⚠️ No CV content provided for summarization"
        
        # Truncate very long CVs to avoid token limits
        max_length = 8000
        if len(cv_text) > max_length:
            logger.warning(f"CV text too long ({len(cv_text)} chars), truncating to {max_length}")
            cv_text = cv_text[:max_length] + "..."
        
        for attempt in range(self.max_retries):
            try:
                # Create prompt messages
                messages = self._create_summary_prompt(cv_text, candidate_name)
                
                # Generate summary
                response = self.llm.invoke(messages)
                summary = response.content.strip()
                
                # Validate summary length (should be 3-4 lines)
                lines = [line.strip() for line in summary.split('\n') if line.strip()]
                if len(lines) < 2 or len(lines) > 4:
                    logger.warning(f"Summary has {len(lines)} lines (expected 3-4), retrying...")
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue
                
                # Clean up the summary while preserving line breaks
                summary = self._clean_summary(summary)
                
                logger.info(f"✅ Successfully summarized CV for {candidate_name or 'candidate'}")
                return summary
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # Wait before retry with exponential backoff
                    wait_time = (attempt + 1) * 2
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    logger.error(f"❌ Failed to summarize CV after {self.max_retries} attempts")
                    return f"⚠️ Unable to generate summary: {str(e)}"
    
    def _clean_summary(self, summary: str) -> str:
        """
        Clean and format the generated summary while preserving line breaks
        
        Args:
            summary: Raw summary from LLM
            
        Returns:
            Cleaned and formatted summary
        """
        # Remove any unwanted prefixes or labels
        unwanted_prefixes = [
            "Summary:", "CV Summary:", "Professional Summary:",
            "Candidate Summary:", "Profile:", "Overview:"
        ]
        
        for prefix in unwanted_prefixes:
            if summary.startswith(prefix):
                summary = summary[len(prefix):].strip()
        
        # Clean up formatting but preserve line breaks
        summary = summary.replace('**', '').replace('*', '')  # Remove markdown
        
        # Ensure each line is properly formatted
        cleaned_lines = []
        for line in summary.split('\n'):
            line = line.strip()
            if line:
                # Ensure proper capitalization and punctuation
                if not line[0].isupper():
                    line = line[0].upper() + line[1:]
                if not line.endswith('.'):
                    line += '.'
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def summarize_multiple_cvs(self, cv_data: List[Dict[str, str]], batch_delay: float = 1.0) -> Dict[str, str]:
        """
        Summarize multiple CVs with batch processing
        
        Args:
            cv_data: List of dictionaries with 'candidate_name' and 'cv_text' keys
            batch_delay: Delay between processing each CV (to avoid rate limiting)
            
        Returns:
            Dictionary mapping candidate names to their summaries
        """
        logger.info(f"Starting batch summarization of {len(cv_data)} CVs")
        
        summaries = {}
        successful = 0
        failed = []
        
        for i, cv_info in enumerate(cv_data, 1):
            candidate_name = cv_info.get('candidate_name', f'Candidate_{i}')
            cv_text = cv_info.get('cv_text', '')
            
            logger.info(f"Processing {i}/{len(cv_data)}: {candidate_name}")
            
            try:
                summary = self.summarize_cv(cv_text, candidate_name)
                summaries[candidate_name] = summary
                successful += 1
                
                # Progress indicator
                if i % 5 == 0:
                    logger.info(f"Progress: {i}/{len(cv_data)} CVs processed")
                
            except Exception as e:
                logger.error(f"Failed to summarize CV for {candidate_name}: {e}")
                failed.append({
                    'candidate_name': candidate_name,
                    'error': str(e)
                })
                summaries[candidate_name] = f"⚠️ Summarization failed: {str(e)}"
            
            # Delay between requests to avoid rate limiting
            if i < len(cv_data) and batch_delay > 0:
                time.sleep(batch_delay)
        
        logger.info(f"✅ Batch summarization complete: {successful}/{len(cv_data)} successful")
        
        if failed:
            logger.warning(f"Failed candidates: {[f['candidate_name'] for f in failed]}")
            self._save_failed_summaries(failed)
        
        return summaries
    
    def _save_failed_summaries(self, failed_summaries: List[Dict]):
        """Save information about failed summarizations"""
        try:
            filename = f"failed_summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(failed_summaries, f, indent=2, ensure_ascii=False)
            logger.info(f"Failed summaries saved to {filename}")
        except Exception as e:
            logger.error(f"Could not save failed summaries: {e}")
    
    def save_summaries_to_file(self, summaries: Dict[str, str], filename: str = None) -> str:
        """
        Save summaries to a text file with proper formatting
        
        Args:
            summaries: Dictionary of candidate names to summaries
            filename: Optional custom filename
            
        Returns:
            Path to the saved file
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"cv_summaries_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("CV SUMMARIES REPORT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total candidates: {len(summaries)}\n")
                f.write("=" * 60 + "\n\n")
                
                for i, (candidate_name, summary) in enumerate(summaries.items(), 1):
                    f.write(f"{i}. {candidate_name}\n")
                    f.write("-" * 40 + "\n")
                    f.write(summary)  # Keep the existing line breaks
                    f.write("\n\n")
            
            logger.info(f"✅ Summaries saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Failed to save summaries to file: {e}")
            return None
    
    def get_summary_statistics(self, summaries: Dict[str, str]) -> Dict:
        """
        Get statistics about generated summaries
        
        Args:
            summaries: Dictionary of summaries
            
        Returns:
            Dictionary with statistics
        """
        if not summaries:
            return {}
        
        total_summaries = len(summaries)
        successful_summaries = len([s for s in summaries.values() if not s.startswith("⚠️")])
        failed_summaries = total_summaries - successful_summaries
        
        # Calculate average summary length
        successful_texts = [s for s in summaries.values() if not s.startswith("⚠️")]
        avg_length = sum(len(s) for s in successful_texts) / len(successful_texts) if successful_texts else 0
        avg_lines = sum(len(s.split('\n')) for s in successful_texts) / len(successful_texts) if successful_texts else 0
        
        return {
            'total_summaries': total_summaries,
            'successful_summaries': successful_summaries,
            'failed_summaries': failed_summaries,
            'success_rate': (successful_summaries / total_summaries) * 100 if total_summaries > 0 else 0,
            'average_length_chars': round(avg_length, 1),
            'average_lines': round(avg_lines, 1)
        }



