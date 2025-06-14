# vector_store.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import chromadb
import os
from dotenv import load_dotenv
import logging
import time
import hashlib
from typing import List
import numpy as np
import shutil

load_dotenv()
logger = logging.getLogger(__name__)
persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/cv_database")

# Clean corrupted old DB
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)


class SafeGoogleGenerativeAIEmbeddings(Embeddings):
    """Wrapper around Google embeddings with timeout handling and text compression"""
    
    def __init__(self, base_embeddings, max_retries=3, initial_timeout=30):
        self.base_embeddings = base_embeddings
        self.max_retries = max_retries
        self.initial_timeout = initial_timeout
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with retry logic and text compression for large documents"""
        embeddings = []
        
        for i, text in enumerate(texts):
            text_size = len(text)
            logger.info(f"Embedding document {i+1}/{len(texts)} (size: {text_size} chars)")
            
            # If text is too large, create a compressed version for embedding
            if text_size > 10000:  # Adjust threshold as needed
                logger.warning(f"Large document detected ({text_size} chars). Using compressed version for embedding.")
                compressed_text = self._compress_text_for_embedding(text)
            else:
                compressed_text = text
            
            # Try to embed with retries
            for attempt in range(self.max_retries):
                try:
                    # Embed single document
                    embedding = self.base_embeddings.embed_documents([compressed_text])[0]
                    embeddings.append(embedding)
                    break
                    
                except Exception as e:
                    if "504" in str(e) or "Deadline" in str(e):
                        logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}. Error: {e}")
                        
                        if attempt < self.max_retries - 1:
                            # Wait before retry with exponential backoff
                            wait_time = (attempt + 1) * 5
                            logger.info(f"Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                            
                            # If still failing, compress more aggressively
                            if attempt == 1:
                                compressed_text = self._compress_text_for_embedding(text, aggressive=True)
                        else:
                            # Last resort: use a fallback embedding
                            logger.error(f"Failed to embed after {self.max_retries} attempts. Using fallback embedding.")
                            embeddings.append(self._create_fallback_embedding(text))
                    else:
                        raise e
            
            # Small delay between documents to avoid rate limiting
            if i < len(texts) - 1:
                time.sleep(0.5)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return self.base_embeddings.embed_query(text)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    logger.error(f"Failed to embed query after {self.max_retries} attempts")
                    return self._create_fallback_embedding(text)
    
    def _compress_text_for_embedding(self, text: str, aggressive: bool = False) -> str:
        """Compress text for embedding while preserving key information"""
        if aggressive:
            # More aggressive compression: extract key sections only
            sections = {
                "skills": self._extract_section(text, ["skills", "technical", "competencies"]),
                "experience": self._extract_section(text, ["experience", "work", "employment"]),
                "education": self._extract_section(text, ["education", "academic", "qualification"]),
                "summary": text[:1000]  # First 1000 chars as summary
            }
            
            compressed = "\n".join([f"{k}: {v[:500]}" for k, v in sections.items() if v])
            return compressed[:5000]  # Limit to 5000 chars
        else:
            # Moderate compression: keep important parts
            # Remove excessive whitespace
            text = " ".join(text.split())
            
            # If still too long, take beginning, middle, and end
            if len(text) > 8000:
                chunk_size = 2500
                beginning = text[:chunk_size]
                middle_start = len(text) // 2 - chunk_size // 2
                middle = text[middle_start:middle_start + chunk_size]
                end = text[-chunk_size:]
                
                text = f"{beginning}\n...\n{middle}\n...\n{end}"
            
            return text
    
    def _extract_section(self, text: str, keywords: List[str]) -> str:
        """Extract section based on keywords"""
        text_lower = text.lower()
        for keyword in keywords:
            idx = text_lower.find(keyword)
            if idx != -1:
                # Extract section (up to 1000 chars after keyword)
                return text[idx:idx + 1000]
        return ""
    
    def _create_fallback_embedding(self, text: str) -> List[float]:
        """Create a deterministic fallback embedding based on text content"""
        # Create a hash-based embedding
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to a vector (768 dimensions to match typical embedding size)
        embedding = []
        for i in range(0, len(text_hash), 2):
            # Convert hex pairs to float values between -1 and 1
            value = int(text_hash[i:i+2], 16) / 127.5 - 1.0
            embedding.append(value)
        
        # Pad or truncate to standard size (768)
        target_size = 768
        if len(embedding) < target_size:
            # Repeat pattern to fill
            embedding = embedding * (target_size // len(embedding) + 1)
        embedding = embedding[:target_size]
        
        # Add some variance based on text length and content
        text_features = [
            len(text) / 10000.0,  # Normalized length
            text.count(' ') / 1000.0,  # Word count approximation
            sum(1 for c in text if c.isupper()) / 100.0,  # Uppercase ratio
        ]
        
        for i, feature in enumerate(text_features):
            if i < len(embedding):
                embedding[i] = (embedding[i] + feature) / 2.0
        
        return embedding

class CVVectorStore:
    def __init__(self):
        # Initialize Google Gemini embeddings with safety wrapper
        base_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Wrap with safety features
        self.embeddings = SafeGoogleGenerativeAIEmbeddings(base_embeddings)
        
        # Initialize ChromaDB client
        persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/cv_database")
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize or get collection
        collection_name = os.getenv("VECTOR_DB_COLLECTION", "cv_collection")
        
        try:
            self.collection = self.client.get_collection(collection_name)
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_cvs(self, documents, metadata=None):
        """Add CV documents to the vector store with enhanced error handling"""
        logger.info(f"Starting to add {len(documents)} documents to vector store")
        
        successful = 0
        failed = []
        
        # Process documents one by one for better error handling
        for i, doc in enumerate(documents):
            try:
                logger.info(f"Adding document {i+1}/{len(documents)}: {doc.metadata.get('candidate_name', 'Unknown')}")
                self.vectorstore.add_documents([doc])
                successful += 1
                
            except Exception as e:
                logger.error(f"Failed to add document {i+1}: {e}")
                failed.append({
                    'document': doc,
                    'error': str(e),
                    'candidate': doc.metadata.get('candidate_name', 'Unknown')
                })
                
                # Continue with next document
                continue
            
            # Small delay between documents
            if i < len(documents) - 1:
                time.sleep(0.5)
        
        logger.info(f"Successfully added {successful}/{len(documents)} documents")
        
        if failed:
            logger.error(f"Failed documents: {[f['candidate'] for f in failed]}")
            # Save failed documents for manual review
            self._save_failed_documents(failed)
        
        return successful, failed
    
    def _save_failed_documents(self, failed_docs):
        """Save failed documents for manual review"""
        import json
        
        failed_info = []
        for item in failed_docs:
            doc = item['document']
            failed_info.append({
                'candidate': item['candidate'],
                'error': item['error'],
                'content_length': len(doc.page_content),
                'metadata': doc.metadata,
                'content_preview': doc.page_content[:500] + '...' if len(doc.page_content) > 500 else doc.page_content
            })
        with open('failed_documents.json', 'w', encoding='utf-8') as f:
            json.dump(failed_info, f, indent=2, ensure_ascii=False)
        
        logger.info("Failed documents saved to failed_documents.json")
    
    def similarity_search(self, query, k=5, filter=None):
        """Search for similar documents"""
        try:
            if filter:
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=filter
                )
            else:
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k
                )
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def get_all_candidates(self):
        """Get list of all unique candidates in the database"""
        try:
            # Get all documents
            all_docs = self.vectorstore.get()
            candidates = set()
            
            if all_docs and 'metadatas' in all_docs:
                for metadata in all_docs['metadatas']:
                    if metadata and 'candidate_name' in metadata:
                        candidates.add(metadata['candidate_name'])
            
            return list(candidates)
        except Exception as e:
            logger.error(f"Error getting candidates: {e}")
            return []
