import os
import shutil
import time
import json
import logging
import hashlib
import numpy as np
from typing import List

import chromadb
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

load_dotenv()
logger = logging.getLogger(__name__)

class SafeGoogleGenerativeAIEmbeddings(Embeddings):
    def __init__(self, base_embeddings, max_retries=3, initial_timeout=30):
        self.base_embeddings = base_embeddings
        self.max_retries = max_retries
        self.initial_timeout = initial_timeout

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i, text in enumerate(texts):
            for attempt in range(self.max_retries):
                try:
                    embedding = self.base_embeddings.embed_documents([text])[0]
                    embeddings.append(embedding)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed to embed after {self.max_retries} attempts. Using fallback.")
                        embeddings.append(self._create_fallback_embedding(text))
                    else:
                        logger.warning(f"Retrying embed... attempt {attempt+1}")
                        time.sleep(3)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        try:
            return self.base_embeddings.embed_query(text)
        except Exception:
            return self._create_fallback_embedding(text)

    def _create_fallback_embedding(self, text: str) -> List[float]:
        hash_digest = hashlib.md5(text.encode()).hexdigest()
        return [(int(hash_digest[i:i+2], 16) / 127.5 - 1.0) for i in range(0, len(hash_digest), 2)][:768]


class CVVectorStore:
    def __init__(self):
        # Load from env or fallback
        persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_store")
        collection_name = os.getenv("VECTOR_DB_COLLECTION", "cv_collection")

        # 🧹 Clean corrupted store before anything else
        if os.path.exists(persist_directory):
            logger.warning(f"Removing corrupted Chroma store at: {persist_directory}")
            shutil.rmtree(persist_directory)

        # 🔐 Create fresh client and embeddings
        base_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.embeddings = SafeGoogleGenerativeAIEmbeddings(base_embeddings)

        # 💾 Start Chroma DB
        self.client = chromadb.PersistentClient(path=persist_directory)

        # ✅ Create a new collection
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        logger.info(f"✅ Chroma vector store initialized with collection: {collection_name}")

    def add_cvs(self, documents):
        logger.info(f"Adding {len(documents)} documents to Chroma vector store")
        successes, failures = 0, []

        for i, doc in enumerate(documents):
            try:
                self.vectorstore.add_documents([doc])
                successes += 1
            except Exception as e:
                logger.error(f"Failed to add doc {i}: {e}")
                failures.append((doc.metadata.get("candidate_name", "Unknown"), str(e)))

        logger.info(f"✅ Successfully added {successes} documents. ❌ Failed: {len(failures)}")
        return successes, failures

    def similarity_search(self, query, k=5):
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return []

    def get_all_candidates(self):
        try:
            results = self.vectorstore.get()
            if "metadatas" in results:
                return list({meta.get("candidate_name", "Unknown") for meta in results["metadatas"]})
            return []
        except Exception as e:
            logger.error(f"Failed to get candidates: {e}")
            return []
