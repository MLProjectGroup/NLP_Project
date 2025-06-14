#vector_store.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import chromadb
import os
from dotenv import load_dotenv
import logging
import time
import hashlib
import shutil
from typing import List
import numpy as np

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
            compressed = self._compress_text_for_embedding(text) if len(text) > 10000 else text
            for attempt in range(self.max_retries):
                try:
                    embedding = self.base_embeddings.embed_documents([compressed])[0]
                    embeddings.append(embedding)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Embedding failed after {self.max_retries} attempts: {e}")
                        embeddings.append(self._create_fallback_embedding(text))
                    else:
                        logger.warning(f"Retrying embed {i+1}/{len(texts)}: {e}")
                        time.sleep((attempt + 1) * 2)
            time.sleep(0.3)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        for attempt in range(self.max_retries):
            try:
                return self.base_embeddings.embed_query(text)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Query embedding failed: {e}")
                    return self._create_fallback_embedding(text)
                time.sleep((attempt + 1) * 2)

    def _compress_text_for_embedding(self, text: str, aggressive=False) -> str:
        text = " ".join(text.split())
        if len(text) > 8000:
            return text[:3000] + "\n...\n" + text[len(text) // 2:len(text) // 2 + 2000] + "\n...\n" + text[-3000:]
        return text

    def _create_fallback_embedding(self, text: str) -> List[float]:
        hash_ = hashlib.md5(text.encode()).hexdigest()
        vector = [(int(hash_[i:i+2], 16) / 127.5 - 1.0) for i in range(0, len(hash_), 2)]
        return (vector * (768 // len(vector) + 1))[:768]


class CVVectorStore:
    def __init__(self):
        persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_store")
        collection_name = os.getenv("VECTOR_DB_COLLECTION", "cv_collection")

        base_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.embeddings = SafeGoogleGenerativeAIEmbeddings(base_embeddings)

        self.client = chromadb.PersistentClient(path=persist_directory)

        try:
            # Try loading existing collection
            self.collection = self.client.get_collection(collection_name)
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            logger.info(f"✅ Loaded existing Chroma collection: {collection_name}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load collection due to: {e}. Resetting DB...")
            shutil.rmtree(persist_directory, ignore_errors=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            logger.info(f"🆕 Created new Chroma collection: {collection_name}")

    def add_cvs(self, documents, metadata=None):
        logger.info(f"🔄 Adding {len(documents)} CVs to vectorstore")
        successful = 0
        failed = []

        for i, doc in enumerate(documents):
            try:
                self.vectorstore.add_documents([doc])
                successful += 1
            except Exception as e:
                logger.error(f"❌ Failed to add doc {i+1}: {e}")
                failed.append({
                    "doc": doc,
                    "error": str(e),
                    "candidate": doc.metadata.get("candidate_name", "Unknown")
                })

        if failed:
            self._save_failed_documents(failed)
        logger.info(f"✅ Successfully added {successful}/{len(documents)} documents.")
        return successful, failed

    def _save_failed_documents(self, failed):
        import json
        error_log = [{
            "candidate": f["candidate"],
            "error": f["error"],
            "preview": f["doc"].page_content[:300]
        } for f in failed]
        with open("failed_documents.json", "w", encoding="utf-8") as f:
            json.dump(error_log, f, indent=2, ensure_ascii=False)
        logger.info("📝 Saved failed document logs to failed_documents.json")

    def similarity_search(self, query, k=5, filter=None):
        try:
            return self.vectorstore.similarity_search(query=query, k=k, filter=filter)
        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            return []

    def get_all_candidates(self):
        try:
            all_docs = self.vectorstore.get()
            return list({md.get("candidate_name") for md in all_docs["metadatas"] if md and "candidate_name" in md})
        except Exception as e:
            logger.error(f"❌ Failed to fetch candidates: {e}")
            return []

