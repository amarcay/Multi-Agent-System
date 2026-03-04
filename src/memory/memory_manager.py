"""
Gestionnaire de mémoire persistante utilisant ChromaDB et sentence-transformers.
Permet le stockage et la recherche sémantique de faits, préférences et résumés
de conversations par session utilisateur.
"""

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor


class MemoryManager:
    """Gestionnaire de mémoire persistante avec ChromaDB."""

    def __init__(self, persist_dir: str = "./data/memory_chroma"):
        self._persist_dir = persist_dir
        self._executor = ThreadPoolExecutor(max_workers=2)

        # Client ChromaDB persistant
        self._client = chromadb.PersistentClient(path=persist_dir)

        # Modèle d'embeddings local
        self._embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        # Collections
        self._facts_collection = self._client.get_or_create_collection(
            name="user_memories",
            metadata={"description": "Faits et préférences utilisateur par session"},
        )
        self._summaries_collection = self._client.get_or_create_collection(
            name="conversation_summaries",
            metadata={"description": "Résumés de conversations par session"},
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Génère les embeddings pour une liste de textes."""
        return self._embeddings.embed_documents(texts)

    def _embed_query(self, text: str) -> list[float]:
        """Génère l'embedding pour une requête."""
        return self._embeddings.embed_query(text)

    async def add_fact(
        self, session_id: str, fact: str, metadata: dict | None = None
    ) -> None:
        """Ajoute un fait ou une préférence utilisateur."""
        meta = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "type": "fact",
        }
        if metadata:
            meta.update(metadata)

        embedding = await asyncio.get_event_loop().run_in_executor(
            self._executor, self._embed_query, fact
        )

        self._facts_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[fact],
            embeddings=[embedding],
            metadatas=[meta],
        )

    async def add_facts(self, session_id: str, facts: list[str]) -> None:
        """Ajoute plusieurs faits d'un coup."""
        if not facts:
            return

        ids = [str(uuid.uuid4()) for _ in facts]
        metadatas = [
            {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "type": "fact",
            }
            for _ in facts
        ]
        embeddings = await asyncio.get_event_loop().run_in_executor(
            self._executor, self._embed, facts
        )

        self._facts_collection.add(
            ids=ids,
            documents=facts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def add_summary(self, session_id: str, summary: str) -> None:
        """Ajoute un résumé de conversation."""
        meta = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "type": "summary",
        }

        embedding = await asyncio.get_event_loop().run_in_executor(
            self._executor, self._embed_query, summary
        )

        self._summaries_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[summary],
            embeddings=[embedding],
            metadatas=[meta],
        )

    async def retrieve_memories(
        self, session_id: str, query: str, top_k: int = 5
    ) -> list[str]:
        """Recherche sémantique de faits pertinents pour une session."""
        if self._facts_collection.count() == 0:
            return []

        query_embedding = await asyncio.get_event_loop().run_in_executor(
            self._executor, self._embed_query, query
        )

        results = self._facts_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._facts_collection.count()),
            where={"session_id": session_id},
        )

        if results and results["documents"]:
            return results["documents"][0]
        return []

    async def retrieve_all_memories(
        self, query: str, top_k: int = 5
    ) -> list[str]:
        """Recherche sémantique de faits pertinents sur toutes les sessions."""
        if self._facts_collection.count() == 0:
            return []

        query_embedding = await asyncio.get_event_loop().run_in_executor(
            self._executor, self._embed_query, query
        )

        results = self._facts_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._facts_collection.count()),
        )

        if results and results["documents"]:
            return results["documents"][0]
        return []

    async def get_recent_summaries(
        self, session_id: str, limit: int = 3
    ) -> list[str]:
        """Récupère les résumés de conversation les plus récents pour une session."""
        if self._summaries_collection.count() == 0:
            return []

        results = self._summaries_collection.get(
            where={"session_id": session_id},
            limit=limit,
        )

        if results and results["documents"]:
            return results["documents"]
        return []

    def cleanup(self) -> None:
        """Nettoyage des ressources."""
        self._executor.shutdown(wait=False)
