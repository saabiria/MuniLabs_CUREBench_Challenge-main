# retrieval/medical_retriever.py
import os
import pickle
from pathlib import Path
from typing import List

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from retriever_embeddings import LocalEmbedder


class MedicalRetriever:
    """
    Production-ready RAG retriever with HyDE support.

    Key improvements over v1:
    - Distance threshold tightened from 50 -> 30 (fewer noisy matches)
    - Snippet length raised to 800 chars (more context per doc)
    - HyDE (Hypothetical Document Embedding) option for better MCQ recall
    - Returns up to top_k docs instead of being capped to 1 externally
    """

    def __init__(
        self,
        docs_path: str = "datasets/medical_docs/",
        index_path: str = "retrieval/faiss_index.bin",
        embeddings_path: str = "retrieval/doc_embeddings.pkl",
        top_k: int = 3,
        distance_threshold: float = 30.0,   # tightened from 50
        snippet_length: int = 800,           # raised from 500
    ):
        self.docs_path = Path(docs_path)
        self.index_path = index_path
        self.embeddings_path = embeddings_path
        self.top_k = top_k
        self.distance_threshold = distance_threshold
        self.snippet_length = snippet_length

        self.embedder = LocalEmbedder()
        self.documents = self.load_documents()

        if len(self.documents) == 0:
            raise ValueError("No documents found in datasets/medical_docs/")

        if os.path.exists(self.index_path) and os.path.exists(self.embeddings_path):
            self.load_index()
        else:
            self.build_index()

    # -----------------------------
    # DOCUMENT LOADING
    # -----------------------------
    def load_documents(self) -> List[str]:
        docs = []
        for file in sorted(self.docs_path.glob("*.txt")):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    docs.append(f.read())
            except Exception as e:
                print(f"Error reading {file}: {e}")
        return docs

    # -----------------------------
    # EMBEDDING
    # -----------------------------
    def embed(self, text: str) -> np.ndarray:
        return self.embedder.embed(text)

    # -----------------------------
    # INDEX BUILD / LOAD
    # -----------------------------
    def build_index(self):
        print("Building FAISS index (LOCAL embeddings)...")

        if faiss is None:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")

        embeddings = []
        for doc in self.documents:
            try:
                emb = self.embed(doc)
                embeddings.append(emb)
            except Exception as e:
                print("Embedding error:", e)

        if not embeddings:
            raise ValueError("No embeddings were created.")

        embeddings = np.array(embeddings).astype("float32")

        with open(self.embeddings_path, "wb") as f:
            pickle.dump(embeddings, f)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)
        self.embeddings = embeddings

        print(f"FAISS index built with {self.index.ntotal} vectors")

    def load_index(self):
        print("Loading FAISS index...")
        with open(self.embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f)
        self.index = faiss.read_index(self.index_path)
        print(f"FAISS index loaded: {self.index.ntotal} vectors, {len(self.documents)} docs")

    # -----------------------------
    # RETRIEVE  (standard)
    # -----------------------------
    def retrieve(self, query: str) -> List[str]:
        """Standard retrieval: embed the raw query and search."""
        if not query or not query.strip():
            return []
        try:
            query_emb = np.array([self.embed(query)]).astype("float32")
            return self._search(query_emb)
        except Exception as e:
            print("Retriever error:", e)
            return []

    # -----------------------------
    # RETRIEVE WITH HyDE
    # -----------------------------
    def retrieve_hyde(self, query: str, hypothesis: str) -> List[str]:
        """
        HyDE retrieval: embed a model-generated hypothetical answer
        instead of the raw question for better semantic alignment.

        Usage (from ClinicalAgent):
            hypothesis = self.deepseek.generate(
                f"Write a short expert answer to: {question}", max_tokens=120
            )
            docs = self.retriever.retrieve_hyde(question, hypothesis)

        Falls back to standard retrieval if hypothesis is empty.
        """
        if not hypothesis or not hypothesis.strip():
            return self.retrieve(query)

        try:
            hyp_emb = np.array([self.embed(hypothesis)]).astype("float32")
            return self._search(hyp_emb)
        except Exception as e:
            print("HyDE retriever error:", e)
            return self.retrieve(query)

    # -----------------------------
    # INTERNAL SEARCH
    # -----------------------------
    def _search(self, query_emb: np.ndarray) -> List[str]:
        """
        Search the FAISS index with the provided embedding vector.
        Applies distance_threshold and snippet_length constraints.
        """
        distances, indices = self.index.search(query_emb, self.top_k)
        results = []

        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            if distances[0][i] > self.distance_threshold:
                continue
            doc = self.documents[idx][: self.snippet_length]
            results.append(doc)

        return results


# -----------------------------
# GLOBAL WRAPPER
# -----------------------------
_retriever = None


def retrieve_medical_context(query: str, top_k: int = 3) -> str:
    global _retriever

    if _retriever is None:
        _retriever = MedicalRetriever(top_k=top_k)

    docs = _retriever.retrieve(query)

    if not docs:
        return "No relevant medical context found."

    return "\n\n---\n\n".join(docs)
