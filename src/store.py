from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401
            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = doc.metadata.copy()
        metadata["doc_id"] = doc.id
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content)
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_vec = self._embedding_fn(query)
        
        # Tính similarity cho từng record
        scored_records = []
        for rec in records:
            score = compute_similarity(query_vec, rec["embedding"])
            scored_records.append({
                **rec,
                "score": score
            })
        
        # Sắp xếp giảm dần theo score
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        
        # Trả về top_k
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma:
            # Chuẩn bị dữ liệu cho Chroma
            ids = [doc.id for doc in docs]
            documents = [doc.content for doc in docs]
            embeddings = [self._embedding_fn(doc.content) for doc in docs]
            
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings
            )
        else:
            # In-memory
            for doc in docs:
                self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                include=['documents', 'embeddings', 'metadatas', 'distances']
            )
            # Format kết quả cho giống in-memory
            formatted = []
            for i in range(len(results['ids'][0])):
                formatted.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "embedding": results['embeddings'][0][i],
                    "score": 1.0 - results['distances'][0][i]  # Chroma trả về distance, ta đổi sang similarity
                })
            return formatted
        else:
            # In-memory
            query_vec = self._embedding_fn(query)
            
            scored_records = []
            for rec in self._store:
                score = compute_similarity(query_vec, rec["embedding"])
                scored_records.append({
                    **rec,
                    "score": score
                })
            
            scored_records.sort(key=lambda x: x["score"], reverse=True)
            return scored_records[:top_k]

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        else:
            return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma:
            # Chroma hỗ trợ filter trực tiếp
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=metadata_filter,
                include=['documents', 'embeddings', 'metadatas', 'distances']
            )
            # Format kết quả
            formatted = []
            for i in range(len(results['ids'][0])):
                formatted.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "embedding": results['embeddings'][0][i],
                    "score": 1.0 - results['distances'][0][i]
                })
            return formatted
        else:
            # In-memory: filter thủ công
            query_vec = self._embedding_fn(query)
            
            filtered_records = []
            for rec in self._store:
                # Kiểm tra metadata có khớp không
                match = True
                if metadata_filter:
                    for key, value in metadata_filter.items():
                        if rec["metadata"].get(key) != value:
                            match = False
                            break
                
                if match:
                    score = compute_similarity(query_vec, rec["embedding"])
                    filtered_records.append({
                        **rec,
                        "score": score
                    })
            
            filtered_records.sort(key=lambda x: x["score"], reverse=True)
            return filtered_records[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            # Chroma hỗ trợ delete theo where
            self._collection.delete(where={"doc_id": doc_id})
            return True  # Giả sử thành công, hoặc có thể check count trước
        else:
            # In-memory: filter và remove
            initial_count = len(self._store)
            self._store = [rec for rec in self._store if rec["metadata"].get("doc_id") != doc_id]
            return len(self._store) < initial_count
