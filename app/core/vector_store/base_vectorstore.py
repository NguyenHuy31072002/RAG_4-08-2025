from typing import List
import uuid

from app.core.schema.response_schema import (
    DocumentEmbeddingRequest,
    RequestResult,
    RetrievedResult
)


class BaseVectorStore:
    def __init__(self, embedding_dims: int):
        self._embedding_dims = embedding_dims

    async def insert_embedding(self, doc_request: DocumentEmbeddingRequest) -> RequestResult:
        """
        Insert new document embedding into the vector store.
        """
        raise NotImplementedError()

    async def delete_point(self, doc_id: str) -> RequestResult:
        """
        Delete a document's vector based on its doc_id.
        """
        raise NotImplementedError()

    async def retrieve_points(self, embedding: List[float], similarity_top_k: int = 3) -> List[RetrievedResult]:
        """
        Retrieve top-k similar documents based on the input embedding.
        """
        raise NotImplementedError()
