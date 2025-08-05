from fastapi import HTTPException
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from typing import List, Dict
import uuid

from app.core.schema.response_schema import (
    DeleteResponse,
    DeleteRequest,
    DeleteAllResponse,
    RequestResult,
    ResponseStatus,
    RequestType,
    RetrievedResult,
    DocumentEmbeddingRequest
)

from app.core.vector_store.base_vectorstore import BaseVectorStore


class QdrantService(BaseVectorStore):
    def __init__(
        self,
        embedding_dims: int,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "mardown",
        distance: Distance = Distance.COSINE
    ):
        super().__init__(embedding_dims=embedding_dims)
        self._client = AsyncQdrantClient(host=host, port=port)
        self._collection_name = collection_name
        self._distance_metric = distance

    async def create_collection(self):
        if not await self._client.collection_exists(self._collection_name):
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dims,
                    distance=self._distance_metric
                )
            )

    def _validate_embedding(self, embedding: List[float]):
        if len(embedding) != self._embedding_dims:
            raise HTTPException(
                status_code=400,
                detail=f"Embedding size must be {self._embedding_dims}, got {len(embedding)}"
            )

    async def insert_embedding(self, doc_request: DocumentEmbeddingRequest) -> RequestResult:
        self._validate_embedding(doc_request.embedding)

        if await self._get_point_id(doc_request.doc_id):
            raise HTTPException(
                status_code=409,
                detail=f"Document with id {doc_request.doc_id} already exists."
            )

        payload = {
            "doc_id": doc_request.doc_id,
            "content": doc_request.content,
            "metadata": doc_request.metadata
        }

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=doc_request.embedding,
            payload=payload
        )

        await self._client.upsert(collection_name=self._collection_name, points=[point])

        return RequestResult(
            request_id=str(uuid.uuid4()),
            status=ResponseStatus.COMPLETED,
            request_type=RequestType.INSERT
        )

    async def _get_point_id(self, doc_id: str, limit: int = 1):
        payload_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

        result = await self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=payload_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        return result[0][0] if result[0] else None

    async def delete_embeddings(self, doc_ids: List[str]) -> DeleteResponse:
        doc_ids = list(set(doc_ids))  # Remove duplicates
        point_ids_to_delete = []
        failed_ids = []

        for doc_id in doc_ids:
            point = await self._get_point_id(doc_id)
            if point:
                point_ids_to_delete.append(point.id)
            else:
                failed_ids.append(doc_id)

        if not point_ids_to_delete:
            return DeleteResponse(
                deleted_ids=[],
                failed_ids=failed_ids,
                message="No matching documents found for deletion."
            )

        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=point_ids_to_delete
        )

        deleted_ids = [doc_id for doc_id in doc_ids if doc_id not in failed_ids]

        return DeleteResponse(
            deleted_ids=deleted_ids,
            failed_ids=failed_ids,
            message=f"Deleted {len(deleted_ids)} documents, failed {len(failed_ids)}."
        )

    async def delete_all_data(self) -> DeleteAllResponse:
        try:
            if not await self._client.collection_exists(self._collection_name):
                return DeleteAllResponse(
                    success=False,
                    message="Collection does not exist."
                )

            await self._client.delete_collection(self._collection_name)
            await self.create_collection()

            return DeleteAllResponse(
                success=True,
                message="All data in the collection has been deleted and collection recreated."
            )
        except Exception as e:
            return DeleteAllResponse(
                success=False,
                message=f"Failed to delete all data: {str(e)}"
            )

    async def delete_point(self, doc_id: str) -> RequestResult:
        point = await self._get_point_id(doc_id)
        if not point:
            raise HTTPException(status_code=404, detail=f"Document with id {doc_id} not found.")

        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=[point.id]
        )

        return RequestResult(
            request_id=str(uuid.uuid4()),
            status=ResponseStatus.COMPLETED,
            request_type=RequestType.DELETE,
            data=point.payload
        )

    async def retrieve_points(self, embedding: List[float], similarity_top_k: int = 3) -> List[RetrievedResult]:
        self._validate_embedding(embedding)

        results = await self._client.search(
            collection_name=self._collection_name,
            query_vector=embedding,
            limit=similarity_top_k
        )

        return [
            RetrievedResult(score=result.score, payload=result.payload)
            for result in results
        ]

    async def batch_retrieve(self, embeddings: List[List[float]], top_k: int = 3) -> List[List[RetrievedResult]]:
        for emb in embeddings:
            self._validate_embedding(emb)

        requests = [{"vector": vector, "limit": top_k} for vector in embeddings]

        results = await self._client.search_batch(
            collection_name=self._collection_name,
            requests=requests
        )

        return [
            [
                RetrievedResult(
                    score=res.score,
                    payload=res.payload if res.payload is not None else {}
                ) for res in group
            ]
            for group in results
        ]

    async def get_all_documents(self) -> List[Dict]:
        results = []
        collections_response = await self._client.get_collections()
        collections = collections_response.collections

        for collection in collections:
            offset = None
            while True:
                points, next_page = await self._client.scroll(
                    collection_name=collection.name,
                    offset=offset,
                    limit=100,
                    with_payload=True,
                    with_vectors=False
                )
                for point in points:
                    payload = point.payload or {}
                    text = payload.get("content", "") or payload.get("text", "")
                    results.append({
                        "id": point.id,
                        "text": text
                    })
                if next_page is None:
                    break
                offset = next_page

        return results
