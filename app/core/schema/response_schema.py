from pydantic import BaseModel
from strenum import StrEnum
from typing import List, Optional

from typing import List, Optional, Any, Dict

from pydantic import BaseModel
from typing import List, Dict, Any

class DocumentEmbeddingRequest(BaseModel):
    doc_id: str
    embedding: List[float]
    content: str
    metadata: Dict[str, Any]

class RequestResult(BaseModel):
    request_id: str
    status: str
    request_type: str
    data: Optional[Any] = None

class RetrievedResult(BaseModel):
    score: float
    payload: dict

class DocumentEmbeddingRequest(BaseModel):
    doc_id: str
    embedding: List[float]
    content: str  # Nội dung văn bản
    metadata: Dict[str, Any]


class BatchQueryRequest(BaseModel):
    queries: List[str]
    top_k: int = 3
    search_method: str = "hybrid"  # 'vector', 'keyword', 'hybrid'
    alpha: float = 0.5  # Dành cho hybrid
    
class DeleteRequest(BaseModel):
    ids: List[str]

class DeleteResponse(BaseModel):
    message: str
    deleted_ids: List[str]
    failed_ids: List[str]

class UploadResponse(BaseModel):
    message: str
    chunk_count: int 
    ids: List[str]

class DeleteAllResponse(BaseModel):
    success: bool
    message: str


class ResponseStatus(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"




class RequestType(StrEnum):
    QUERY = "QUERY"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"




# ==== Request Schema ====
class Message(BaseModel):
    role: str
    content: str

class QueryPayload(BaseModel):
    data: List[Message]
