from app.db.qdrant.qdrant_service import QdrantService, Distance

from app.core.config.constant import (
                                      EMBEDDING_DIMS)



qdrant_service = None

def init_qdrant_service() -> QdrantService:
    global qdrant_service
    qdrant_service = QdrantService(host = "qdrant",
                                   embedding_dims = EMBEDDING_DIMS,
                                   distance = Distance.COSINE)

def get_qdrant_service() -> QdrantService:
    return qdrant_service