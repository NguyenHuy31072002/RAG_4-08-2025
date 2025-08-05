from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers.development.rag_basic import rag_basic
from .routers.development.insert_data_qdrant import insert_data_qdrant
import time
from app.startup.startup import init_qdrant_service, get_qdrant_service


# tag
tags_metadata = [
    {
        "name": "RAG Basic",
        "description": "Basic RAG operations",
    },
]

app = FastAPI(
    title="RAG Basic API",
    description="API for basic RAG operations",
    version="1.0.0",
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods, adjust as needed
    allow_headers=["*"],  # Allows all headers, adjust as needed
)   

app.include_router(rag_basic, prefix="/rag_basic", tags=["RAG Basic"])
app.include_router(insert_data_qdrant, prefix="/insert_data_qdrant", tags=["Insert Data Qdrant"])

@app.on_event("startup")
async def startup_event():
    """
    Startup event to initialize resources if needed.
    """
    start = time.perf_counter()
    print("Starting up the application...")
    
    # Simulate some startup tasks
    init_qdrant_service()
    end = time.perf_counter()
    print(f"Application started in {end - start:.2f} seconds.")
    
    # Optionally, you can initialize other services or resources here
    qdrant_service = get_qdrant_service()
    print(f"Qdrant service initialized: {qdrant_service}")
    # Here you can add any startup logic, like connecting to a database
    print("Application startup: initializing resources...")

