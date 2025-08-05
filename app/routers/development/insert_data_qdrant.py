from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os
import torch
from app.core.embedding.embeddings import EmbeddingsWrapper
from langchain_community.embeddings import SentenceTransformerEmbeddings
from app.db.qdrant.qdrant_service import QdrantService
from app.core.schema.response_schema import UploadResponse, DocumentEmbeddingRequest
from app.core.config.constant import EMBEDDING_MODEL
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_docling.loader import ExportType
from langchain_text_splitters import RecursiveCharacterTextSplitter

insert_data_qdrant = APIRouter()


@insert_data_qdrant.post("/insert_pdf", response_model=UploadResponse)
async def insert_pdf(file: UploadFile = File(...)):
    """Xử lý file PDF - giữ nguyên logic cũ"""
    # B1. Ghi file PDF tạm vào đĩa
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        pdf_path = tmp.name

    try:
        # B2. Dùng DoclingLoader để tự load & chunk PDF
        loader = DoclingLoader(
            file_path=pdf_path,
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(tokenizer=EMBEDDING_MODEL)
        )
        docs = loader.load()
        chunks = [doc.page_content for doc in docs]

        if not chunks:
            return {
                "message": "No text chunks were extracted from the PDF.",
                "chunk_count": 0,
                "ids": []
            }

        # B3. Encode embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_embeddings = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        embeddings = EmbeddingsWrapper(base_embeddings).encode(chunks)

        # B4. Insert vào Qdrant
        qdrant = QdrantService(embedding_dims=len(embeddings[0]))
        await qdrant.create_collection()

        ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = f"id_{i}"
            req = DocumentEmbeddingRequest(
                doc_id=doc_id,
                embedding=embedding,
                content=chunk,
                metadata={"source": file.filename, "index": i}
            )
            await qdrant.insert_embedding(req)
            ids.append(doc_id)

        return {
            "message": "PDF uploaded, processed and inserted to Qdrant successfully.",
            "chunk_count": len(chunks),
            "ids": ids
        }

    finally:
        # B5. Dọn dẹp file tạm
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


@insert_data_qdrant.post("/insert_markdown", response_model=UploadResponse)
async def insert_markdown(file: UploadFile = File(...)):
    """Xử lý file Markdown"""
    # Kiểm tra extension file
    if not file.filename.lower().endswith('.md'):
        raise HTTPException(status_code=400, detail="File must be a Markdown (.md) file")
    
    # B1. Đọc nội dung file markdown
    try:
        content = await file.read()
        markdown_content = content.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Unable to decode file as UTF-8")

    try:
        # B2. Chunk nội dung markdown
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Có thể điều chỉnh
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(markdown_content)

        if not chunks:
            return {
                "message": "No text chunks were extracted from the Markdown file.",
                "chunk_count": 0,
                "ids": []
            }

        # B3. Encode embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_embeddings = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        embeddings = EmbeddingsWrapper(base_embeddings).encode(chunks)

        # B4. Insert vào Qdrant
        qdrant = QdrantService(embedding_dims=len(embeddings[0]))
        await qdrant.create_collection()

        ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = f"md_{i}"
            req = DocumentEmbeddingRequest(
                doc_id=doc_id,
                embedding=embedding,
                content=chunk,
                metadata={"source": file.filename, "index": i, "type": "markdown"}
            )
            await qdrant.insert_embedding(req)
            ids.append(doc_id)

        return {
            "message": "Markdown file uploaded, processed and inserted to Qdrant successfully.",
            "chunk_count": len(chunks),
            "ids": ids
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing markdown file: {str(e)}")


@insert_data_qdrant.post("/insert_document", response_model=UploadResponse)
async def insert_document(file: UploadFile = File(...)):
    """Endpoint thống nhất xử lý cả PDF và Markdown"""
    filename = file.filename.lower()
    
    if filename.endswith('.pdf'):
        return await insert_pdf(file)
    elif filename.endswith('.md'):
        return await insert_markdown(file)
    else:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file type. Only PDF and Markdown (.md) files are supported."
        )