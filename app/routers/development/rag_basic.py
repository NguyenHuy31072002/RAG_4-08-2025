from typing import List
from fastapi import APIRouter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from app.core.config.constant import EMBEDDING_DIMS, EMBEDDING_MODEL, MODEL_RERANKER
from fastapi.concurrency import run_in_threadpool
import os
from dotenv import load_dotenv
from app.reflection.core import Reflection
from app.core.embedding.embeddings import EmbeddingsWrapper
from app.semantic_router.route import Route
from app.semantic_router.router import SemanticRouter

from app.db.qdrant.qdrant_service import QdrantService
from app.semantic_router.samples import productsSample, chitchatSample
from app.core.re_rank.core import Reranker

from together import Together
from app.core.schema.response_schema import QueryPayload

load_dotenv()

rag_basic = APIRouter()

reflector = Reflection()

PRODUCT_ROUTE_NAME = "products"
CHITCHAT_ROUTE_NAME = "chitchat"

productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productsSample)
chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)

base_embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cuda:0"})
embedding_wrapper = EmbeddingsWrapper(base_embeddings)

qdrant_service = QdrantService(embedding_dims=EMBEDDING_DIMS)

semanticRouter = SemanticRouter(embedding_wrapper, routes=[productRoute, chitchatRoute])
reranker = Reranker(model_name=MODEL_RERANKER)
qdrant_service = QdrantService(embedding_dims=EMBEDDING_DIMS)

# ==== LLM client: ví dụ TogetherClient (sửa tuỳ theo model bạn dùng) ====
from together import Together
together_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

def process_query(query):
    return query.lower()

@rag_basic.post("/search")
async def handle_query(payload: QueryPayload):
    data = [m.dict() for m in payload.data]
    print(f"Received data: {data}")

    # Step 1: Reflect query
    query = reflector(data)
    print(f"Reflected query: {query}")
    query_processed = process_query(query)
    # Step 2: Semantic route
    guided_score, guided_route = semanticRouter.guide(query_processed)
    print(f"Guided route: {guided_route}, Score: {guided_score}")

    if guided_route == PRODUCT_ROUTE_NAME:
        # Step 3: vector search từ RAG (Qdrant)
        query_embedding = embedding_wrapper.encode([query])
        print("len", len(query_embedding))
        retrieved = await qdrant_service.retrieve_points(
            embedding=query_embedding,
            similarity_top_k=5
        )
        passages = [r.payload.get("content", "") for r in retrieved]

        # Step 4: rerank
        scores, ranked_passages = reranker(query, passages)
        print(f"Ranked passages: {ranked_passages}")
        print(f"Scores: {scores}")
        source_information = "\n".join([f"{i+1}. {p}" for i, p in enumerate(ranked_passages)])

        # Step 5: tạo prompt
        combined_prompt = (
            f"Hãy trở thành chuyên gia tư vấn dịch vụ cho ngân hàng LPBank.\n"
            f"Câu hỏi của khách hàng: {query}\n"
            f"Dựa vào các thông tin sản phẩm sau, hãy trả lời:\n{source_information}"
        )
        data.append({
            "role": "user",
            "content": combined_prompt
        })

        # Step 6: Gọi LLM (ví dụ Together)
        response = together_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=data
        )
        final_response = response.choices[0].message.content
    else:
        # Chitchat - chỉ cần dùng LLM trả lời
        print("Guide to LLMs")
        response = together_client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=data
        )
        final_response = response.choices[0].message.content

    return {
        "content": final_response,
        "role": "assistant"
    }












