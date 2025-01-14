from __future__ import annotations

import logging
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response

from mtg_ai.ai.rag import MTGRAGSearchSystem
from mtg_ai.cards.database import MTGDatabase
from mtg_ai_rag_webserver.models import ErrorResponse, RagRequest, RAGResponse

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MTG AI RAG Webserver",
    description="MTG AI RAG",
    contact={"name": "Thomas Meyer"},
    license={
        "name": "MIT",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
    },
    version="0.1.0",
)

rag_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
card_database = MTGDatabase()
rag_service = MTGRAGSearchSystem(
    database=card_database, embedding_model=rag_embedding_model_name
)

@app.middleware("http")
async def log_requests(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    logger.info(f"Middleware Rquest: {request.method} URL: {request.url} Body: {await request.json()}")
    response = await call_next(request)
    logger.info(f"Middleware Response: {response.status_code}")
    return response


@app.post(
    "/query",
    response_model=RAGResponse,
    responses={
        "422": {"model": ErrorResponse},
        "424": {"model": ErrorResponse},
        "429": {"model": ErrorResponse},
        "500": {"model": ErrorResponse},
    },
)
async def query(request: RagRequest):
    message = None
    if isinstance(request.query.content, list):
        logger.info(f"Content: {request.query.content}")
        text = [t.text for t in request.query.content if t.type == "text" and t.text is not None]
        message = " ".join(text)
    elif isinstance(request.query.content, str):
        message = request.query.content
    if message is None:
        return ErrorResponse(error="Invalid message", error_type="InvalidMessage")
    logger.info(f"Performing RAG query for '{message}'")
    result = rag_service.search(message, request.filters, request.top_k)
    logger.info(f"Result Type: {type(result)}")
    return RAGResponse(content=result.format())


@app.post("/index")
def index():
    rag_service.index()
    return {"message": "Indexing complete"}


@app.delete("/index/{doc_id}")
def delete(doc_id: str):
    # rag_service.delete(doc_id)
    return {"message": "Document deleted"}


@app.get("/status")
def status():
    return {"status": "ok"}


@app.post("/update-embeddings")
def update_embeddings():
    # rag_service.update_embeddings()
    return {"message": "Embeddings updated"}
