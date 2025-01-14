from typing import Any, Optional

from huggingface_hub import ChatCompletionInputMessage
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    error_type: str


class RagRequest(BaseModel):
    query: ChatCompletionInputMessage
    filters: Optional[dict[str, Any]] = None
    top_k: int = 10


class RAGResponse(BaseModel):
    content: str
