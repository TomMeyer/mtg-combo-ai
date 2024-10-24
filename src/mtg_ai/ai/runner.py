from typing import Any, Optional

from mtg_ai.ai.ai_model import MTGCardAI
from mtg_ai.ai.rag import MTGRAGSearchSystem, SearchResult


class MTGAIRunner:
    def __init__(
        self,
        ai_model_name: str = "./results",
        rag_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.ai = MTGCardAI(model_name=ai_model_name)
        self.rag = MTGRAGSearchSystem(
            database=self.ai.database, embedding_model=rag_embedding_model_name
        )
        self.rag.index_cards()

    def run(
        self,
        query: str,
        max_new_tokens: int = 500,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        temperature: float = 0.11,
        min_p: float = 0.1,
    ) -> str:
        search_results: list[SearchResult] = self.rag.search(
            query, filters=filters, top_k=top_k
        )
        formatted_rag_data = "\n".join([result.format() for result in search_results])

        return self.ai.run(
            query,
            rag_data=formatted_rag_data,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            min_p=min_p,
        )

    def batch_run(
        self,
        queries: list[str],
        max_new_tokens: int = 500,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
    ) -> list[str]:
        raise NotImplementedError("Batch run not implemented yet")
