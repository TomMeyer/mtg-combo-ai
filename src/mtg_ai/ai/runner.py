from typing import Any, Generator, Literal, Optional, overload

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

    @overload
    def run(
        self,
        query: str,
        stream_output: Literal[False],
        max_new_tokens: int = ...,
        filters: Optional[dict[str, Any]] = ...,
        top_k: int = ...,
        temperature: float = ...,
        min_p: float = ...,
        history: Optional[list[dict[str, Any]]] = ...,
    ) -> str: ...

    @overload
    def run(
        self,
        query: str,
        stream_output: Literal[True],
        max_new_tokens: int = ...,
        filters: Optional[dict[str, Any]] = ...,
        top_k: int = ...,
        temperature: float = ...,
        min_p: float = ...,
        history: Optional[list[dict[str, Any]]] = ...,
    ) -> Generator[str, None, None]: ...

    def run(
        self,
        query: str,
        stream_output: bool = False,
        max_new_tokens: int = 500,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        temperature: float = 0.11,
        min_p: float = 0.1,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> str | Generator[str, None, None]:
        search_results: list[SearchResult] = self.rag.search(
            query, filters=filters, top_k=top_k
        )
        formatted_rag_data = "\n".join([result.format() for result in search_results])
        ai_output = self.ai.run(
            query,
            rag_data=formatted_rag_data,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            min_p=min_p,
            history=history,
        )

        if stream_output:
            yield from ai_output
        else:
            generated_text = ""
            for new_text in ai_output:
                generated_text += f" {new_text}"
            return generated_text

    def batch_run(
        self,
        queries: list[str],
        max_new_tokens: int = 500,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
    ) -> list[str]:
        raise NotImplementedError("Batch run not implemented yet")
