import datetime
import heapq
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from haystack import Document
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from tqdm.auto import tqdm

from mtg_ai.cards import MTGDatabase
from mtg_ai.constants import PathLike
from mtg_ai.utils import is_tqdm_disabled

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    card_name: str
    content: Optional[str]
    score: Optional[float]
    retriever_type: str
    metadata: dict[str, Any]

    def format(self) -> str:
        if not self.content:
            raise ValueError(f"No content found for search result {self.card_name}")

        metadata_lines = []

        for line in self.content.splitlines():
            # if "Card Name:" in line and "side" in self.metadata:
            # # card_name = self.card_name
            # names = line.split(" // ")
            # if self.metadata["side"] == "a":
            #     card_name = names[0]
            # elif self.metadata["side"] == "b":
            #     card_name = names[1]
            # line = f"Card Name: {card_name}"
            metadata_lines.append(line)

        logger.debug(f"Mana Cost: {self.metadata.get('mana_cost', '')}")

        if "mana_cost" in self.metadata and not self.metadata["mana_cost"].isspace():
            metadata_lines.append(f"Mana Cost: {self.metadata['mana_cost']}")

        if "cmc" in self.metadata:
            metadata_lines.append(
                f"Coverted Mana Cost (cmc): {int(self.metadata['cmc'])}"
            )

        if (
            "color_identity" in self.metadata
            and not self.metadata["color_identity"].isspace()
        ):
            metadata_lines.append(f"Color Identity: {self.metadata['color_identity']}")

        if "rarity" in self.metadata and not self.metadata["rarity"].isspace():
            metadata_lines.append(f"Rarity: {self.metadata['rarity']}")

        if "loyalty" in self.metadata and self.metadata["loyalty"]:
            metadata_lines.append(f"Loyalty: {self.metadata['loyalty']}")
        if metadata_lines:
            metadata_lines.append("\n")
        metadata_content = "\n".join(metadata_lines)

        formatted_content = f"Relevancy Score: {self.score:.2f}\n{metadata_content}"
        return formatted_content

    def __str__(self):
        return f"{self.card_name} ({self.score:.2f})"


@dataclass
class SearchResults:
    results: list[SearchResult]

    def format(self) -> str:
        return "\n".join([r.format() for r in self.results])


class MTGRAGSearchSystem:
    def __init__(
        self,
        database: MTGDatabase,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        document_cache_dir: PathLike = Path("./mtg_ai_rag_cache"),
        index_on_init: bool = True,
    ):
        self._is_initialized = False
        logger.info("Initializing MTG Search System")

        self.document_cache_dir = Path(document_cache_dir)
        self.database = database

        # initialize document store
        self.document_store = InMemoryDocumentStore()
        logger.debug("Document Store initialized")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # initialize embedders
        self.document_embedder = SentenceTransformersDocumentEmbedder(
            model=embedding_model, progress_bar=not is_tqdm_disabled()
        )
        self.text_embedder = SentenceTransformersTextEmbedder(
            model=embedding_model, progress_bar=not is_tqdm_disabled()
        )
        self.document_embedder.warm_up()
        self.text_embedder.warm_up()

        logger.debug(f"Embedders initialized with model: {embedding_model}")

        # initialize retrievers
        self.bm25_retriever = InMemoryBM25Retriever(document_store=self.document_store)
        self.embedding_retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store
        )
        logger.debug("Retrievers initialized")

        if index_on_init:
            self.index_cards()

        logger.debug("MTG Search System initialization complete")

    def _build_documents(self) -> list[Document]:
        logger.info("Building Card Documents")
        # database.df["text"] = database.df.text.fillna(" ")
        logger.debug(f"Building Documents for {len(self.database.df)} cards")
        documents = []
        for _, row in tqdm(
            self.database.df.iterrows(),
            desc="Building Documents",
            disable=is_tqdm_disabled(),
        ):
            card_dict = row.dropna().to_dict()
            content_parts = []

            card_name = card_dict["name"]

            if "side" in card_dict and " // " in card_name:
                names = card_name.split(" // ")
                if card_dict["side"] == "a":
                    card_name = names[0]
                elif card_dict["side"] == "b":
                    card_name = names[1]
            content_parts.append(f"Card Name: {card_dict['name']}")

            if "manaCost" in card_dict:
                content_parts.append(f"Mana Cost: {card_dict['manaCost']}")

            if "type" in card_dict:
                content_parts.append(f"Type: {card_dict['type']}")

            if "text" in card_dict:
                content_parts.append(f"Card Text: {card_dict['text']}")

            if "power" in card_dict and "toughness" in card_dict:
                content_parts.append(
                    f"Power/Toughness: {card_dict['power']}/{card_dict['toughness']}"
                )
            elif "power" in card_dict:
                content_parts.append(f"Power: {card_dict['power']}")
            elif "toughness" in card_dict:
                content_parts.append(f"Toughness: {card_dict['toughness']}")

            if "loyalty" in card_dict:
                content_parts.append(f"Loyalty: {card_dict['loyalty']}")

            if "side" in card_dict:
                content_parts.append(f"Side: {card_dict['side']}")

            content = "\n".join(content_parts)

            metadata: dict[str, Any] = {
                "name": card_dict["name"],
                "mana_cost": card_dict.get("manaCost", ""),
                "cmc": float(card_dict.get("cmc", "")),
                "color_identity": card_dict.get("colorIdentity", ""),
                "type": card_dict.get("type", ""),
                "power": card_dict.get("power", ""),
                "toughness": card_dict.get("toughness", ""),
                "rarity": card_dict.get("rarity", ""),
                "loyalty": card_dict.get("loyalty", ""),
                "side": card_dict.get("side", ""),
            }

            document = Document(
                content=content, id=str(card_dict["uuid"]), meta=metadata
            )
            documents.append(document)

        logger.info(f"Built {len(documents)} documents")
        return documents

    def index_cards(self) -> int:
        # try:
        #     self.load_document_store(self.document_cache_dir)
        #     return self.document_store.count_documents()
        # except FileNotFoundError:
        #     pass

        logger.info(f"Indexing {len(self.database.df)} cards")

        documents = self._build_documents()
        logger.debug("Generating document embeddings")

        documents_with_embeddings = self.document_embedder.run(documents=documents)[
            "documents"
        ]

        logger.debug("Writing documents to document store")
        self.document_store.write_documents(documents_with_embeddings)

        logger.info(f"Successfully indexed {len(documents)} cards")
        # self.save_document_store("./mtg_ai_rag_cache")
        self._is_initialized = True
        return len(documents)

    def save_document_store(self, dir_path: PathLike):
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        path = path.joinpath("mtg_card_document_store.json")

        if path.exists():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = path.rename(
                f"{path.parent}/{path.stem}_{timestamp}.{path.suffix}.backup"
            )
            logger.info(f"Created backup of existing document store at {backup_path}")

        try:
            self.document_store.save_to_disk(str(path))
        except Exception as e:
            logger.error(f"Failed to save document store to {path}: {e}")
            backup_files = list(path.parent.glob(f"{path.stem}*.backup"))
            if backup_files:
                logger.info("Backups found. Restoring backup document store")
                backup_files.sort()
                backup_path = backup_files[-1]
                backup_path.rename(str(path))
            raise

    def load_document_store(self, dir_path: PathLike):
        path = Path(dir_path)
        path = path.joinpath("mtg_card_document_store.json")
        if not path.exists():
            raise FileNotFoundError(f"Document store file not found at {path}")

        logger.info(f"Loading document store from {path}")
        self.document_store = InMemoryDocumentStore.load_from_disk(str(path))
        logger.info(
            "Document store loaded with "
            f"{self.document_store.count_documents()} documents"
        )

    def search(
        self, query: str, filters: Optional[dict[str, Any]] = None, top_k: int = 10
    ) -> SearchResults:
        """
        Search for cards with optional filters

        Args:
            query: Search query
            filters: Optional filters for metadata
                (e.g., {"colors": ["R"], "type": "Instant"})
            top_k: Number of results to return

        Returns:
            List of SearchResult objects

        Example filters:
        {
            "colors": ["R", "G"],
            "type": "Creature",
            "cmc": {"$lte": 3},
            "rarity": "rare"
        }
        """
        if not self._is_initialized:
            self.index_cards()
        # Get BM25 results

        logger.info(
            f"Performing hybrid search for query: '{query}' with filters: {filters}"
        )
        logger.debug("Executing BM25 Search")
        bm25_results = self.bm25_retriever.run(
            query=query,
            filters=filters,
        )
        logger.debug(f"BM25 returned {len(bm25_results['documents'])} results")

        # Get embedding results
        query_embedding = self.text_embedder.run(text=query)
        embedding_results = self.embedding_retriever.run(
            query_embedding=query_embedding["embedding"],
            filters=filters,
        )

        # Combine and deduplicate results
        all_results = []
        seen_ids = set()

        def read_results(
            results: list[Document],
            seen_ids: set[str],
            retriever_type: str,
            all_results: list[SearchResult],
        ) -> None:
            for doc in results:
                if doc.id in seen_ids:
                    continue
                all_results.append(
                    SearchResult(
                        card_name=doc.meta["name"],
                        content=doc.content,
                        score=doc.score,
                        retriever_type=retriever_type,
                        metadata=doc.meta,
                    )
                )
                seen_ids.add(doc.id)

        read_results(
            bm25_results["documents"],
            retriever_type="bm25",
            seen_ids=seen_ids,
            all_results=all_results,
        )
        read_results(
            embedding_results["documents"],
            retriever_type="embeddings",
            seen_ids=seen_ids,
            all_results=all_results,
        )

        # Get top k results
        top_results: list[SearchResult] = heapq.nlargest(
            top_k, all_results, key=lambda x: x.score
        )

        logger.info(f"Hybrid search complete. Returning {len(top_results)} results")
        logger.debug("Top results:" + ", ".join([r.card_name for r in top_results]))

        result = SearchResults(results=top_results)
        return result
