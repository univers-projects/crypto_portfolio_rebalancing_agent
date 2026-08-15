"""ChromaDB-сховище бази знань та tool `knowledge_search`.

Ключова властивість agentic RAG: цей tool — звичайний інструмент у списку.
Він не викликається автоматично на кожен запит; LLM сам вирішує, чи потрібен
йому концептуальний контекст (напр. "what is max drawdown?"), чи достатньо
числових tools (напр. "show current portfolio").
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, cast

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from langchain_core.tools import StructuredTool

from app.config import get_settings
from app.domain.errors import KnowledgeBaseError
from app.rag.documents import KNOWLEDGE_DOCUMENTS
from app.tools.base import ToolPayload, success, tool_contract
from app.tools.schemas import KnowledgeSearchInput

logger = logging.getLogger(__name__)

COLLECTION_NAME = "portfolio_knowledge"

# Телеметрія Chroma вимкнена нижче через ChromaSettings, але її клієнт усе одно
# пише помилки сумісності в лог. У інтерактивному режимі це шум, тому глушимо.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)


def _embedding_function() -> ONNXMiniLM_L6_V2:
    """Локальна ONNX-модель ембедингів, явно прив'язана до CPU.

    CoreML execution provider на частині macOS-машин падає під час inference,
    тому провайдер фіксуємо на CPU: повільніше, але стабільно і відтворювано.
    """
    return ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def get_collection() -> Any:
    """Отримати (і за потреби наповнити) колекцію ChromaDB.

    Використовується вбудована ONNX-модель ембедингів — жодних зовнішніх
    API-викликів, тому індексація працює офлайн і детерміновано.
    """
    settings = get_settings()
    try:
        client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=cast(Any, _embedding_function()),
        )
    except Exception as error:  # noqa: BLE001 — Chroma кидає різні типи винятків
        raise KnowledgeBaseError(f"Не вдалося ініціалізувати ChromaDB: {error}") from error

    # Індексуємо лише якщо колекція порожня або неповна (ідемпотентно)
    if collection.count() < len(KNOWLEDGE_DOCUMENTS):
        _index_documents(collection)
    return collection


def _index_documents(collection: Any) -> None:
    """Записати всі документи бази знань у колекцію."""
    try:
        collection.upsert(
            ids=[document.doc_id for document in KNOWLEDGE_DOCUMENTS],
            documents=[document.content for document in KNOWLEDGE_DOCUMENTS],
            metadatas=[
                {"title": document.title, "topic": document.topic}
                for document in KNOWLEDGE_DOCUMENTS
            ],
        )
        logger.info("Проіндексовано %d документів бази знань", len(KNOWLEDGE_DOCUMENTS))
    except Exception as error:  # noqa: BLE001
        raise KnowledgeBaseError(f"Не вдалося проіндексувати документи: {error}") from error


@tool_contract("knowledge_search")
def _knowledge_search(**kwargs: Any) -> ToolPayload:
    params = KnowledgeSearchInput(**kwargs)
    collection = get_collection()

    try:
        raw = collection.query(
            query_texts=[params.query],
            n_results=min(params.top_k, len(KNOWLEDGE_DOCUMENTS)),
        )
    except Exception as error:  # noqa: BLE001
        raise KnowledgeBaseError(f"Помилка пошуку в базі знань: {error}") from error

    ids = raw.get("ids", [[]])[0]
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    if not ids:
        return success({"query": params.query, "results": [], "result_count": 0})

    results = [
        {
            "doc_id": doc_id,
            "title": metadata.get("title", ""),
            "topic": metadata.get("topic", ""),
            # Cosine distance -> зрозуміліша для LLM оцінка релевантності
            "relevance": round(max(0.0, 1.0 - float(distance)), 4),
            "content": content,
        }
        for doc_id, content, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        )
    ]

    return success(
        {"query": params.query, "results": results, "result_count": len(results)}
    )


knowledge_search = StructuredTool.from_function(
    func=_knowledge_search,
    name="knowledge_search",
    description=(
        "Шукає у внутрішній базі знань з портфельного ризик-менеджменту: risk management, "
        "diversification, volatility, max drawdown, liquidity, crypto market risk, "
        "rebalancing principles, turnover та transaction costs, position sizing, "
        "stablecoin risk, asset correlation. "
        "Викликай ЛИШЕ тоді, коли потрібне концептуальне або методологічне пояснення "
        "(наприклад, як інтерпретувати метрику чи коли ребаланс виправданий). "
        "НЕ викликай для отримання цін, метрик або складу портфеля — для цього є "
        "окремі числові tools."
    ),
    args_schema=KnowledgeSearchInput,
)
