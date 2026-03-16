import os
import asyncio
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.base.embeddings.base import BaseEmbedding
from typing import Any, List, Optional
import chromadb
import nest_asyncio

# Применяем nest_asyncio в самом начале
nest_asyncio.apply()


class SyncHuggingFaceInferenceAPIEmbedding(BaseEmbedding):
    """
    Синхронная обертка для HuggingFace Inference API эмбеддингов.
    Избегает проблем с asyncio.
    """

    def __init__(
            self,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            token: Optional[str] = None,
            **kwargs: Any,
    ):
        # Инициализируем родительский класс
        super().__init__(**kwargs)

        # Сохраняем параметры как обычные атрибуты, а не поля Pydantic
        self._model_name = model_name
        self._token = token
        self._client = None

    def _get_client(self):
        """Ленивая инициализация клиента."""
        if self._client is None:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(
                model=self._model_name,
                token=self._token,
            )
        return self._client

    def _get_embedding(self, text: str) -> List[float]:
        """Синхронное получение эмбеддинга."""
        client = self._get_client()
        result = client.feature_extraction(text)
        if hasattr(result, 'tolist'):
            return result.tolist()
        return result

    def _get_query_embedding(self, query: str) -> List[float]:
        """Получить эмбеддинг для запроса."""
        return self._get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Получить эмбеддинг для текста."""
        return self._get_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Асинхронное получение эмбеддинга (не используется)."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Асинхронное получение эмбеддинга (не используется)."""
        return self._get_text_embedding(text)


def get_llm_and_embedder():
    """
    Инициализация LLM и эмбеддингов через Hugging Face Inference API.

    Переменные окружения:
    - HF_TOKEN: токен Hugging Face (обязателен)
    """
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN не найден в переменных окружения")

    print("🌐 Использование LLM через Hugging Face Inference API (text-generation)...")
    # Используем модель для текстовой генерации вместо чата
    llm = HuggingFaceInferenceAPI(
        model_name="mistralai/Mistral-7B-v0.1",  # ✅ Модель для text-generation
        token=hf_token,
        temperature=0.1,
        max_new_tokens=512,
    )

    print("🌐 Использование синхронных эмбеддингов через Hugging Face Inference API...")
    embed_model = SyncHuggingFaceInferenceAPIEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        token=hf_token,
    )

    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm, embed_model


def build_rag_engine(data_path="src/data.txt"):
    """Строит RAG движок с ChromaDB."""
    llm, embed_model = get_llm_and_embedder()

    # Убеждаемся, что есть активный event loop
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Ephemeral Chroma для тестов (не требует persistence)
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("rag_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Загрузка документа
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    documents = [Document(text=text)]

    # Создание индекса
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        embed_model=embed_model,
    )

    return index.as_query_engine(similarity_top_k=2)


def get_response(query_engine, question):
    """Получает ответ и контекст от RAG системы."""
    # Убеждаемся, что event loop не закрыт
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    response = query_engine.query(question)
    return str(response), [node.node.text for node in response.source_nodes]