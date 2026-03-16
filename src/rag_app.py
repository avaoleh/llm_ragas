import os
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.embeddings import BaseEmbedding
from huggingface_hub import InferenceClient
import chromadb
from typing import List, Any, Optional
import requests
import numpy as np


class SyncHuggingFaceEmbedding(BaseEmbedding):
    """
    Полностью синхронная версия эмбеддингов Hugging Face.
    Не использует asyncio, использует прямой HTTP-запрос.
    """

    def __init__(
            self,
            model_name: str,
            token: str,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.token = token
        # Используем простой HTTP-клиент без asyncio
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self.headers = {"Authorization": f"Bearer {token}"}

    def _get_embedding(self, text: str) -> List[float]:
        """Получение эмбеддинга через прямой HTTP-запрос."""
        try:
            # Простой синхронный HTTP-запрос
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": text, "options": {"wait_for_model": True}},
                timeout=30
            )
            response.raise_for_status()

            # Парсим результат
            result = response.json()

            # Преобразуем в список float
            if isinstance(result, list):
                # Если результат - список чисел (эмбеддинг)
                if result and isinstance(result[0], (int, float)):
                    return [float(x) for x in result]
                # Если результат - список списков (может быть для нескольких предложений)
                elif result and isinstance(result[0], list):
                    # Берём первый (или усредняем, но для одного предложения берём первый)
                    return [float(x) for x in result[0]]

            # Если не удалось распознать формат, пробуем преобразовать через numpy
            try:
                import numpy as np
                arr = np.array(result)
                return arr.flatten().tolist()
            except:
                raise ValueError(f"Не удалось преобразовать результат в эмбеддинг: {type(result)}")

        except Exception as e:
            print(f"Ошибка при получении эмбеддинга: {e}")
            # В случае ошибки возвращаем нулевой вектор (но лучше пробросить исключение)
            raise

    def _get_query_embedding(self, query: str) -> List[float]:
        """Получение эмбеддинга для запроса."""
        return self._get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Получение эмбеддинга для текста."""
        return self._get_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Асинхронная версия - не используется, но должна быть реализована."""
        # Просто вызываем синхронную версию
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Асинхронная версия - не используется, но должна быть реализована."""
        return self._get_text_embedding(text)


def get_llm_and_embedder():
    """Инициализация LLM и эмбеддингов через Hugging Face Inference API."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN не найден в переменных окружения")

    # ✅ LLM через Inference API (пока оставляем как есть)
    print("🌐 Использование LLM через Hugging Face Inference API...")
    llm = HuggingFaceInferenceAPI(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        token=hf_token,
        temperature=0.1,
        max_new_tokens=512
    )

    # ✅ Используем полностью синхронную версию эмбеддингов
    print("🌐 Использование синхронных эмбеддингов через прямой HTTP API...")
    embed_model = SyncHuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        token=hf_token
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model


def build_rag_engine(data_path="src/data.txt"):
    """Строит RAG движок с ChromaDB."""
    llm, embed_model = get_llm_and_embedder()

    # Создаем временную Chroma для тестов
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("rag_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Загружаем документ из файла
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Разбиваем документ на части для лучшего поиска
    paragraphs = text.split('\n\n')
    documents = [Document(text=p) for p in paragraphs if p.strip()]

    # Создаем индекс с векторами
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        embed_model=embed_model,
        show_progress=False
    )

    # Возвращаем движок для запросов
    return index.as_query_engine(similarity_top_k=2)


def get_response(query_engine, question):
    """
    Получает ответ и контекст от RAG системы.
    Использует только синхронные вызовы, никакого asyncio.
    """

    # Просто выполняем запрос синхронно
    response = query_engine.query(question)
    return str(response), [node.node.text for node in response.source_nodes]