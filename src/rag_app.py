import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface_api import HuggingFaceEmbeddingAPI
import chromadb


# Инициализация настроек LLM и Embeddings через Hugging Face API
# Убедитесь, что переменная окружения HF_TOKEN установлена
def get_llm_and_embedder():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN не найден в переменных окружения")

    # Используем бесплатную API точку доступа (или свою, если есть подписка)
    # Модель для генерации ответов
    llm = HuggingFaceInferenceAPI(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        token=hf_token
    )

    # Модель для эмбеддингов
    embed_model = HuggingFaceEmbeddingAPI(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        token=hf_token
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model


def build_rag_engine(data_path="src/data.txt"):
    """
    Строит RAG движок: загружает документы, создает векторное хранилище и индекс.
    """
    llm, embed_model = get_llm_and_embedder()

    # Инициализация ChromaDB (в памяти для тестов, можно persistent)
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Загрузка документов
    # Создаем временную папку с файлом, если нужно, или читаем напрямую
    from llama_index.core import Document
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    documents = [Document(text=text)]

    # Создание индекса
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        embed_model=embed_model
    )

    # Возвращаем движок запросов
    query_engine = index.as_query_engine(similarity_top_k=2)
    return query_engine


def get_response(query_engine, question):
    """Получает ответ от RAG системы."""
    response = query_engine.query(question)
    return str(response), [node.node.text for node in response.source_nodes]