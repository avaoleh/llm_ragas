import os
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore

# LLM импорты
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.huggingface_local import HuggingFaceLLM

# Embeddings импорты
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import chromadb


def get_llm_and_embedder():
    """
    Инициализация LLM и эмбеддингов с поддержкой переключения API/Локальные.

    Переменные окружения:
    - USE_LOCAL_LLM: "true" для локальной модели, "false" для API
    - USE_LOCAL_EMBEDDINGS: "true" для локальных эмбеддингов, "false" для API
    - HF_TOKEN: токен Hugging Face (обязателен для API режима)
    """
    hf_token = os.getenv("HF_TOKEN")
    use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

    # Проверка токена для API режима
    if not hf_token and (not use_local_llm or not use_local_embeddings):
        raise ValueError("HF_TOKEN не найден в переменных окружения (требуется для API режима)")

    # ✅ Инициализация LLM
    if use_local_llm:
        print("🔄 Использование локальной LLM для тестов...")
        llm = HuggingFaceLLM(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_new_tokens=512,
            device_map="auto",
            generate_kwargs={"temperature": 0.1, "do_sample": False},
        )
    else:
        print("🌐 Использование LLM через Hugging Face Inference API...")
        llm = HuggingFaceInferenceAPI(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",  # ✅ Стабильная версия
            token=hf_token,
            temperature=0.1,
            max_new_tokens=512,
        )

    # ✅ Инициализация Embeddings
    if use_local_embeddings:
        print("🔄 Использование локальных эмбеддингов для тестов...")
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./cache",
            trust_remote_code=True,
        )
    else:
        print("🌐 Использование эмбеддингов через Hugging Face Inference API...")
        embed_model = HuggingFaceInferenceAPIEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            token=hf_token,
        )

    Settings.llm = llm
    Settings.embed_model = embed_model

    print(f"✅ LLM: {'Локальная' if use_local_llm else 'API'}")
    print(f"✅ Embeddings: {'Локальные' if use_local_embeddings else 'API'}")

    return llm, embed_model


def build_rag_engine(data_path="src/data.txt"):
    """Строит RAG движок с ChromaDB."""
    llm, embed_model = get_llm_and_embedder()

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
    response = query_engine.query(question)
    return str(response), [node.node.text for node in response.source_nodes]