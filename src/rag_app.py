import os
import asyncio
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
import chromadb
import nest_asyncio
import threading

# Применяем nest_asyncio в самом начале для решения проблем с event loop
nest_asyncio.apply()


def get_llm_and_embedder():
    """Инициализация LLM и эмбеддингов через Hugging Face Inference API."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN не найден в переменных окружения")

    # ✅ LLM через Inference API
    print("🌐 Использование LLM через Hugging Face Inference API...")
    llm = HuggingFaceInferenceAPI(
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        token=hf_token,
        temperature=0.1,
        max_new_tokens=512
    )

    # ✅ Embeddings модель через Inference API
    print("🌐 Использование эмбеддингов через Hugging Face Inference API...")
    embed_model = HuggingFaceInferenceAPIEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        token=hf_token
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model


def build_rag_engine(data_path="src/data.txt"):
    """Строит RAG движок с ChromaDB."""
    llm, embed_model = get_llm_and_embedder()

    # Проверяем наличие активного event loop
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # Если нет запущенного цикла, создаем новый
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Создаем временную Chroma для тестов (не требует сохранения на диск)
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("rag_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Загружаем документ из файла
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    documents = [Document(text=text)]

    # Создаем индекс с векторами
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
        embed_model=embed_model
    )

    # Возвращаем движок для запросов, ищем 2 наиболее похожих фрагмента
    return index.as_query_engine(similarity_top_k=2)


def get_response(query_engine, question):
    """
    Получает ответ и контекст от RAG системы.
    Использует отдельный поток с собственным event loop для избежания проблем с asyncio.
    """

    # Переменные для хранения результата и ошибки
    result = None
    error = None

    def target():
        """Функция, выполняемая в отдельном потоке."""
        nonlocal result, error
        try:
            # Создаем новый event loop для этого потока
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Применяем nest_asyncio к этому loop
            nest_asyncio.apply(loop)

            # Выполняем запрос к RAG системе
            result = query_engine.query(question)

            # Корректно закрываем event loop
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        except Exception as e:
            error = e

    # Запускаем поток и ждем его завершения
    thread = threading.Thread(target=target)
    thread.start()
    thread.join()

    # Если произошла ошибка, пробрасываем её
    if error:
        raise error

    # Возвращаем ответ и контекст (текст найденных фрагментов)
    return str(result), [node.node.text for node in result.source_nodes]