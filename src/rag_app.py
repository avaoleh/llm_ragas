import os
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
# ✅ ПРАВИЛЬНЫЕ ИМЕНА КЛАССОВ
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface_api import HuggingFaceEmbedding
import chromadb


def get_llm_and_embedder():
    """Инициализация LLM и эмбеддингов через Hugging Face Inference API."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN не найден в переменных окружения")

    # ✅ LLM через Inference API
    llm = HuggingFaceInferenceAPI(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        token=hf_token,
        temperature=0.1,
        max_new_tokens=512
    )

    # ✅ Embeddings модель (ПРАВИЛЬНОЕ ИМЯ: HuggingFaceAPIEmbedding)
    embed_model = HuggingFaceAPIEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        token=hf_token
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
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
        embed_model=embed_model
    )

    return index.as_query_engine(similarity_top_k=2)


def get_response(query_engine, question):
    """Получает ответ и контекст от RAG системы."""
    response = query_engine.query(question)
    return str(response), [node.node.text for node in response.source_nodes]