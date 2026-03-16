import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.rag_app import build_rag_engine, get_response
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
import json


@pytest.fixture
def mock_embedding_vector():
    """Фикстура с вектором эмбеддинга (384 измерения для all-MiniLM-L6-v2)."""
    return np.zeros(384).tolist()


def test_rag_evaluation(mock_embedding_vector):
    """Тестирует RAG систему с помощью Ragas с мокированием эмбеддингов."""

    # Загрузка тестовых данных
    with open("tests/goldens.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    questions = [item["question"] for item in test_data]
    ground_truths = [[item["ground_truth"]] for item in test_data]

    # ✅ Мокируем вызовы эмбеддингов
    with patch(
            'llama_index.embeddings.huggingface_api.base.HuggingFaceInferenceAPIEmbedding._aget_query_embedding',
            return_value=mock_embedding_vector
    ) as mock_query_embed, patch(
        'llama_index.embeddings.huggingface_api.base.HuggingFaceInferenceAPIEmbedding._aget_text_embedding',
        return_value=mock_embedding_vector
    ) as mock_text_embed:
        # Сбор ответов
        answers = []
        retrieved_contexts = []

        print("--- Генерация ответов моделью ---")
        query_engine = build_rag_engine()

        for question in questions:
            ans, contexts = get_response(query_engine, question)
            answers.append(ans)
            retrieved_contexts.append(contexts)

        # Проверка что моки были вызваны
        assert mock_query_embed.called, "Эмбеддинги запросов не были вызваны"
        assert mock_text_embed.called, "Эмбеддинги текстов не были вызваны"

    # Подготовка данных для Ragas
    data_samples = {
        "question": questions,
        "answer": answers,
        "contexts": retrieved_contexts,
        "ground_truth": ground_truths
    }

    dataset = Dataset.from_dict(data_samples)

    # Оценка
    print("--- Оценка метрик Ragas ---")
    score = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )

    print(score)
    score_df = score.to_pandas()
    score_df.to_json("ragas_results.json", orient="records", indent=2)

    # ✅ Проверка пороговых значений
    assert score["faithfulness"] > 0.5, f"Faithfulness {score['faithfulness']} ниже порога 0.5"
    assert score["answer_relevancy"] > 0.5, f"Answer relevancy {score['answer_relevancy']} ниже порога 0.5"
    assert score["context_precision"] > 0.5, f"Context precision {score['context_precision']} ниже порога 0.5"
    assert score["context_recall"] > 0.5, f"Context recall {score['context_recall']} ниже порога 0.5"