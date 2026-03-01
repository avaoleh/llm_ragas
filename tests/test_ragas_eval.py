import os
import sys
import json
import pytest
from datasets import Dataset

# ✅ Исправленные импорты Ragas
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_recall

# Добавляем src в путь (резервный вариант, если pytest.ini не сработает)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_app import build_rag_engine, get_response  # type: ignore

# Пороговые значения для Quality Gates
THRESHOLD_FAITHFULNESS = 0.7
THRESHOLD_ANSWER_RELEVANCY = 0.7
THRESHOLD_CONTEXT_RECALL = 0.7


@pytest.fixture(scope="module")
def query_engine():
    """Инициализируем RAG движок один раз для всех тестов."""
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN не установлен. Пропускаем тесты.")
    return build_rag_engine()


@pytest.fixture(scope="module")
def goldens():
    """Загружаем золотые данные."""
    with open("tests/goldens.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_rag_evaluation(query_engine, goldens):
    """Основной тест: генерирует ответы, оценивает через Ragas и проверяет пороги."""
    questions, answers, contexts, ground_truths = [], [], [], []

    print("\n--- Генерация ответов моделью ---")
    for item in goldens:
        question = item["question"]
        gt_answer = item["answer"]
        ans, retrieved_contexts = get_response(query_engine, question)

        questions.append(question)
        answers.append(ans)
        contexts.append(retrieved_contexts)
        ground_truths.append([gt_answer])  # Ragas ожидает список строк

    # Формирование датасета для Ragas
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("\n--- Запуск оценки Ragas ---")
    result = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        raise_exceptions=False
    )

    # Вывод и сохранение результатов
    print(result.to_pandas())

    results_json = result.to_dict()
    with open("ragas_results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Результаты сохранены в ragas_results.json")

    # Извлечение средних значений
    avg_faithfulness = result['faithfulness']
    avg_relevancy = result['answer_relevancy']
    avg_recall = result['context_recall']

    print(f"\n📊 Faithfulness: {avg_faithfulness:.4f}")
    print(f"📊 Answer Relevancy: {avg_relevancy:.4f}")
    print(f"📊 Context Recall: {avg_recall:.4f}")

    # ✅ Quality Gates Asserts
    assert avg_faithfulness >= THRESHOLD_FAITHFULNESS, \
        f"❌ Faithfulness ({avg_faithfulness:.4f}) < {THRESHOLD_FAITHFULNESS}"
    assert avg_relevancy >= THRESHOLD_ANSWER_RELEVANCY, \
        f"❌ Answer Relevancy ({avg_relevancy:.4f}) < {THRESHOLD_ANSWER_RELEVANCY}"
    assert avg_recall >= THRESHOLD_CONTEXT_RECALL, \
        f"❌ Context Recall ({avg_recall:.4f}) < {THRESHOLD_CONTEXT_RECALL}"

    print("\n✅ Все метрики прошли пороги качества!")